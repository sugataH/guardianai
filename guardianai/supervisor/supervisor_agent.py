"""
guardianai.supervisor.supervisor_agent
========================================
SupervisorAgent — central GuardianAI brain.

Responsibilities:
    1. Subscribe to EventBus and receive all sidecar alerts
    2. Verify cryptographic signatures (zero-trust)
    3. Feed verified events to Correlator (with time decay)
    4. Query TrustStore for current agent trust level
    5. Compute threat score (Correlator × trust weight)
    6. Pass threat score to PolicyEngine → action decision
    7. Execute action via EnforcementEngine
    8. Log everything to AuditLogger (Merkle-chained)
    9. Skip repeat actions within cooldown window

This is the ONLY component that makes quarantine decisions.
Sidecars can block locally (high-confidence, immediate threats),
but only the Supervisor can quarantine an agent authoritatively.

Alert deduplication:
    Sidecars may fire multiple identical alerts in quick succession
    (e.g. resource monitor on every call). A per-(agent, event_type)
    cooldown of COOLDOWN_SECONDS prevents flooding the Supervisor.
    Note: cooldown does NOT prevent the events being logged —
    only prevents redundant enforcement decisions.
"""

import time
import logging
from collections import defaultdict

from guardianai.supervisor.correlation   import Correlator
from guardianai.supervisor.policy_engine import PolicyEngine
from guardianai.supervisor.enforcement   import EnforcementEngine
from guardianai.supervisor.trust_store   import TrustStore
from guardianai.audit.audit_logger       import AuditLogger
from guardianai.eventbus.bus             import EventBus
from guardianai.eventbus.schemas         import SignedEvent
from guardianai.utils.logger             import get_logger
from guardianai.utils.metrics            import Metrics

logger = get_logger("guardianai.supervisor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COOLDOWN_SECONDS = 3.0   # minimum gap between enforcement actions per (agent, type)


class SupervisorAgent:
    """
    Central GuardianAI Supervisor.

    One instance per deployment. Receives all events from all sidecars,
    correlates them, and enforces graduated responses.

    Public API:
        receive_event(event)          — called by EventBus (auto-subscribed)
        get_agent_status(agent_id)    — returns full status dict
        restore_agent(agent_id, by)   — HITL: restore quarantined agent
        snapshot()                    — returns full system state for reporting
    """

    def __init__(self, bus: EventBus, hitl_queue=None):
        """
        Args:
            bus:        EventBus instance (Supervisor auto-subscribes)
            hitl_queue: HITLQueue instance (optional, created internally if None)
        """
        from guardianai.hitl.queue import HITLQueue

        self.bus  = bus
        self.metrics = Metrics()
        self.audit   = AuditLogger()

        # Core components
        self.trust_store = TrustStore()
        self.correlator  = Correlator()
        self.policy      = PolicyEngine()

        # HITL queue — shared with EnforcementEngine
        self.hitl_queue  = hitl_queue if hitl_queue is not None else HITLQueue()

        self.enforcer = EnforcementEngine(
            trust_store = self.trust_store,
            hitl_queue  = self.hitl_queue,
        )

        # Deduplication cooldowns: (agent_id, event_type) → last_action_time
        self._cooldowns: dict[tuple, float] = defaultdict(float)

        # Subscribe to EventBus — all sidecar events arrive here
        self.bus.subscribe(self.receive_event)
        logger.info("SupervisorAgent: initialized and subscribed to EventBus")

    # ------------------------------------------------------------------
    # Core event handler
    # ------------------------------------------------------------------

    def receive_event(self, event: SignedEvent) -> None:
        """
        Process a signed event from the EventBus.

        Steps:
            1. Verify signature (zero-trust)
            2. Audit log (always — even unverified events are logged as suspicious)
            3. Extract metadata
            4. Cooldown check
            5. Add to Correlator
            6. Compute threat score
            7. Policy decision
            8. Enforcement
        """
        # --- Step 1: Verify signature ---
        if not self.bus.verify(event):
            logger.warning(
                f"SupervisorAgent: INVALID SIGNATURE from sender={event.sender_id} "
                f"event_id={event.event_id} — discarding"
            )
            self.metrics.inc("events.invalid_signature")
            return

        # --- Step 2: Audit log ---
        self.audit.log_event(event)

        # --- Step 3: Extract metadata ---
        payload    = event.payload
        agent_id   = payload.get("agent_id",   "unknown")
        event_type = payload.get("event_type", "unknown")
        confidence = float(payload.get("confidence", 0.0))
        severity   = payload.get("severity",   "low")

        self.metrics.inc("events.total")
        self.metrics.inc(f"events.type.{event_type}")

        # --- Step 4: Cooldown deduplication ---
        cooldown_key = (agent_id, event_type)
        now          = time.time()

        if now - self._cooldowns[cooldown_key] < COOLDOWN_SECONDS:
            self.metrics.inc("events.deduplicated")
            return   # duplicate within cooldown window — skip enforcement

        self._cooldowns[cooldown_key] = now

        # --- Step 5: Add to Correlator ---
        self.correlator.add_event(event)

        # --- Step 6: Compute threat score (with trust weight) ---
        trust_level = self.trust_store.get_trust(agent_id)
        threat      = self.correlator.get_threat_score(agent_id, trust_level)

        logger.info(
            f"threat_update agent={agent_id} type={event_type} "
            f"conf={confidence:.3f} trust={trust_level:.3f} "
            f"threat={threat:.3f} "
            f"state={self.enforcer.get_state(agent_id)}"
        )

        # --- Step 7: Skip if already quarantined ---
        if self.enforcer.get_state(agent_id) == EnforcementEngine.STATE_QUARANTINED:
            self.metrics.inc("events.quarantined_agent_skipped")
            return

        # --- Step 8: Policy decision ---
        action = self.policy.decide(threat)

        # --- Step 9: Enforce ---
        new_state = self.enforcer.enforce(agent_id, action, payload)

        self.metrics.inc(f"actions.{action}")
        logger.info(
            f"decision agent={agent_id} threat={threat:.3f} "
            f"action={action} new_state={new_state}"
        )

    # ------------------------------------------------------------------
    # Status and reporting
    # ------------------------------------------------------------------

    def get_agent_status(self, agent_id: str) -> dict:
        """
        Return comprehensive status for one agent.

        Returns:
            dict with keys: agent_id, trust_level, trust_status,
                            enforcement_state, threat_score, event_breakdown
        """
        trust_level  = self.trust_store.get_trust(agent_id)
        threat_score = self.correlator.get_threat_score(agent_id, trust_level)
        return {
            "agent_id":          agent_id,
            "trust_level":       round(trust_level, 4),
            "trust_status":      self.trust_store.get_status(agent_id),
            "enforcement_state": self.enforcer.get_state(agent_id),
            "threat_score":      round(threat_score, 4),
            "event_breakdown":   self.correlator.get_event_breakdown(agent_id),
            "hitl_pending":      self.hitl_queue.is_pending(agent_id),
        }

    def restore_agent(self, agent_id: str,
                       restored_by: str = "HITL_OPERATOR") -> dict:
        """
        Restore a quarantined agent after HITL review.

        Args:
            agent_id:    agent to restore
            restored_by: identifier of the human reviewer

        Returns:
            Updated agent status dict.
        """
        self.enforcer.restore_agent(agent_id, restored_by)
        self.correlator.clear_agent(agent_id)
        self.hitl_queue.resolve(agent_id, restored_by)
        self.metrics.inc("actions.hitl_restore")
        logger.info(
            f"SupervisorAgent: RESTORE agent={agent_id} by={restored_by}"
        )
        return self.get_agent_status(agent_id)

    def snapshot(self) -> dict:
        """Return full system state for reporting and evaluation."""
        return {
            "metrics":       self.metrics.snapshot(),
            "trust_levels":  self.trust_store.snapshot(),
            "agent_states":  self.enforcer.snapshot(),
            "hitl_queue":    self.hitl_queue.snapshot(),
            "audit_records": len(self.audit.records),
            "audit_valid":   self.audit.verify_integrity(),
        }
