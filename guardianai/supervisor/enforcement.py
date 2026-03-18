"""
guardianai.supervisor.enforcement
===================================
Enforcement Engine — executes graduated responses decided by PolicyEngine.

Enforcement actions:
    ALLOW      → no-op, monitoring continues
    WARN       → log + trust degrade
    THROTTLE   → log + trust degrade + HITL notification
    BLOCK      → log + trust degrade + HITL queue entry (non-blocking)
    QUARANTINE → log + trust revoke + HITL queue (blocking, agent suspended)

The Enforcement Engine is the bridge between policy decisions and:
    1. TrustStore (trust degradation / revocation)
    2. HITLQueue  (escalation for human review)
    3. AuditLogger (tamper-proof record)

It does NOT raise exceptions for BLOCK/QUARANTINE — those are raised by
the local SidecarRuntime. The Supervisor's enforcement is authoritative
and persistent (survives sidecar restarts).
"""

import logging
from guardianai.supervisor.trust_store import TrustStore
from guardianai.supervisor.policy_engine import (
    ACTION_ALLOW, ACTION_WARN, ACTION_THROTTLE, ACTION_BLOCK, ACTION_QUARANTINE
)

logger = logging.getLogger(__name__)


class EnforcementEngine:
    """
    Executes enforcement actions decided by PolicyEngine.

    Maintains agent state transitions:
        NORMAL → WARNED → THROTTLED → BLOCKED → QUARANTINED

    Agents can only move forward through states unless HITL restores them.
    """

    # Agent states (ordered by severity)
    STATE_NORMAL      = "NORMAL"
    STATE_WARNED      = "WARNED"
    STATE_THROTTLED   = "THROTTLED"
    STATE_BLOCKED     = "BLOCKED"
    STATE_QUARANTINED = "QUARANTINED"

    _STATE_RANK = {
        STATE_NORMAL:      0,
        STATE_WARNED:      1,
        STATE_THROTTLED:   2,
        STATE_BLOCKED:     3,
        STATE_QUARANTINED: 4,
    }

    def __init__(self, trust_store: TrustStore, hitl_queue):
        """
        Args:
            trust_store: TrustStore instance (shared with Supervisor)
            hitl_queue:  HITLQueue instance for escalation
        """
        self._trust  = trust_store
        self._hitl   = hitl_queue
        self._states: dict[str, str] = {}   # agent_id → current state

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enforce(self, agent_id: str, action: str,
                event_payload: dict) -> str:
        """
        Execute the enforcement action for an agent.

        Args:
            agent_id:      target agent
            action:        decision from PolicyEngine
            event_payload: original alert payload (for HITL context)

        Returns:
            New agent state after enforcement.
        """
        current_state = self._states.get(agent_id, self.STATE_NORMAL)

        if action == ACTION_ALLOW:
            return current_state

        if action == ACTION_WARN:
            self._trust.degrade(agent_id, "low")
            new_state = self._advance_state(current_state, self.STATE_WARNED)
            self._states[agent_id] = new_state
            logger.info(f"Enforce: WARN agent={agent_id} state={new_state}")
            return new_state

        if action == ACTION_THROTTLE:
            self._trust.degrade(agent_id, "medium")
            new_state = self._advance_state(current_state, self.STATE_THROTTLED)
            self._states[agent_id] = new_state
            self._hitl.notify(agent_id, action, event_payload)
            logger.warning(f"Enforce: THROTTLE agent={agent_id} state={new_state}")
            return new_state

        if action == ACTION_BLOCK:
            self._trust.degrade(agent_id, "high")
            new_state = self._advance_state(current_state, self.STATE_BLOCKED)
            self._states[agent_id] = new_state
            self._hitl.enqueue(agent_id, action, event_payload)
            logger.warning(f"Enforce: BLOCK agent={agent_id} state={new_state}")
            return new_state

        if action == ACTION_QUARANTINE:
            self._trust.degrade(agent_id, "high")
            self._trust.degrade(agent_id, "high")   # double degrade for quarantine
            self._states[agent_id] = self.STATE_QUARANTINED
            self._hitl.enqueue(agent_id, action, event_payload, urgent=True)
            logger.warning(
                f"Enforce: QUARANTINE agent={agent_id} "
                f"SUSPENDED — awaiting HITL review"
            )
            return self.STATE_QUARANTINED

        return current_state

    def get_state(self, agent_id: str) -> str:
        """Return current enforcement state for an agent."""
        return self._states.get(agent_id, self.STATE_NORMAL)

    def restore_agent(self, agent_id: str, restored_by: str = "HITL") -> str:
        """
        Restore a quarantined agent after HITL approval.
        State returns to NORMAL; trust returns to DEGRADED level.
        """
        self._trust.restore(agent_id, restored_by)
        self._states[agent_id] = self.STATE_NORMAL
        logger.info(
            f"Enforce: RESTORE agent={agent_id} by={restored_by} "
            f"state=NORMAL trust={self._trust.get_trust(agent_id):.3f}"
        )
        return self.STATE_NORMAL

    def snapshot(self) -> dict:
        """Return current states for all tracked agents."""
        return dict(self._states)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _advance_state(self, current: str, target: str) -> str:
        """
        Monotonically advance state — never downgrade.
        An agent that is BLOCKED cannot be moved to WARNED.
        """
        if self._STATE_RANK.get(target, 0) > self._STATE_RANK.get(current, 0):
            return target
        return current
