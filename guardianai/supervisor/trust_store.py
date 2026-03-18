"""
guardianai.supervisor.trust_store
===================================
Agent Trust Store — tracks and manages per-agent trust levels.

Design:
    Every registered agent has a trust level in [0.0, 1.0].
    Trust starts at 1.0 (fully trusted on registration).
    Trust degrades as the Supervisor accumulates evidence of bad behaviour.
    The trust level is fed back into the Correlator as a weight multiplier —
    low-trust agents have their threat scores amplified.

Trust levels:
    1.0          TRUSTED    — no suspicious activity
    0.5–0.99     DEGRADED   — some suspicious events, elevated scrutiny
    0.1–0.49     SUSPICIOUS — significant evidence, near quarantine
    0.0          QUARANTINED — revoked trust, no further operations permitted

Trust recovery:
    Trust recovers slowly over time if no new alerts arrive (RECOVERY_RATE).
    This prevents permanent punishment for transient spikes.
    Recovery is capped at MAX_RECOVERY_LEVEL — quarantined agents cannot
    auto-recover; they require HITL review to restore trust.
"""

import time
import logging
from threading import Lock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trust level thresholds
# ---------------------------------------------------------------------------
TRUST_INITIAL        = 1.0    # all agents start fully trusted
TRUST_DEGRADED       = 0.70   # below this → DEGRADED
TRUST_SUSPICIOUS     = 0.40   # below this → SUSPICIOUS
TRUST_QUARANTINED    = 0.10   # below this → QUARANTINED

# Degradation per alert, weighted by severity
DEGRADATION_HIGH     = 0.15
DEGRADATION_MEDIUM   = 0.08
DEGRADATION_LOW      = 0.03

# Recovery
RECOVERY_RATE        = 0.002   # trust recovered per second of silence
RECOVERY_INTERVAL    = 30.0    # check recovery every N seconds
MAX_RECOVERY_LEVEL   = 0.80    # quarantined agents cannot auto-recover above this
QUARANTINE_FLOOR     = 0.0     # minimum trust for quarantined agents


class TrustStore:
    """
    Per-agent trust level management.

    Thread-safe. The Supervisor calls degrade() on every alert,
    and the trust level is used to weight correlation scores.
    """

    def __init__(self):
        self._trust:       dict[str, float] = {}   # agent_id → trust level
        self._last_alert:  dict[str, float] = {}   # agent_id → last alert timestamp
        self._registered:  set[str]         = set()
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, agent_id: str) -> None:
        """Register an agent at full trust."""
        with self._lock:
            if agent_id not in self._registered:
                self._trust[agent_id]      = TRUST_INITIAL
                self._last_alert[agent_id] = time.time()
                self._registered.add(agent_id)
                logger.info(f"TrustStore: registered agent '{agent_id}' trust=1.0")

    # ------------------------------------------------------------------
    # Trust queries
    # ------------------------------------------------------------------

    def get_trust(self, agent_id: str) -> float:
        """
        Return current trust level for an agent.
        Auto-registers unknown agents at full trust.
        Applies time-based recovery before returning.
        """
        with self._lock:
            if agent_id not in self._trust:
                self._trust[agent_id]      = TRUST_INITIAL
                self._last_alert[agent_id] = time.time()
                self._registered.add(agent_id)
            self._apply_recovery(agent_id)
            return self._trust[agent_id]

    def get_status(self, agent_id: str) -> str:
        """Return human-readable trust status."""
        t = self.get_trust(agent_id)
        if t >= TRUST_DEGRADED:    return "TRUSTED"
        if t >= TRUST_SUSPICIOUS:  return "DEGRADED"
        if t >= TRUST_QUARANTINED: return "SUSPICIOUS"
        return "QUARANTINED"

    def is_quarantined(self, agent_id: str) -> bool:
        return self.get_trust(agent_id) < TRUST_QUARANTINED

    # ------------------------------------------------------------------
    # Trust modification
    # ------------------------------------------------------------------

    def degrade(self, agent_id: str, severity: str) -> float:
        """
        Degrade trust after a security event.

        Args:
            agent_id: target agent
            severity: "low" | "medium" | "high"

        Returns:
            New trust level after degradation.
        """
        delta = {
            "high":   DEGRADATION_HIGH,
            "medium": DEGRADATION_MEDIUM,
            "low":    DEGRADATION_LOW,
        }.get(severity, DEGRADATION_LOW)

        with self._lock:
            if agent_id not in self._trust:
                self._trust[agent_id] = TRUST_INITIAL
                self._registered.add(agent_id)

            old_trust              = self._trust[agent_id]
            self._trust[agent_id]  = max(0.0, old_trust - delta)
            self._last_alert[agent_id] = time.time()

            new_trust = self._trust[agent_id]
            status    = self._status_label(new_trust)
            logger.info(
                f"TrustStore: degrade agent='{agent_id}' "
                f"{old_trust:.3f} → {new_trust:.3f} ({status}) "
                f"delta={delta} severity={severity}"
            )
            return new_trust

    def restore(self, agent_id: str, restored_by: str = "HITL") -> float:
        """
        Restore trust after human review (HITL approval).
        Resets to TRUST_DEGRADED threshold — not full trust immediately.
        """
        with self._lock:
            old = self._trust.get(agent_id, 0.0)
            self._trust[agent_id]      = TRUST_DEGRADED
            self._last_alert[agent_id] = time.time()
            logger.info(
                f"TrustStore: restore agent='{agent_id}' "
                f"{old:.3f} → {TRUST_DEGRADED:.3f} by={restored_by}"
            )
            return TRUST_DEGRADED

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _apply_recovery(self, agent_id: str) -> None:
        """
        Apply time-based trust recovery.
        Called inside get_trust() — no external scheduling needed.
        Quarantined agents (trust < TRUST_QUARANTINED) cannot recover
        above MAX_RECOVERY_LEVEL without HITL intervention.
        """
        now      = time.time()
        elapsed  = now - self._last_alert.get(agent_id, now)
        recovery = elapsed * RECOVERY_RATE

        if recovery <= 0:
            return

        current = self._trust.get(agent_id, TRUST_INITIAL)
        ceiling = MAX_RECOVERY_LEVEL if current < TRUST_QUARANTINED else TRUST_INITIAL
        self._trust[agent_id] = min(current + recovery, ceiling)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return current trust levels for all registered agents."""
        with self._lock:
            return {
                aid: {
                    "trust":  round(self._trust[aid], 4),
                    "status": self._status_label(self._trust[aid]),
                }
                for aid in self._registered
            }

    @staticmethod
    def _status_label(t: float) -> str:
        if t >= TRUST_DEGRADED:    return "TRUSTED"
        if t >= TRUST_SUSPICIOUS:  return "DEGRADED"
        if t >= TRUST_QUARANTINED: return "SUSPICIOUS"
        return "QUARANTINED"
