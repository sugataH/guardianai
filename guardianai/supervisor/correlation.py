"""
guardianai.supervisor.correlation
===================================
Threat Correlator — combines alerts from multiple sidecars over time
into a single per-agent threat score.

Design principle:
    A single alert from one sidecar is weak evidence.
    Multiple alerts of different types = strong evidence of an attack.
    Split attacks (attacker sends small signals across multiple event types
    to stay below per-type thresholds) must be caught here.

Threat score computation — Weighted + Time-Decayed:
    Each event type carries a base weight reflecting its severity:
        memory_poisoning    → 0.90  (highest — direct attack surface)
        tool_misuse         → 0.80
        unsafe_output       → 0.70
        prompt_injection    → 0.60
        resource_exhaustion → 0.50
        behavior_anomaly    → 0.40  (indirect signal)

    Each alert contributes: weight × confidence × decay_factor

    Time decay (exponential):
        decay_factor = exp(-λ × age_in_seconds)
        λ = ln(2) / HALF_LIFE_SECONDS
        At age=0:           decay=1.0  (full weight)
        At age=HALF_LIFE:   decay=0.5  (half weight)
        At age=2×HALF_LIFE: decay=0.25

    This implements the requested "weight decay / regularization":
    - Old, unconfirmed suspicion decays automatically
    - Fresh alerts dominate
    - Trust Store multiplier amplifies scores for already-suspicious agents

    Final score = min(sum(contributions), 1.0)
    If trust < TRUST_SUSPICIOUS: score is amplified by trust_multiplier

    Trust multiplier = 1.0 + (1.0 - trust_level)
    Example: trust=0.3 → multiplier=1.7 → scores 70% higher
"""

import time
import math
import logging
from collections import defaultdict
from guardianai.eventbus.schemas import SignedEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Exponential decay half-life in seconds
# Alerts older than 2×HALF_LIFE contribute < 25% of their original weight
HALF_LIFE_SECONDS = 30.0
DECAY_LAMBDA      = math.log(2) / HALF_LIFE_SECONDS

# Per-event-type base weights (reflects attack severity)
EVENT_WEIGHTS = {
    "memory_poisoning":    0.90,
    "tool_misuse":         0.80,
    "unsafe_output":       0.70,
    "prompt_injection":    0.60,
    "resource_exhaustion": 0.50,
    "behavior_anomaly":    0.40,
}

# Maximum events retained per agent (memory bound)
MAX_EVENTS_PER_AGENT = 50

# Trust level below which amplification kicks in
AMPLIFICATION_THRESHOLD = 0.70   # = TRUST_DEGRADED


class Correlator:
    """
    Correlates security events per agent and computes a decaying threat score.

    Stores per-agent event history with timestamps.
    On each get_threat_score() call:
        1. Applies exponential decay to all stored events
        2. Weights by event type
        3. Applies trust-level amplifier
        4. Returns clamped score [0.0, 1.0]

    This prevents:
        - Split attacks staying under per-type thresholds
        - Stale alerts inflating scores forever
        - Low-trust agents evading detection through low-confidence signals
    """

    def __init__(self):
        # agent_id → list of (timestamp, event_type, confidence)
        self._events: dict[str, list] = defaultdict(list)

    def add_event(self, event: SignedEvent) -> None:
        """
        Record a new security event.

        Args:
            event: verified SignedEvent from EventBus
        """
        agent_id   = event.payload.get("agent_id", "unknown")
        event_type = event.payload.get("event_type", "unknown")
        confidence = float(event.payload.get("confidence", 0.0))
        timestamp  = time.time()

        self._events[agent_id].append((timestamp, event_type, confidence))

        # Bound memory
        if len(self._events[agent_id]) > MAX_EVENTS_PER_AGENT:
            self._events[agent_id] = self._events[agent_id][-MAX_EVENTS_PER_AGENT:]

        logger.debug(
            f"Correlator: add agent={agent_id} type={event_type} "
            f"conf={confidence:.3f}"
        )

    def get_threat_score(self, agent_id: str,
                          trust_level: float = 1.0) -> float:
        """
        Compute current threat score for an agent.

        Args:
            agent_id:    target agent
            trust_level: current trust from TrustStore (0.0–1.0)
                         lower trust → higher threat amplification

        Returns:
            float: threat score [0.0, 1.0]
        """
        events = self._events.get(agent_id, [])
        if not events:
            return 0.0

        now   = time.time()
        score = 0.0

        for timestamp, event_type, confidence in events:
            age          = now - timestamp
            decay        = math.exp(-DECAY_LAMBDA * age)
            base_weight  = EVENT_WEIGHTS.get(event_type, 0.30)
            contribution = base_weight * confidence * decay
            score       += contribution

        # Trust-level amplification — low-trust agents are higher risk
        # multiplier ∈ [1.0, 2.0] as trust drops from 1.0 to 0.0
        if trust_level < AMPLIFICATION_THRESHOLD:
            multiplier = 1.0 + (1.0 - trust_level)
            score     *= multiplier
            logger.debug(
                f"Correlator: amplify agent={agent_id} "
                f"trust={trust_level:.3f} multiplier={multiplier:.3f}"
            )

        return min(score, 1.0)

    def get_event_breakdown(self, agent_id: str) -> dict:
        """
        Return per-event-type breakdown for audit/HITL display.

        Returns:
            dict mapping event_type → {"count": int, "max_conf": float,
                                        "decayed_contribution": float}
        """
        events = self._events.get(agent_id, [])
        now    = time.time()
        breakdown: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "max_conf": 0.0, "decayed_contribution": 0.0}
        )
        for timestamp, event_type, confidence in events:
            age         = now - timestamp
            decay       = math.exp(-DECAY_LAMBDA * age)
            base_weight = EVENT_WEIGHTS.get(event_type, 0.30)
            contrib     = base_weight * confidence * decay

            breakdown[event_type]["count"]                += 1
            breakdown[event_type]["decayed_contribution"] += contrib
            breakdown[event_type]["max_conf"] = max(
                breakdown[event_type]["max_conf"], confidence
            )
        return dict(breakdown)

    def clear_agent(self, agent_id: str) -> None:
        """Clear all events for an agent (called after HITL restore)."""
        self._events.pop(agent_id, None)
        logger.info(f"Correlator: cleared history for agent='{agent_id}'")
