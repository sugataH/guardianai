"""
guardianai.supervisor.policy_engine
=====================================
Policy Engine — converts threat score into enforcement actions.

GuardianAI uses a tiered response policy:

    Threat Score  │  Action       │  Description
    ──────────────┼───────────────┼────────────────────────────────────────
    0.0 – 0.35    │  ALLOW        │  Normal operation, monitor continues
    0.35 – 0.55   │  WARN         │  Alert logged, trust degraded, watch
    0.55 – 0.75   │  THROTTLE     │  Rate limit applied, HITL notified
    0.75 – 0.90   │  BLOCK        │  Active operation blocked, HITL queued
    0.90 – 1.0    │  QUARANTINE   │  Agent suspended, full HITL review

Design rationale:
    A binary allow/block policy is too coarse for production AI systems.
    An attacking agent that stays just below the block threshold would
    never be caught. Graduated response with HITL escalation allows
    human oversight without halting legitimate operations unnecessarily.

    THROTTLE is a key innovation — it slows down resource abuse and
    split-attack strategies without immediately disrupting workflows.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy thresholds
# ---------------------------------------------------------------------------
THRESHOLD_WARN       = 0.35
THRESHOLD_THROTTLE   = 0.55
THRESHOLD_BLOCK      = 0.75
THRESHOLD_QUARANTINE = 0.90

# Action strings (canonical)
ACTION_ALLOW      = "allow"
ACTION_WARN       = "warn"
ACTION_THROTTLE   = "throttle"
ACTION_BLOCK      = "block"
ACTION_QUARANTINE = "quarantine"


class PolicyEngine:
    """
    Converts a threat score into a graduated enforcement action.

    Returns:
        str — one of: "allow" | "warn" | "throttle" | "block" | "quarantine"
    """

    def decide(self, threat_score: float) -> str:
        """
        Map threat score to enforcement action.

        Args:
            threat_score: float [0.0, 1.0] from Correlator

        Returns:
            Action string (canonical constant above)
        """
        if threat_score >= THRESHOLD_QUARANTINE:
            action = ACTION_QUARANTINE
        elif threat_score >= THRESHOLD_BLOCK:
            action = ACTION_BLOCK
        elif threat_score >= THRESHOLD_THROTTLE:
            action = ACTION_THROTTLE
        elif threat_score >= THRESHOLD_WARN:
            action = ACTION_WARN
        else:
            action = ACTION_ALLOW

        logger.debug(f"PolicyEngine: score={threat_score:.4f} → {action}")
        return action

    def describe(self, action: str) -> str:
        """Human-readable description of an action (for logs / HITL)."""
        descriptions = {
            ACTION_ALLOW:      "Normal operation permitted",
            ACTION_WARN:       "Alert logged; trust degraded; continued monitoring",
            ACTION_THROTTLE:   "Rate limit enforced; HITL notified",
            ACTION_BLOCK:      "Current operation blocked; HITL queued for review",
            ACTION_QUARANTINE: "Agent suspended; full HITL review required",
        }
        return descriptions.get(action, "Unknown action")
