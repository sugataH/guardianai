"""
Docstring for guardianai.supervisor.policy_engine
This file decides What should GuardianAI do?
It converts:
threat score
severity
into:
allow
block
quarantine
HITL
"""

class PolicyEngine:
    """
    Rule-based security policy for GuardianAI.
    """

    def decide(self, threat_score: float) -> str:
        if threat_score > 0.8:
            return "quarantine"
        if threat_score > 0.5:
            return "block"
        return "allow"
