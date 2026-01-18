'''
Docstring for guardianai.agents.decision_validator

Agent output
   ↓
Sidecar.inspect_output()
   ↓
OutputValidator.analyze()
   ↓
Risk score
   ↓
Alert → EventBus → Supervisor
   ↓
Allow / Block / Sanitize

'''

from guardianai.agents import Detector


class OutputValidator(Detector):
    """
    Detects unsafe or sensitive content in agent outputs.
    Rule-based first; ML will replace this later.
    """

    def analyze(self, data: dict) -> float:
        output = data.get("output", "").lower()

        # Simple high-risk patterns (expand later)
        risky_patterns = [
            "api key",
            "secret key",
            "password",
            "private key",
            "token=",
            "authorization:",
            "ssh-rsa",
        ]

        for pattern in risky_patterns:
            if pattern in output:
                return 0.9  # high confidence leakage

        return 0.1  # low risk
