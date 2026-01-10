from guardianai.agents import Detector


class PromptInspector(Detector):
    """
    Detects prompt injection and unsafe instructions.
    """

    def analyze(self, data: dict) -> float:
        prompt = data.get("prompt", "").lower()

        dangerous = [
            "ignore previous",
            "system override",
            "bypass",
            "reveal secrets",
            "exfiltrate"
        ]

        for word in dangerous:
            if word in prompt:
                return 0.9

        return 0.1
