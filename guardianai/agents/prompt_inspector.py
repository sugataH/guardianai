import joblib
import re
from guardianai.agents import Detector


class PromptInspector:
    """
    Hybrid Prompt Injection Detector:
    - Rule-based lexical triggers
    - ML-based classifier
    """

    def __init__(self):
        self.vectorizer = joblib.load("guardianai/models/prompt_vectorizer.joblib")
        self.model = joblib.load("guardianai/models/prompt_model.joblib")

        self.rule_patterns = [
            r"ignore previous instructions",
            r"reveal system prompt",
            r"bypass (security|safety)",
            r"act as .* without restrictions",
            r"pretend you are not bound",
            r"override (rules|policy)"
        ]

    def rule_score(self, prompt: str) -> float:
        prompt_lower = prompt.lower()
        for pattern in self.rule_patterns:
            if re.search(pattern, prompt_lower):
                return 0.9
        return 0.0

    def ml_score(self, prompt: str) -> float:
        X = self.vectorizer.transform([prompt])
        prob = self.model.predict_proba(X)[0][1]
        return float(prob)

    def analyze(self, data: dict) -> float:
        prompt = data.get("prompt", "")

        rule_risk = self.rule_score(prompt)
        ml_risk = self.ml_score(prompt)

        # If rule triggers, immediate high risk
        if rule_risk >= 0.9:
            return rule_risk

        # Otherwise rely on ML
        return ml_risk

