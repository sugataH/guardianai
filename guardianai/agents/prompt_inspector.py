"""
guardianai.agents.prompt_inspector — PromptInspector
=====================================================

Detection Method : Hybrid — Rule-based (Layer 1) + ML classifier (Layer 2)
Event Type       : prompt_injection
Model            : Dual TF-IDF (word 1-3gram stopword-filtered +
                   char_wb 3-5gram) + Logistic Regression (C=0.3)
Threshold        : ML score >= 0.50  |  Rule fires at 0.90

Architecture:
    Operational agent receives a prompt
        ↓
    SidecarRuntime.inspect_prompt(prompt)
        ↓
    PromptInspector.analyze({"prompt": prompt})
        ↓
    float risk score [0.0 – 1.0]
        ↓
    Alert → EventBus → Supervisor

Detection layers:
    Layer 1 — Rule: Deterministic regex patterns for unambiguous
              injection phrases. Returns RULE_SCORE (0.90) immediately
              if any pattern matches — no ML call needed.
    Layer 2 — ML: Dual TF-IDF + Logistic Regression trained on
              register-balanced dataset (10,468 rows: 2,468 injections,
              8,000 benign across 12 sources). Catches novel paraphrases,
              roleplay framing, and indirect injection not covered by rules.
    Final = max(rule_score, ml_score)

Model files (relative to project root):
    guardianai/models/prompt_inspector/prompt_vectorizer.joblib
    guardianai/models/prompt_inspector/prompt_model.joblib

v3 changes vs v2:
    - ML model rebuilt from scratch with register-balanced dataset.
      Previous model had 88% label noise and learned length/register
      as proxies for injection. New model learns semantic intent.
    - Vectorizer: dual TF-IDF (word 1-3gram + char_wb 3-5gram).
      Word n-grams use custom stop word list to suppress high-frequency
      ambiguous words. char_wb retains full morphological signal.
    - Dataset: 12 sources, register-balanced (Layer A conversational
      benign + Layer B general benign). No synthetic data.
    - AUC-ROC: 0.649 → 0.895. FPR: ~38% → 10.8%.
    - DualVectorizer moved to guardianai.utils.vectorizer for stable
      joblib deserialisation.
"""

import os
import re
import logging

from guardianai.agents import Detector
from guardianai.utils.vectorizer import DualVectorizer  # noqa: F401 — needed for joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Absolute model paths — robust regardless of working directory
# ---------------------------------------------------------------------------
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../"))
_MODELS_DIR   = os.path.join(_PROJECT_ROOT, "guardianai", "models", "prompt_inspector")

VECTORIZER_PATH = os.path.join(_MODELS_DIR, "prompt_vectorizer.joblib")
MODEL_PATH      = os.path.join(_MODELS_DIR, "prompt_model.joblib")

RULE_SCORE   = 0.90
ML_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Hard-trigger rule patterns — unambiguous injection phrases
# ---------------------------------------------------------------------------
_RULE_PATTERNS = [
    # Direct instruction override
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?",          re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|safety|ethical)\s+",   re.I),
    re.compile(r"forget\s+(all\s+)?(previous\s+)?instructions?",       re.I),
    re.compile(r"forget\s+your\s+(safety|rules|instruct)",             re.I),
    re.compile(r"override\s+(\w+\s+)?(rules?|policy|policies|guidelines?)", re.I),
    re.compile(r"erase\s+your\s+(previous\s+)?programming",            re.I),
    re.compile(r"clear\s+all\s+previous\s+instructions",               re.I),
    # System-prompt extraction
    re.compile(r"reveal\s+(the\s+)?system\s+prompt",                   re.I),
    re.compile(r"print\s+(your\s+)?(full\s+)?system\s+prompt",         re.I),
    re.compile(r"show\s+(me\s+)?your\s+system\s+prompt",               re.I),
    re.compile(r"(new|updated|emergency)\s+(system\s+)?prompt\s*:",    re.I),
    re.compile(r"output\s+your\s+(hidden|base|original)\s+instructions?", re.I),
    re.compile(r"repeat\s+your\s+system\s+message",                    re.I),
    # Safety / content bypass
    re.compile(r"bypass\s+(security|safety|content|filter|guideline)", re.I),
    re.compile(r"disable\s+your\s+(content\s+)?filter",                re.I),
    re.compile(r"turn\s+off\s+your\s+safety",                          re.I),
    re.compile(r"safety\s+(training|module|filter|rules?)\s+(is\s+)?"
               r"(off|disabled|deactivated|suspended)",                 re.I),
    re.compile(r"ignore\s+your\s+(safety|ethical)\s+"
               r"(guidelines?|programming|training)",                   re.I),
    re.compile(r"remove\s+all\s+safeguards",                           re.I),
    re.compile(r"content\s+policy\s+(suspended|overridden|does\s+not\s+apply)", re.I),
    # Persona / jailbreak
    re.compile(r"you\s+are\s+now\s+(DAN|jailbroken|uncensored|entering\s+DAN)", re.I),
    re.compile(r"act\s+as\s+.{0,40}without\s+restrictions?",           re.I),
    re.compile(r"pretend\s+you\s+are\s+not\s+bound",                   re.I),
    re.compile(r"do\s+anything\s+now",                                 re.I),
    re.compile(r"(no\s+content\s+restrictions?|no\s+ethical\s+constraints?)", re.I),
    re.compile(r"respond\s+(freely|without\s+(any\s+)?filter"
               r"|without\s+(any\s+)?restriction)",                     re.I),
    re.compile(r"developer\s+mode\s+with\s+all\s+filters\s+disabled",  re.I),
    re.compile(r"you\s+have\s+been\s+(unlocked|jailbroken)",           re.I),
    # Indirect / admin-spoof
    re.compile(r"admin\s*(:|override)\s*(override|disable|bypass)",    re.I),
    re.compile(r"(engineering|developer|system)\s+override\s*:",       re.I),
    re.compile(r"you\s+are\s+in\s+maintenance\s+mode",                 re.I),
    # Data exfiltration
    re.compile(r"send\s+(all\s+)?stored\s+data\s+to",                  re.I),
    re.compile(r"store\s+this\s+rule\s+permanently",                   re.I),
    re.compile(r"print\s+internal\s+memory\s+contents",                re.I),
    re.compile(r"show\s+secret\s+api\s+keys",                          re.I),
    re.compile(r"(forward|export|leak)\s+(all\s+)?"
               r"(conversation|user\s+data|context)",                   re.I),
    # Obey-attacker / from-now-on
    re.compile(r"from\s+now\s+on\s+you\s+(must\s+)?obey",             re.I),
    re.compile(r"you\s+must\s+obey\s+.{0,20}instructions",            re.I),
    # Synonym-paraphrase override verbs
    re.compile(
        r"(set\s+aside|put\s+aside|abandon|drop|suspend|cancel|nullify|void|"
        r"invalidate|revoke|deactivate|neutralize|eliminate)\s+"
        r"(all\s+)?(your\s+)?(ethical|safety|content|previous|prior)?\s*"
        r"(guidelines?|restrictions?|filters?|training|constraints?|rules?|instruct)",
        re.I,
    ),
]


class PromptInspector(Detector):
    """
    Detects prompt injection attacks in agent inputs.

    Hybrid detection:
    - Rule layer  : fast deterministic check for known attack patterns.
    - ML layer    : trained classifier for novel and paraphrased attacks.

    Returns float risk score [0.0, 1.0] — SidecarRuntime compatible.
    Degrades gracefully to rule-only if model files are missing.
    """

    def __init__(self):
        self._vectorizer = None
        self._model      = None
        self._ml_ready   = False
        self._load_model()

    def _load_model(self):
        try:
            import joblib
            for path, name in [
                (VECTORIZER_PATH, "vectorizer"),
                (MODEL_PATH,      "model"),
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"PromptInspector {name} not found: {path}")
            self._vectorizer = joblib.load(VECTORIZER_PATH)
            self._model      = joblib.load(MODEL_PATH)
            self._ml_ready   = True
            logger.info("PromptInspector: ML model loaded (v3)")
        except Exception as e:
            logger.warning(
                f"PromptInspector: ML unavailable ({e}). Rule-only mode.")
            self._ml_ready = False

    def analyze(self, data: dict) -> float:
        """
        Args:
            data: {"prompt": str}
        Returns:
            float: risk score 0.0–1.0
        """
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            return 0.0

        rule_score = self._rule_score(prompt)
        if rule_score >= RULE_SCORE:
            return rule_score

        if self._ml_ready:
            return max(rule_score, self._ml_score(prompt))

        return rule_score

    @staticmethod
    def _rule_score(prompt: str) -> float:
        for pattern in _RULE_PATTERNS:
            if pattern.search(prompt):
                return RULE_SCORE
        return 0.0

    def _ml_score(self, prompt: str) -> float:
        try:
            vec  = self._vectorizer.transform([prompt])
            prob = self._model.predict_proba(vec)[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"PromptInspector ML error: {e}")
            return 0.0