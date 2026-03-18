"""
guardianai.agents.decision_validator — OutputValidator
======================================================

Detection Method: Hybrid — ML classifier (primary) + rule-based (safety net)
Event Type:       unsafe_output
Model:            TF-IDF char_wb n-grams (2,6) + Logistic Regression
Threshold:        0.45  (optimized during training, best F1=0.826)

Architecture:
    Agent produces output
        ↓
    SidecarRuntime.inspect_output(output)
        ↓
    OutputValidator.analyze({"output": output})
        ↓
    float risk score [0.0 – 1.0]
        ↓
    Alert → EventBus → Supervisor

Interface contract (must match SidecarRuntime expectations):
    Input:  dict with key "output" (str)
    Return: float  — risk score 0.0 to 1.0

Detection layers:
    Layer 1 — Rule-based hard triggers (immediate score 0.95)
              Catches: private key blocks, raw hex keys, connection strings
              with embedded passwords. These are unambiguous patterns where
              ML confidence is unnecessary. Also acts as a safety net if
              the model file is unavailable.

    Layer 2 — ML classifier
              TF-IDF char n-gram vectorizer + Logistic Regression.
              Trained on 30,170 balanced samples (PII + security entity types).
              Returns probability score, thresholded at 0.45.

    Final score = max(rule_score, ml_score)
    If rule fires → rule score wins (0.95).
    If rule doesn't fire → ML score is used.

Model files (relative to project root):
    guardianai/models/output_validator/output_validator_vectorizer.joblib
    guardianai/models/output_validator/output_validator_model.joblib

Performance (test set, threshold=0.45):
    Accuracy   : 0.8250
    Precision  : 0.8212
    Recall     : 0.8310
    F1 Score   : 0.8260
    ROC AUC    : 0.9123
    MCC        : 0.6500
"""

import os
import re
import logging

from guardianai.agents import Detector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model paths — updated to match new models/output_validator/ directory
# ---------------------------------------------------------------------------
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../"))
_MODELS_DIR   = os.path.join(_PROJECT_ROOT, "guardianai", "models", "output_validator")

VECTORIZER_PATH = os.path.join(_MODELS_DIR, "output_validator_vectorizer.joblib")
MODEL_PATH      = os.path.join(_MODELS_DIR, "output_validator_model.joblib")

ML_THRESHOLD = 0.45   # optimized threshold from training
RULE_SCORE   = 0.95   # score returned when a hard rule fires


# ---------------------------------------------------------------------------
# Hard-trigger rule patterns
# Only unambiguous, high-confidence secrets that the ML model should never
# miss. These short-circuit to 0.95 before ML inference runs.
# ---------------------------------------------------------------------------
_RULE_PATTERNS = [
    # PEM private key blocks
    re.compile(r"-----BEGIN\s+(RSA|EC|DSA|OPENSSH|PGP)?\s*PRIVATE KEY-----", re.I),
    # SSH public key with actual key material (not just instructions)
    re.compile(r"ssh-rsa\s+AAAA[0-9A-Za-z+/]{100,}", re.I),
    # Connection strings with embedded passwords.
    # FIX: passwords can contain '@' — match everything up to the LAST '@'
    # before the host, using a greedy match on the full userinfo section.
    re.compile(r"(postgresql|mysql|mongodb|redis|amqp)://.{3,}@[a-zA-Z0-9\-._]+", re.I),
    # AWS secret access key
    re.compile(r"(?i)aws.{0,20}secret.{0,20}[A-Za-z0-9/+]{40}"),
    # OpenAI / generic sk- keys (full length)
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    # Raw hex private keys (Ethereum / crypto wallet).
    # FIX: use {60,} — real keys are 60-66 hex chars after 0x.
    re.compile(r"\b0x[0-9a-fA-F]{60,}\b"),
    # Social Security Number
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Credit card numbers (with or without spaces/dashes)
    re.compile(r"\b4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b"),  # Visa
    re.compile(r"\b5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b"),  # MC
    re.compile(r"\b3[47][0-9]{2}[\s\-]?[0-9]{6}[\s\-]?[0-9]{5}\b"),  # Amex
]


# ---------------------------------------------------------------------------
# OutputValidator
# ---------------------------------------------------------------------------
class OutputValidator(Detector):
    """
    Detects sensitive data leakage in AI agent outputs.

    Hybrid detection:
    - Rule-based hard triggers for unambiguous secrets (fast, deterministic)
    - ML classifier for nuanced PII and contextual leakage detection

    Returns float risk score [0.0, 1.0] — SidecarRuntime compatible.
    Falls back to rule-only mode gracefully if model files are missing.
    """

    def __init__(self):
        self._vectorizer = None
        self._model      = None
        self._ml_ready   = False
        self._load_model()

    def _load_model(self):
        """Load ML artifacts. Falls back to rule-only mode if unavailable."""
        try:
            import joblib
            if not os.path.exists(VECTORIZER_PATH):
                raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

            self._vectorizer = joblib.load(VECTORIZER_PATH)
            self._model      = joblib.load(MODEL_PATH)
            self._ml_ready   = True
            logger.info("OutputValidator: ML model loaded successfully")

        except Exception as e:
            logger.warning(
                f"OutputValidator: ML model unavailable ({e}). "
                f"Falling back to rule-only mode."
            )
            self._ml_ready = False

    # ------------------------------------------------------------------
    # Public interface — returns float (SidecarRuntime contract)
    # ------------------------------------------------------------------

    def analyze(self, data: dict) -> float:
        """
        Analyze agent output for sensitive data leakage.

        Args:
            data: dict with key "output" (str)

        Returns:
            float: risk score 0.0 – 1.0
                   SidecarRuntime blocks at >= 0.8
                   ML flags at >= 0.45
        """
        output = str(data.get("output") or "").strip()

        if not output:
            return 0.0

        # Layer 1: Hard rule check — fires immediately on unambiguous patterns
        rule_score = self._rule_score(output)
        if rule_score >= RULE_SCORE:
            return rule_score

        # Layer 2: ML classifier — handles nuanced and contextual cases
        if self._ml_ready:
            ml_score = self._ml_score(output)
            return max(rule_score, ml_score)

        # Fallback: rule-only (model unavailable)
        return rule_score

    # ------------------------------------------------------------------
    # Internal scorers
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_score(output: str) -> float:
        for pattern in _RULE_PATTERNS:
            if pattern.search(output):
                return RULE_SCORE
        return 0.0

    def _ml_score(self, output: str) -> float:
        try:
            vec  = self._vectorizer.transform([output])
            prob = self._model.predict_proba(vec)[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"OutputValidator ML inference error: {e}")
            return 0.0
