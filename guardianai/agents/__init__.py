"""
guardianai.agents
=================
Base class for all GuardianAI detectors.

Every detector must implement analyze(data: dict) -> float | dict.
SidecarRuntime uses _score() to normalise return values to float.
"""
from abc import ABC, abstractmethod


class Detector(ABC):
    """
    Abstract base for all GuardianAI security detectors.

    Contract:
        analyze(data: dict) → float | dict
        If dict, must contain key "score" (float 0.0–1.0).
    """

    @abstractmethod
    def analyze(self, data: dict):
        """
        Analyse a security event.

        Args:
            data: event-specific dict (see each detector for schema)

        Returns:
            float  OR  dict with at least {"score": float}
            Score is always in [0.0, 1.0] where 1.0 = maximum risk.
        """
        pass
