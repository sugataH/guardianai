from abc import ABC, abstractmethod


class Detector(ABC):
    """
    Base class for all GuardianAI detectors.
    """

    @abstractmethod
    def analyze(self, data: dict) -> float:
        """
        Returns risk score between 0 and 1.
        """
        pass
