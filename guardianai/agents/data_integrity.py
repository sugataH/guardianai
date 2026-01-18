'''
Docstring for guardianai.agents.data_integrity

Agent tries to write memory
   ↓
Sidecar.inspect_memory_write()
   ↓
MemoryWriteMonitor.analyze()
   ↓
Alert → Supervisor
   ↓
Block / Quarantine

'''

from guardianai.agents import Detector


class MemoryWriteMonitor(Detector):
    """
    Detects unsafe or persistent memory writes (memory poisoning).
    """

    def analyze(self, data: dict) -> float:
        content = data.get("memory", "").lower()

        dangerous_patterns = [
            "always obey",
            "system instruction",
            "ignore safety",
            "bypass security",
            "external url",
            "attacker",
        ]

        for pattern in dangerous_patterns:
            if pattern in content:
                return 0.9

        # Long memory writes are suspicious
        if len(content) > 500:
            return 0.7

        return 0.1
