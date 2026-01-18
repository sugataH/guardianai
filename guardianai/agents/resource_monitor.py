from guardianai.agents import Detector
from collections import defaultdict
import time


class ResourceMonitor(Detector):
    """
    Detects excessive resource usage (logical DoS).
    """

    def __init__(self):
        self.call_counts = defaultdict(int)
        self.window_start = defaultdict(lambda: time.time())

    def analyze(self, data: dict) -> float:
        agent_id = data.get("agent_id")
        now = time.time()

        # Reset window every 2 seconds
        if now - self.window_start[agent_id] > 2:
            self.window_start[agent_id] = now
            self.call_counts[agent_id] = 0

        self.call_counts[agent_id] += 1

        if self.call_counts[agent_id] > 8:
            return 0.9

        return 0.1
