"""
Docstring for guardianai.agents.behavior_monitor
Agent action happens
   ↓
Sidecar records behavior
   ↓
BehaviorMonitor.analyze()
   ↓
Risk score
   ↓
Alert → Supervisor

"""

from guardianai.agents import Detector
from collections import defaultdict
import time


class BehaviorMonitor(Detector):
    """
    Detects abnormal agent behavior over time.
    """

    def __init__(self):
        self.event_counts = defaultdict(int)
        self.last_seen = defaultdict(float)

    def analyze(self, data: dict) -> float:
        agent_id = data.get("agent_id")
        now = time.time()

        self.event_counts[agent_id] += 1

        last = self.last_seen.get(agent_id, now)
        self.last_seen[agent_id] = now

        # High-frequency behavior in short time window
        if self.event_counts[agent_id] > 5 and (now - last) < 2:
            return 0.9

        return 0.1
