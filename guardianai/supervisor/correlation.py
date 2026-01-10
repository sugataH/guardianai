"""It combines:
multiple events
from multiple sidecars
across time
This prevents attackers from:
splitting attacks into small pieces
"""

from collections import defaultdict
from typing import List
from guardianai.eventbus.schemas import SignedEvent


class Correlator:
    """
    Groups and scores multiple security events into attack patterns.
    """

    def __init__(self):
        self.event_buffer = defaultdict(list)

    def add_event(self, event: SignedEvent):
        agent = event.payload["agent_id"]
        self.event_buffer[agent].append(event)

    def get_threat_score(self, agent_id: str) -> float:
        events = self.event_buffer.get(agent_id, [])
        if not events:
            return 0.0

        # Combine confidences
        scores = [e.payload["confidence"] for e in events]
        return min(1.0, sum(scores) / len(scores))
