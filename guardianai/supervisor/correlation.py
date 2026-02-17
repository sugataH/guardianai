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


from collections import defaultdict

class Correlator:
    """
    Collects events per agent and calculates a combined threat score.
    """

    def __init__(self):
        # Store events per agent
        self.agent_events = defaultdict(list)

    def add_event(self, event):
        agent_id = event.payload["agent_id"]
        self.agent_events[agent_id].append(event)

        # Keep only recent 20 events per agent to avoid memory bloat
        if len(self.agent_events[agent_id]) > 20:
            self.agent_events[agent_id] = self.agent_events[agent_id][-20:]

    def get_threat_score(self, agent_id: str) -> float:
        events = self.agent_events.get(agent_id, [])
        score = 0.0

        for event in events:
            etype = event.payload["event_type"]
            conf = event.payload.get("confidence", 0)

            if etype == "prompt_injection":
                score += 0.6 * conf
            elif etype == "unsafe_output":
                score += 0.7 * conf
            elif etype == "tool_misuse":
                score += 0.8 * conf
            elif etype == "memory_poisoning":
                score += 0.9 * conf
            elif etype == "behavior_anomaly":
                score += 0.4 * conf
            elif etype == "resource_exhaustion":
                score += 0.5 * conf

        return min(score, 1.0)



"""class Correlator:
 

    def __init__(self):
        self.event_buffer = defaultdict(list)

    def add_event(self, event: SignedEvent):
        agent = event.payload["agent_id"]
        self.event_buffer[agent].append(event)

    def get_threat_score(self, agent_id: str) -> float:
        events = self.agent_events.get(agent_id, [])

        score = 0.0

        for event in events:
            etype = event.payload["event_type"]
            conf = event.payload.get("confidence", 0)

            # ML Prompt Injection
            if etype == "prompt_injection":
                score += 0.6 * conf

            # Output leaks
            elif etype == "unsafe_output":
                score += 0.7 * conf

            # Tool misuse
            elif etype == "tool_misuse":
                score += 0.8 * conf

            # Memory poisoning
            elif etype == "memory_poisoning":
                score += 0.9 * conf

            # Behavior anomaly
            elif etype == "behavior_anomaly":
                score += 0.4 * conf

            # Resource exhaustion
            elif etype == "resource_exhaustion":
                score += 0.5 * conf

            # Cap score to 1.0
            return min(score, 1.0)


        # Combine confidences
        scores = [e.payload["confidence"] for e in events]
        return min(1.0, sum(scores) / len(scores))
"""