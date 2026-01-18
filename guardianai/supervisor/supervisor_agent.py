"""receives SignedEvents
verifies them
correlates
applies policy
enforces actions"""

import time
from collections import defaultdict

from guardianai.supervisor.correlation import Correlator
from guardianai.supervisor.policy_engine import PolicyEngine
from guardianai.eventbus.bus import EventBus
from guardianai.eventbus.schemas import SignedEvent
from guardianai.utils.logger import get_logger
from guardianai.utils.metrics import Metrics



class SupervisorAgent:
    """
    Central GuardianAI brain.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.correlator = Correlator()
        self.policy = PolicyEngine()

        # Cooldown management
        self.cooldowns = defaultdict(float)
        self.cooldown_window = 5.0  # seconds

        # Agent state tracking
        self.agent_states = {}

        #logger
        self.logger = get_logger("guardianai.supervisor")

        self.bus.subscribe(self.receive_event)
        
        #metrics
        self.metrics = Metrics()



    def receive_event(self, event: SignedEvent):
        # Verify signature first
        if not self.bus.verify(event):
            return

        # Metrics: count only verified events
        self.metrics.inc("alerts.total")
        self.metrics.inc(f"alerts.type.{event.payload['event_type']}")

        agent_id = event.payload["agent_id"]
        event_type = event.payload["event_type"]
        now = time.time()

        cooldown_key = (agent_id, event_type)

        # Deduplicate alerts within cooldown window
        if now - self.cooldowns[cooldown_key] < self.cooldown_window:
            return

        self.cooldowns[cooldown_key] = now

        # Add to correlation engine
        self.correlator.add_event(event)

        # Compute threat
        threat = self.correlator.get_threat_score(agent_id)

        # Do not repeat actions on quarantined agents
        current_state = self.agent_states.get(agent_id, "NORMAL")
        if current_state == "QUARANTINED":
            return

        # Decide action
        action = self.policy.decide(threat)

        self.logger.info(
            f"decision agent={agent_id} threat={threat:.2f} action={action}"
        )

        if action == "quarantine":
            self.metrics.inc("actions.quarantine")
            self.logger.warning(f"quarantine agent={agent_id}")
            self.agent_states[agent_id] = "QUARANTINED"
