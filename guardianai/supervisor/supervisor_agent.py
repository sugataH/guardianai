"""receives SignedEvents
verifies them
correlates
applies policy
enforces actions"""

from guardianai.supervisor.correlation import Correlator
from guardianai.supervisor.policy_engine import PolicyEngine
from guardianai.eventbus.bus import EventBus
from guardianai.eventbus.schemas import SignedEvent


class SupervisorAgent:
    """
    Central GuardianAI brain.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.correlator = Correlator()
        self.policy = PolicyEngine()

        self.bus.subscribe(self.receive_event)

    def receive_event(self, event: SignedEvent):
        # Verify signature
        if not self.bus.verify(event):
            print("[Supervisor] Rejected forged event")
            return

        agent_id = event.payload["agent_id"]

        # Add to correlation engine
        self.correlator.add_event(event)

        # Compute threat
        threat = self.correlator.get_threat_score(agent_id)

        # Decide what to do
        action = self.policy.decide(threat)

        print(f"[Supervisor] Agent={agent_id} Threat={threat:.2f} Action={action}")

        if action == "quarantine":
            print(f"[Supervisor] Quarantining {agent_id}")
