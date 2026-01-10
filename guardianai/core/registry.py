from typing import Dict
from guardianai.core.identity import AgentIdentity


class AgentRegistry:
    """
    Zero-trust registry of all known agents.
    """

    def __init__(self):
        self.agents: Dict[str, AgentIdentity] = {}

    def register(self, identity: AgentIdentity):
        self.agents[identity.agent_id] = identity

    def get(self, agent_id: str):
        return self.agents.get(agent_id)

    def is_trusted(self, agent_id: str) -> bool:
        return agent_id in self.agents
