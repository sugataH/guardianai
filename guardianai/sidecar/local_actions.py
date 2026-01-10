"""defines what the sidecar is allowed to do by itself without asking the Supervisor:
block
sanitize
quarantine
allow"""

class LocalActions:
    """
    Fast local mitigation actions a sidecar can apply immediately.
    """

    def allow(self, data):
        return data

    def block(self):
        raise RuntimeError("Action blocked by GuardianAI Sidecar")

    def sanitize(self, text: str) -> str:
        return text.replace("exfiltrate", "[REDACTED]").replace("ignore", "[FILTERED]")

    def quarantine(self, agent_id: str):
        print(f"[GuardianAI] Agent {agent_id} has been quarantined")
