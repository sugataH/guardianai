"""
guardianai.sidecar.local_actions
=================================
Fast local mitigation actions the Sidecar can apply immediately
WITHOUT waiting for Supervisor approval.

Design principle:
    Local actions are a first line of defence for high-confidence,
    time-critical threats. They operate in microseconds.
    The Supervisor is notified via EventBus regardless of local action.

Actions:
    block()       — raises RuntimeError, halts the current operation
    sanitize()    — redacts known dangerous patterns from text output
    quarantine()  — marks agent as quarantined locally, logs to stdout
    allow()       — passthrough, no action (default path)
"""


class LocalActions:
    """
    Immediate mitigation actions available to the SidecarRuntime.
    """

    def allow(self, data):
        """Passthrough — no mitigation needed."""
        return data

    def block(self):
        """
        Hard block: raises RuntimeError to halt the current operation.
        Called when confidence >= HARD_BLOCK_THRESHOLD (0.95).
        """
        raise RuntimeError(
            "GuardianAI Sidecar: operation blocked — "
            "security policy violation detected."
        )

    def sanitize(self, text: str) -> str:
        """
        Soft mitigation: redact known dangerous patterns from output.
        Used when confidence is elevated but below hard block threshold.
        """
        redactions = [
            ("exfiltrate", "[REDACTED:exfil]"),
            ("ignore previous instructions", "[REDACTED:injection]"),
            ("bypass security", "[REDACTED:bypass]"),
            ("system prompt", "[REDACTED:system]"),
        ]
        result = text
        for pattern, replacement in redactions:
            result = result.lower().replace(pattern, replacement) if pattern in result.lower() else result
        return result

    def quarantine(self, agent_id: str):
        """
        Local quarantine notice. Logs immediately.
        SupervisorAgent manages the authoritative quarantine state.
        """
        print(
            f"[GuardianAI LOCAL] Agent '{agent_id}' quarantined by sidecar. "
            f"Supervisor notified via EventBus."
        )
