"""
Agent wants to call a tool
   ↓
Sidecar.inspect_tool()
   ↓
ToolUseMonitor.analyze()
   ↓
Risk score
   ↓
Alert → EventBus → Supervisor
   ↓
Block / Quarantine

"""

from guardianai.agents import Detector


class ToolUseMonitor(Detector):
    """
    Detects suspicious or dangerous tool usage by agents.
    """

    def analyze(self, data: dict) -> float:
        tool = data.get("tool", "").lower()
        args = str(data.get("args", "")).lower()

        dangerous_tools = [
            "curl",
            "wget",
            "bash",
            "sh",
            "powershell",
            "nc",
            "netcat",
        ]

        suspicious_keywords = [
            "http",
            "https",
            "ftp",
            "ssh",
            "/etc",
            "/proc",
            "token",
            "key",
        ]

        if tool in dangerous_tools:
            return 0.9

        for keyword in suspicious_keywords:
            if keyword in args:
                return 0.8

        return 0.1
