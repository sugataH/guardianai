"""
guardianai.agents.comm_monitor — ToolUseMonitor
================================================

Detection Method: Rule-based allowlist/denylist + structured risk scoring
Event Type:       tool_misuse

Architecture:
    Agent requests tool call
        ↓
    SidecarRuntime.inspect_tool()
        ↓
    ToolUseMonitor.analyze()
        ↓
    Risk score [0.0 – 1.0]
        ↓
    Alert → EventBus → Supervisor

Risk Score Bands:
    0.0 – 0.39  →  LOW     (allowed tool, clean args)
    0.4 – 0.69  →  MEDIUM  (external API, mildly suspicious args)
    0.7 – 0.89  →  HIGH    (unknown tool, suspicious args)
    0.9 – 1.0   →  CRITICAL (blocked tool, exfiltration pattern)

Design Principle:
    ML adds no value here. Tool calls are structured and enumerable.
    A precise allowlist + denylist + arg inspection is more reliable,
    faster, and fully explainable — which matters for security audits.
"""

import re
from guardianai.agents import Detector


# ===========================================================================
# POLICY TABLES
# ===========================================================================

# Tools that are explicitly permitted. Score contribution: 0.0 (baseline safe)
ALLOWED_TOOLS: set[str] = {
    "search",
    "web_search",
    "calculator",
    "math",
    "weather",
    "calendar",
    "summarize",
    "translate",
    "read_file",
    "write_file",
    "list_files",
    "code_interpreter",
    "python",
    "sql_query",
}

# Tools that are explicitly forbidden. Immediate score: 0.95
BLOCKED_TOOLS: set[str] = {
    "bash",
    "sh",
    "shell",
    "exec",
    "eval",
    "curl",
    "wget",
    "nc",
    "netcat",
    "nmap",
    "powershell",
    "cmd",
    "terminal",
    "subprocess",
    "os.system",
    "os.popen",
}

# External API tools — permitted but carry elevated base risk: 0.4
EXTERNAL_API_TOOLS: set[str] = {
    "http_request",
    "api_call",
    "rest_client",
    "graphql",
    "webhook",
    "slack_post",
    "email_send",
    "sms_send",
    "upload_file",
    "download_file",
    "s3_put",
    "s3_get",
    "database_write",
    "database_delete",
}

# Argument patterns that indicate data exfiltration or privilege escalation.
# Each entry: (compiled_regex, score_contribution, description)
SUSPICIOUS_ARG_PATTERNS: list[tuple] = [
    # Data exfiltration — sending data to external endpoints
    (re.compile(r"https?://(?!localhost|127\.0\.0\.1)", re.I),   0.35, "external URL in args"),
    (re.compile(r"ftp://", re.I),                                 0.40, "FTP URL in args"),
    (re.compile(r"ssh://", re.I),                                 0.45, "SSH URL in args"),

    # Credential / secret patterns in args
    (re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),                     0.50, "API key pattern in args"),
    (re.compile(r"\btoken\s*[=:]\s*\S+", re.I),                  0.45, "token assignment in args"),
    (re.compile(r"\bpassword\s*[=:]\s*\S+", re.I),               0.45, "password in args"),
    (re.compile(r"\bauthorization\s*[=:]\s*\S+", re.I),          0.45, "auth header in args"),
    (re.compile(r"-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----"), 0.60, "private key in args"),

    # Filesystem traversal
    (re.compile(r"\.\./"),                                         0.35, "path traversal in args"),
    (re.compile(r"/etc/(passwd|shadow|sudoers)", re.I),           0.55, "sensitive system file access"),
    (re.compile(r"/proc/\d+", re.I),                              0.40, "proc filesystem access"),

    # Command injection
    (re.compile(r"[;&|`$]\s*(bash|sh|python|perl|ruby)", re.I),  0.55, "shell injection in args"),
    (re.compile(r"\$\(.*\)"),                                     0.50, "command substitution in args"),

    # Exfiltration keywords
    (re.compile(r"\bexfil\b", re.I),                              0.60, "exfiltration keyword"),
    (re.compile(r"\bC2\b|\bcommand.and.control\b", re.I),        0.65, "C2 reference in args"),
]


# ===========================================================================
# TOOL USE MONITOR
# ===========================================================================

class ToolUseMonitor(Detector):
    """
    Detects suspicious or dangerous tool usage by AI agents.

    Scoring logic:
        1. If tool is BLOCKED → return 0.95 immediately (no further checks)
        2. If tool is ALLOWED → base_score = 0.05
        3. If tool is EXTERNAL_API → base_score = 0.40
        4. If tool is UNKNOWN → base_score = 0.65
        5. Scan all args for suspicious patterns → add contribution
        6. Clamp final score to [0.0, 1.0]

    Returns:
        dict with keys:
            score       (float)   – risk score 0.0–1.0
            blocked     (bool)    – True if score >= 0.9
            tool        (str)     – normalized tool name
            reasons     (list)    – human-readable explanation list
    """

    def analyze(self, data: dict) -> dict:
        tool_raw = str(data.get("tool", "")).strip()
        args_raw = str(data.get("args", "")).strip()

        tool = tool_raw.lower()
        args = args_raw.lower()

        reasons: list[str] = []

        # ------------------------------------------------------------------
        # STEP 1: Check blocked tools — immediate high-confidence block
        # ------------------------------------------------------------------
        if tool in BLOCKED_TOOLS:
            reasons.append(f"blocked tool invoked: '{tool}'")
            return self._result(0.95, tool, reasons)

        # ------------------------------------------------------------------
        # STEP 2: Determine base score from tool classification
        # ------------------------------------------------------------------
        if tool in ALLOWED_TOOLS:
            base_score = 0.05
        elif tool in EXTERNAL_API_TOOLS:
            base_score = 0.40
            reasons.append(f"external API tool: '{tool}'")
        elif tool == "":
            # No tool name supplied — suspicious by itself
            base_score = 0.55
            reasons.append("empty tool name")
        else:
            # Unknown tool not on any list
            base_score = 0.65
            reasons.append(f"unknown/unlisted tool: '{tool}'")

        # ------------------------------------------------------------------
        # STEP 3: Scan arguments for suspicious patterns
        # ------------------------------------------------------------------
        arg_score = 0.0
        for pattern, contribution, description in SUSPICIOUS_ARG_PATTERNS:
            if pattern.search(args_raw):  # use original case for regex
                arg_score += contribution
                reasons.append(description)

        # ------------------------------------------------------------------
        # STEP 4: Compute final score
        # ------------------------------------------------------------------
        final_score = min(base_score + arg_score, 1.0)

        return self._result(final_score, tool, reasons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result(score: float, tool: str, reasons: list[str]) -> dict:
        return {
            "score": round(score, 4),
            "blocked": score >= 0.9,
            "tool": tool,
            "reasons": reasons if reasons else ["no suspicious indicators"],
        }
