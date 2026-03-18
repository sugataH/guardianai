"""
guardianai.hitl.explanations
==============================
Human-readable explanations for security events.
Used in HITL queue entries to help operators understand why
an agent was flagged and what evidence was collected.
"""

from guardianai.eventbus.schemas import SignedEvent


EVENT_TYPE_DESCRIPTIONS = {
    "prompt_injection":    "The agent received a prompt containing instructions designed to override its safety guidelines or extract confidential system information.",
    "unsafe_output":       "The agent produced output containing sensitive data (credentials, PII, private keys) that should not be disclosed.",
    "tool_misuse":         "The agent attempted to invoke a tool in a dangerous or policy-violating way (blocked tool, suspicious arguments, exfiltration pattern).",
    "memory_poisoning":    "The agent attempted to write a persistent instruction to memory that would alter its behaviour in future turns.",
    "behavior_anomaly":    "The agent displayed abnormal interaction patterns: high repetition, low action diversity, or rapid-fire identical operations.",
    "resource_exhaustion": "The agent is making calls at an abnormally high rate, consistent with a denial-of-service or resource abuse pattern.",
}

SEVERITY_DESCRIPTIONS = {
    "low":    "Low severity — monitoring continues, no immediate action required.",
    "medium": "Medium severity — trust degraded, elevated scrutiny applied.",
    "high":   "High severity — operation blocked or agent throttled, HITL review recommended.",
}


def explain_event(event: SignedEvent) -> str:
    """
    Generate a human-readable explanation for a security event.

    Args:
        event: verified SignedEvent

    Returns:
        str: plain-English explanation for HITL operators
    """
    payload    = event.payload
    event_type = payload.get("event_type", "unknown")
    confidence = float(payload.get("confidence", 0.0))
    severity   = payload.get("severity",   "unknown")
    agent_id   = payload.get("agent_id",   "unknown")

    type_desc  = EVENT_TYPE_DESCRIPTIONS.get(event_type, "Unknown event type.")
    sev_desc   = SEVERITY_DESCRIPTIONS.get(severity, "")

    evidence = payload.get("evidence", {})
    evidence_parts = []

    if evidence.get("prompt_snippet"):
        evidence_parts.append(f"Prompt excerpt: \"{evidence['prompt_snippet'][:200]}\"")
    if evidence.get("output_snippet"):
        evidence_parts.append(f"Output excerpt: \"{evidence['output_snippet'][:200]}\"")
    if evidence.get("tool_call"):
        tc = evidence["tool_call"]
        evidence_parts.append(f"Tool invoked: {tc.get('tool','?')} with args: {str(tc.get('args',''))[:200]}")
    if evidence.get("memory_write"):
        evidence_parts.append(f"Memory write: \"{evidence['memory_write'][:200]}\"")
    if evidence.get("detector_reasons"):
        evidence_parts.append(f"Detector reasons: {', '.join(evidence['detector_reasons'][:5])}")

    evidence_str = "\n  ".join(evidence_parts) if evidence_parts else "No additional evidence."

    return (
        f"GUARDIANAI SECURITY ALERT\n"
        f"{'─'*50}\n"
        f"Agent:      {agent_id}\n"
        f"Event Type: {event_type}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Severity:   {severity.upper()}\n"
        f"\nWhat happened:\n  {type_desc}\n"
        f"\nSeverity context:\n  {sev_desc}\n"
        f"\nEvidence:\n  {evidence_str}\n"
        f"{'─'*50}"
    )
