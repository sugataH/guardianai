"""
guardianai.schemas.alert_schema
================================
Defines the contract between Sidecars, Supervisor, Audit, and HITL.
Every alert in GuardianAI follows this schema exactly.

Event flow:
    SidecarRuntime → SecurityAlert → EventBus (as SignedEvent)
    → SupervisorAgent → AuditLogger + HITL (if quarantined)
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid


class AlertEvidence(BaseModel):
    """
    Structured evidence collected at detection time.
    Machine-readable and human-auditable.
    All fields are optional — only relevant fields are populated per event type.
    """
    prompt_snippet:    Optional[str]             = None  # prompt_injection
    output_snippet:    Optional[str]             = None  # unsafe_output
    tool_call:         Optional[Dict[str, Any]]  = None  # tool_misuse
    memory_write:      Optional[str]             = None  # memory_poisoning
    behavior_metrics:  Optional[Dict[str, Any]]  = None  # behavior_anomaly, resource_exhaustion
    detector_reasons:  Optional[List[str]]       = None  # human-readable reason list from detector


class SecurityAlert(BaseModel):
    """
    Core security event exchanged between Sidecars and the Supervisor.

    Fields:
        event_id        — UUID, unique per alert
        agent_id        — which operational agent triggered this
        sidecar_id      — which sidecar instance detected it
        event_type      — one of six canonical types (see EVENT_TYPES)
        timestamp       — UTC time of detection
        severity        — low / medium / high
        confidence      — detector risk score [0.0, 1.0]
        policy_violated — optional policy name if known
        explanation     — human-readable explanation from detector
        evidence        — structured evidence (AlertEvidence)
        signature       — HMAC-SHA256 over payload, filled by EventBus
    """

    # --- Identity ---
    event_id:   str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id:   str
    sidecar_id: str

    # --- Classification ---
    event_type: str  # prompt_injection | unsafe_output | tool_misuse |
                     # memory_poisoning | behavior_anomaly | resource_exhaustion

    # --- Timing ---
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # --- Risk ---
    severity:   str    # "low" | "medium" | "high"
    confidence: float  = Field(ge=0.0, le=1.0)

    # --- Context ---
    policy_violated: Optional[str] = None
    explanation:     Optional[str] = None

    # --- Evidence ---
    evidence: AlertEvidence

    # --- Cryptographic integrity ---
    signature: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# Canonical event types — used for correlation weights
EVENT_TYPES = {
    "prompt_injection",
    "unsafe_output",
    "tool_misuse",
    "memory_poisoning",
    "behavior_anomaly",
    "resource_exhaustion",
}
