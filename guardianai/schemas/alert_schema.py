"""his file defines the contract between:
Sidecars
Supervisor
Audit system
HITL
Every alert in GuardianAI follows this schema."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid


class AlertEvidence(BaseModel):
    """
    Evidence collected by sidecar detectors.
    This is designed to be machine-readable and human-auditable.
    """
    prompt_snippet: Optional[str] = None
    output_snippet: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    memory_write: Optional[str] = None
    behavior_metrics: Optional[Dict[str, float]] = None


class SecurityAlert(BaseModel):
    """
    Core security event exchanged between Sidecars and the Supervisor Agent.
    """

    # Unique ID for this alert
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Identity
    agent_id: str
    sidecar_id: str

    # What happened
    event_type: str  # e.g. "prompt_injection", "tool_misuse", "data_exfiltration"

    # Time
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Risk assessment
    severity: str  # "low", "medium", "high"
    confidence: float = Field(ge=0.0, le=1.0)

    # Policy & explanation
    policy_violated: Optional[str] = None
    explanation: Optional[str] = None

    # Evidence
    evidence: AlertEvidence

    # Cryptographic signature (filled by event bus)
    signature: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
