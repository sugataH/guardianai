from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime


class SignedEvent(BaseModel):
    """
    Transport wrapper for SecurityAlert.
    This moves through the Event Bus.
    """
    event_id: str
    payload: Dict[str, Any]
    sender_id: str
    timestamp: datetime
    signature: str
