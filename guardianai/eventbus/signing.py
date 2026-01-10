import hmac
import hashlib
import base64
import json
from typing import Dict, Any
from datetime import datetime


SECRET_KEY = b"guardianai_eventbus_secret"


def _canonicalize(obj: Any):
    """
    Convert Pydantic / Python objects into JSON-safe canonical form.
    This is critical for cryptographic signing.
    """
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def sign_payload(payload: Dict) -> str:
    canonical = _canonicalize(payload)
    data = json.dumps(canonical, sort_keys=True).encode()
    signature = hmac.new(SECRET_KEY, data, hashlib.sha256).digest()
    return base64.b64encode(signature).decode()


def verify_signature(payload: Dict, signature: str) -> bool:
    expected = sign_payload(payload)
    return hmac.compare_digest(expected, signature)
