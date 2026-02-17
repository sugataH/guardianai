from typing import List, Callable
from datetime import datetime
from guardianai.eventbus.schemas import SignedEvent
from guardianai.eventbus.signing import sign_payload, verify_signature


class EventBus:
    """
    Cloud-style zero-trust event bus.
    Sidecars publish alerts.
    Supervisor subscribes and verifies them.
    """

    def __init__(self):
        self.subscribers: List[Callable] = []
        self.event_log: List[SignedEvent] = []

    def subscribe(self, callback: Callable):
        self.subscribers.append(callback)

    def publish(self, sender_id: str, alert_payload: dict):
        signed = sign_payload(alert_payload)

        event = SignedEvent(
            event_id=alert_payload["event_id"],
            payload=alert_payload,
            sender_id=sender_id,
            timestamp=datetime.utcnow(),
            signature=signed

        
        )

        self.event_log.append(event)

        # Deliver to subscribers safely
        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception as e:
                print("[EventBus] Subscriber error:", e)

    def verify(self, event: SignedEvent) -> bool:
        return verify_signature(event.payload, event.signature)
