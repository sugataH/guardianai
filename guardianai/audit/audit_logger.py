from guardianai.audit.merkle import MerkleLog
from guardianai.eventbus.schemas import SignedEvent


class AuditLogger:
    """
    Tamper-proof audit log for all GuardianAI security events.
    """

    def __init__(self):
        self.merkle = MerkleLog()
        self.records = []

    def log_event(self, event: SignedEvent):
        record = f"{event.event_id}:{event.signature}"
        self.records.append(record)
        self.merkle.append(record)

    def verify_integrity(self) -> bool:
        return self.merkle.verify(self.records)
