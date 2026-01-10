import hashlib
from typing import List


class MerkleLog:
    """
    Append-only hash chain for GuardianAI events.
    Tampering anywhere breaks the chain.
    """

    def __init__(self):
        self.hashes: List[str] = []

    def _hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    def append(self, record: str) -> str:
        if not self.hashes:
            new_hash = self._hash(record)
        else:
            new_hash = self._hash(self.hashes[-1] + record)

        self.hashes.append(new_hash)
        return new_hash

    def verify(self, records: List[str]) -> bool:
        temp = []
        for record in records:
            if not temp:
                h = self._hash(record)
            else:
                h = self._hash(temp[-1] + record)
            temp.append(h)
        return temp == self.hashes
