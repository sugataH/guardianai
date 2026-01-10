from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64


class AgentIdentity:
    """
    Cryptographic identity for every GuardianAI agent.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()

    def sign(self, data: bytes) -> str:
        signature = self.private_key.sign(
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def verify(self, data: bytes, signature: str) -> bool:
        try:
            self.public_key.verify(
                base64.b64decode(signature),
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return True
        except:
            return False
