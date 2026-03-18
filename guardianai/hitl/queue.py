"""
guardianai.hitl.queue
======================
HITL Queue — Human-in-the-Loop review system.

When the Supervisor determines that an agent requires human review
(BLOCK or QUARANTINE action), a HITL entry is created here.
Human operators review the entry and either:
    - Approve restoration (agent returns to normal operation)
    - Confirm quarantine (agent remains suspended)
    - Escalate (flag for security team)

Queue entries contain:
    - Full alert payload for human review
    - Agent trust history
    - Enforcement action that triggered the review
    - Urgency flag (QUARANTINE = urgent)
    - Status: PENDING | REVIEWED | RESOLVED | ESCALATED

This is the working queue implementation requested.
A REST API (hitl/api.py) wraps this queue for external operator access.
"""

import time
import uuid
import logging
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class HITLEntry:
    """
    A single HITL review entry.
    """

    STATUS_PENDING   = "PENDING"
    STATUS_REVIEWED  = "REVIEWED"
    STATUS_RESOLVED  = "RESOLVED"
    STATUS_ESCALATED = "ESCALATED"

    def __init__(self, agent_id: str, action: str,
                 event_payload: dict, urgent: bool = False):
        self.entry_id      = str(uuid.uuid4())
        self.agent_id      = agent_id
        self.action        = action
        self.event_payload = event_payload
        self.urgent        = urgent
        self.created_at    = time.time()
        self.updated_at    = time.time()
        self.status        = self.STATUS_PENDING
        self.reviewed_by   = None
        self.review_notes  = None

    def to_dict(self) -> dict:
        return {
            "entry_id":     self.entry_id,
            "agent_id":     self.agent_id,
            "action":       self.action,
            "urgent":       self.urgent,
            "status":       self.status,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
            "reviewed_by":  self.reviewed_by,
            "review_notes": self.review_notes,
            "event_type":   self.event_payload.get("event_type", "unknown"),
            "confidence":   self.event_payload.get("confidence", 0.0),
            "severity":     self.event_payload.get("severity",   "unknown"),
        }


class HITLQueue:
    """
    Thread-safe HITL review queue.

    Operators interact via:
        enqueue(agent_id, action, payload, urgent)  — called by EnforcementEngine
        notify(agent_id, action, payload)            — non-blocking notification
        review(entry_id, reviewer, notes)            — operator reviews entry
        resolve(agent_id, resolved_by)               — supervisor restores agent
        list_pending()                               — get all pending entries
        is_pending(agent_id)                         — check if agent has pending entry
        snapshot()                                   — full queue state
    """

    def __init__(self):
        self._entries: dict[str, HITLEntry] = {}   # entry_id → HITLEntry
        self._agent_entries: dict[str, list] = {}  # agent_id → [entry_id]
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Enqueue / notify
    # ------------------------------------------------------------------

    def enqueue(self, agent_id: str, action: str,
                event_payload: dict, urgent: bool = False) -> str:
        """
        Create a HITL entry requiring human review.

        Returns:
            entry_id (str)
        """
        entry = HITLEntry(agent_id, action, event_payload, urgent)
        with self._lock:
            self._entries[entry.entry_id] = entry
            if agent_id not in self._agent_entries:
                self._agent_entries[agent_id] = []
            self._agent_entries[agent_id].append(entry.entry_id)

        urgency = "URGENT" if urgent else "NORMAL"
        logger.warning(
            f"HITLQueue: [{urgency}] enqueued entry_id={entry.entry_id} "
            f"agent={agent_id} action={action}"
        )
        return entry.entry_id

    def notify(self, agent_id: str, action: str,
               event_payload: dict) -> str:
        """
        Non-blocking notification — creates a lower-priority entry.
        Used for THROTTLE actions that don't require immediate review.
        """
        return self.enqueue(agent_id, action, event_payload, urgent=False)

    # ------------------------------------------------------------------
    # Review operations
    # ------------------------------------------------------------------

    def review(self, entry_id: str, reviewer: str,
               notes: Optional[str] = None) -> Optional[dict]:
        """
        Mark a HITL entry as reviewed by a human operator.

        Args:
            entry_id: HITL entry to review
            reviewer: operator identifier
            notes:    optional review notes

        Returns:
            Updated entry dict or None if not found.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                logger.warning(f"HITLQueue: review entry_id={entry_id} not found")
                return None
            entry.status       = HITLEntry.STATUS_REVIEWED
            entry.reviewed_by  = reviewer
            entry.review_notes = notes
            entry.updated_at   = time.time()
            logger.info(
                f"HITLQueue: reviewed entry_id={entry_id} "
                f"agent={entry.agent_id} by={reviewer}"
            )
            return entry.to_dict()

    def resolve(self, agent_id: str,
                resolved_by: str = "HITL_OPERATOR") -> int:
        """
        Resolve all pending entries for an agent (called on restore).

        Returns:
            Number of entries resolved.
        """
        count = 0
        with self._lock:
            for eid in self._agent_entries.get(agent_id, []):
                entry = self._entries.get(eid)
                if entry and entry.status in (
                    HITLEntry.STATUS_PENDING, HITLEntry.STATUS_REVIEWED
                ):
                    entry.status      = HITLEntry.STATUS_RESOLVED
                    entry.reviewed_by = resolved_by
                    entry.updated_at  = time.time()
                    count += 1
        logger.info(
            f"HITLQueue: resolved {count} entries for agent={agent_id} "
            f"by={resolved_by}"
        )
        return count

    def escalate(self, entry_id: str, escalated_by: str) -> Optional[dict]:
        """Mark entry as escalated to security team."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return None
            entry.status      = HITLEntry.STATUS_ESCALATED
            entry.reviewed_by = escalated_by
            entry.updated_at  = time.time()
            logger.warning(
                f"HITLQueue: ESCALATED entry_id={entry_id} "
                f"agent={entry.agent_id} by={escalated_by}"
            )
            return entry.to_dict()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_pending(self, agent_id: str) -> bool:
        """Check if an agent has any pending (unresolved) entries."""
        with self._lock:
            for eid in self._agent_entries.get(agent_id, []):
                entry = self._entries.get(eid)
                if entry and entry.status == HITLEntry.STATUS_PENDING:
                    return True
        return False

    def list_pending(self) -> list:
        """Return all pending entries, urgent first."""
        with self._lock:
            pending = [
                e.to_dict()
                for e in self._entries.values()
                if e.status == HITLEntry.STATUS_PENDING
            ]
        return sorted(pending, key=lambda x: (not x["urgent"], x["created_at"]))

    def get_entry(self, entry_id: str) -> Optional[dict]:
        """Retrieve a specific entry by ID."""
        with self._lock:
            entry = self._entries.get(entry_id)
            return entry.to_dict() if entry else None

    def get_agent_entries(self, agent_id: str) -> list:
        """All entries for a specific agent, newest first."""
        with self._lock:
            eids = self._agent_entries.get(agent_id, [])
            entries = [
                self._entries[eid].to_dict()
                for eid in eids
                if eid in self._entries
            ]
        return sorted(entries, key=lambda x: x["created_at"], reverse=True)

    def snapshot(self) -> dict:
        """Full queue state for reporting."""
        with self._lock:
            all_entries = [e.to_dict() for e in self._entries.values()]
        by_status = {}
        for e in all_entries:
            by_status.setdefault(e["status"], []).append(e)
        return {
            "total":   len(all_entries),
            "pending": len(by_status.get(HITLEntry.STATUS_PENDING,   [])),
            "reviewed":len(by_status.get(HITLEntry.STATUS_REVIEWED,  [])),
            "resolved":len(by_status.get(HITLEntry.STATUS_RESOLVED,  [])),
            "escalated":len(by_status.get(HITLEntry.STATUS_ESCALATED,[])),
            "entries": all_entries,
        }
