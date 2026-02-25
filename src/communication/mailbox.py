"""Per-agent message queue and history."""

from __future__ import annotations

from collections import deque

from src.communication.protocol import Message, MessageRecord


class Mailbox:
    """Per-agent message queue + history.

    Incoming messages are queued for the next tick's Sensation phase.
    History is kept for the Reflection phase.
    """

    MAX_HISTORY = 100

    def __init__(self, agent_id: object):
        self.agent_id = agent_id
        self._inbox: deque[Message] = deque()
        self._history: deque[MessageRecord] = deque(maxlen=self.MAX_HISTORY)

    def receive(self, message: Message) -> None:
        """Add message to inbox for delivery."""
        self._inbox.append(message)

    def drain_inbox(self) -> list[Message]:
        """Pop all unread messages. Called during Sensation phase."""
        messages = list(self._inbox)
        self._inbox.clear()
        for msg in messages:
            self._history.append(MessageRecord(message=msg, was_sender=False))
        return messages

    def record_sent(self, message: Message) -> None:
        """Record that this agent sent a message."""
        self._history.append(MessageRecord(message=message, was_sender=True))

    def recent_history(self, n: int = 20) -> list[MessageRecord]:
        """Last N message records (sent + received)."""
        return list(self._history)[-n:]

    @property
    def inbox_count(self) -> int:
        """Number of unread messages."""
        return len(self._inbox)

    @property
    def history_count(self) -> int:
        """Total messages in history."""
        return len(self._history)
