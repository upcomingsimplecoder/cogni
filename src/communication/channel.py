"""Message bus for routing inter-agent communication."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.communication.mailbox import Mailbox
from src.communication.protocol import Message, MessageType

if TYPE_CHECKING:
    pass  # AgentRegistry and World accessed via duck typing


class MessageBus:
    """Central message routing: direct + broadcast.

    Messages sent during a tick are queued and delivered at tick end,
    so all agents see the same world state during the decision phase.
    """

    def __init__(self):
        self._pending: list[Message] = []
        self._history: list[Message] = []
        self._mailboxes: dict[object, Mailbox] = {}  # AgentID -> Mailbox

    def get_or_create_mailbox(self, agent_id: object) -> Mailbox:
        """Get or create a mailbox for an agent."""
        if agent_id not in self._mailboxes:
            self._mailboxes[agent_id] = Mailbox(agent_id)
        return self._mailboxes[agent_id]

    def send(self, message: Message) -> None:
        """Queue a message for delivery at end of tick."""
        self._pending.append(message)
        # Record in sender's mailbox
        sender_mailbox = self.get_or_create_mailbox(message.sender_id)
        sender_mailbox.record_sent(message)

    def deliver_all(
        self,
        registry: object,
        world: object,
        broadcast_range: int = 6,
        communication_range: int = 10,
    ) -> list[Message]:
        """Deliver all pending messages. Returns list of delivered messages.

        Broadcast: delivered to all agents within broadcast_range of sender.
        Direct: delivered to receiver if within communication_range.

        Args:
            registry: AgentRegistry (duck typed to avoid circular import)
            world: World instance (duck typed)
            broadcast_range: Max tiles for broadcast messages
            communication_range: Max tiles for direct messages
        """
        delivered = []

        for msg in self._pending:
            if msg.message_type == MessageType.BROADCAST or msg.receiver_id is None:
                # Broadcast to all agents in range
                sender = getattr(registry, "get", lambda x: None)(msg.sender_id)
                if sender is None:
                    continue
                nearby_ids: list = getattr(world, "agents_in_radius", lambda *args: [])(
                    sender.x, sender.y, broadcast_range
                )
                for aid in nearby_ids:
                    if aid != msg.sender_id:
                        mailbox = self.get_or_create_mailbox(aid)
                        mailbox.receive(msg)
                        delivered.append(msg)
            else:
                # Direct message
                sender = getattr(registry, "get", lambda x: None)(msg.sender_id)
                receiver = getattr(registry, "get", lambda x: None)(msg.receiver_id)
                if sender is None or receiver is None:
                    continue
                # Check range
                dist = abs(sender.x - receiver.x) + abs(sender.y - receiver.y)
                if dist <= communication_range:
                    mailbox = self.get_or_create_mailbox(msg.receiver_id)
                    mailbox.receive(msg)
                    delivered.append(msg)

        self._history.extend(delivered)
        self._pending.clear()
        return delivered

    @property
    def pending_count(self) -> int:
        """Number of pending messages."""
        return len(self._pending)

    @property
    def total_messages(self) -> int:
        """Total messages delivered historically."""
        return len(self._history)

    def recent_messages(self, n: int = 10) -> list[Message]:
        """Last N delivered messages."""
        return self._history[-n:]
