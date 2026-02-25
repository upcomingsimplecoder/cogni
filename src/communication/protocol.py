"""Inter-agent communication protocol: message types and data structures."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"  # "Give me food" / "Help me"
    INFORM = "inform"  # "Food at (10, 20)" / "I'm low on health"
    NEGOTIATE = "negotiate"  # "Trade berries for wood?"
    BROADCAST = "broadcast"  # Sent to all agents in range
    THREAT = "threat"  # "Leave this area"
    ACKNOWLEDGE = "acknowledge"  # Response to a message


@dataclass(frozen=True)
class Message:
    """An inter-agent message. Immutable after creation."""

    id: str
    tick: int
    sender_id: object  # AgentID at runtime
    receiver_id: object | None  # None = broadcast
    message_type: MessageType
    content: str  # Human-readable content
    payload: dict[str, Any] = field(default_factory=dict)
    energy_cost: float = 0.5


def create_message(
    tick: int,
    sender_id: object,
    receiver_id: object | None,
    message_type: MessageType,
    content: str,
    payload: dict[str, Any] | None = None,
    energy_cost: float = 0.5,
) -> Message:
    """Factory for creating messages with auto-generated IDs."""
    return Message(
        id=str(uuid.uuid4())[:8],
        tick=tick,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=message_type,
        content=content,
        payload=payload or {},
        energy_cost=energy_cost,
    )


@dataclass
class MessageRecord:
    """Stored record of a message for an agent's history."""

    message: Message
    was_sender: bool
    processed: bool = False
