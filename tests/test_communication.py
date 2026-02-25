"""Tests for communication system: MessageBus, Mailbox, and protocol."""

from __future__ import annotations

import pytest

from src.agents.identity import AgentID
from src.communication.channel import MessageBus
from src.communication.mailbox import Mailbox
from src.communication.protocol import (
    MessageType,
    create_message,
)


class MockAgent:
    """Mock agent for testing message bus."""

    def __init__(self, agent_id: AgentID, x: int, y: int):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.alive = True


class MockRegistry:
    """Mock registry for testing message bus."""

    def __init__(self):
        self._agents: dict[AgentID, MockAgent] = {}

    def add_agent(self, agent: MockAgent) -> None:
        """Add a mock agent to the registry."""
        self._agents[agent.agent_id] = agent

    def get(self, aid: AgentID) -> MockAgent | None:
        """Get an agent by ID."""
        return self._agents.get(aid)

    def living_agents(self) -> list[MockAgent]:
        """Return all living agents."""
        return list(self._agents.values())


class MockWorld:
    """Mock world for testing message bus."""

    def __init__(self):
        self._agents_by_location: dict[tuple[int, int], set[AgentID]] = {}

    def add_agent_at(self, agent_id: AgentID, x: int, y: int) -> None:
        """Place an agent at a location."""
        if (x, y) not in self._agents_by_location:
            self._agents_by_location[(x, y)] = set()
        self._agents_by_location[(x, y)].add(agent_id)

    def agents_in_radius(self, cx: int, cy: int, radius: int) -> set[AgentID]:
        """Return all agent IDs within radius (simplified manhattan distance)."""
        agents = set()
        for (x, y), aids in self._agents_by_location.items():
            dist = abs(x - cx) + abs(y - cy)
            if dist <= radius:
                agents.update(aids)
        return agents


# Protocol tests


def test_create_message_creates_message_with_auto_id():
    """create_message() creates Message with auto-generated ID."""
    msg = create_message(
        tick=10,
        sender_id=AgentID("sender001"),
        receiver_id=AgentID("receiver001"),
        message_type=MessageType.INFORM,
        content="Test message",
    )

    assert msg.id is not None
    assert len(msg.id) == 8
    assert msg.tick == 10
    assert msg.sender_id == AgentID("sender001")
    assert msg.receiver_id == AgentID("receiver001")
    assert msg.message_type == MessageType.INFORM
    assert msg.content == "Test message"
    assert msg.payload == {}
    assert msg.energy_cost == 0.5


def test_create_message_with_payload():
    """create_message() accepts payload dict."""
    payload = {"location": (10, 20), "resource": "berries"}
    msg = create_message(
        tick=5,
        sender_id=AgentID("sender002"),
        receiver_id=None,
        message_type=MessageType.BROADCAST,
        content="Found food",
        payload=payload,
        energy_cost=1.0,
    )

    assert msg.payload == payload
    assert msg.energy_cost == 1.0
    assert msg.receiver_id is None


def test_message_is_frozen():
    """Message is frozen (immutable)."""
    msg = create_message(
        tick=1,
        sender_id=AgentID("test"),
        receiver_id=None,
        message_type=MessageType.INFORM,
        content="Immutable test",
    )

    with pytest.raises(Exception, match=""):  # FrozenInstanceError or AttributeError
        msg.content = "Modified"

    with pytest.raises(Exception, match=""):
        msg.tick = 999


def test_messagetype_enum_has_expected_values():
    """MessageType enum has all expected values."""
    expected = {
        "REQUEST",
        "INFORM",
        "NEGOTIATE",
        "BROADCAST",
        "THREAT",
        "ACKNOWLEDGE",
    }
    actual = {mt.name for mt in MessageType}

    assert actual == expected


# Mailbox tests


def test_mailbox_receive_and_drain_inbox_roundtrip():
    """Mailbox: receive + drain_inbox round-trip."""
    agent_id = AgentID("mailbox001")
    mailbox = Mailbox(agent_id)

    msg1 = create_message(
        tick=1,
        sender_id=AgentID("sender001"),
        receiver_id=agent_id,
        message_type=MessageType.INFORM,
        content="Message 1",
    )
    msg2 = create_message(
        tick=2,
        sender_id=AgentID("sender002"),
        receiver_id=agent_id,
        message_type=MessageType.REQUEST,
        content="Message 2",
    )

    mailbox.receive(msg1)
    mailbox.receive(msg2)

    assert mailbox.inbox_count == 2

    messages = mailbox.drain_inbox()

    assert len(messages) == 2
    assert msg1 in messages
    assert msg2 in messages


def test_mailbox_drain_inbox_clears_inbox():
    """Mailbox: drain_inbox clears inbox."""
    agent_id = AgentID("mailbox002")
    mailbox = Mailbox(agent_id)

    msg = create_message(
        tick=1,
        sender_id=AgentID("sender"),
        receiver_id=agent_id,
        message_type=MessageType.INFORM,
        content="Clear test",
    )

    mailbox.receive(msg)
    assert mailbox.inbox_count == 1

    mailbox.drain_inbox()

    assert mailbox.inbox_count == 0


def test_mailbox_record_sent_appears_in_history():
    """Mailbox: record_sent appears in history."""
    agent_id = AgentID("mailbox003")
    mailbox = Mailbox(agent_id)

    msg = create_message(
        tick=1,
        sender_id=agent_id,
        receiver_id=AgentID("receiver"),
        message_type=MessageType.REQUEST,
        content="Sent message",
    )

    mailbox.record_sent(msg)

    history = mailbox.recent_history(n=10)
    assert len(history) == 1
    assert history[0].message == msg
    assert history[0].was_sender is True


def test_mailbox_recent_history_returns_sent_and_received():
    """Mailbox: recent_history returns sent + received."""
    agent_id = AgentID("mailbox004")
    mailbox = Mailbox(agent_id)

    sent_msg = create_message(
        tick=1,
        sender_id=agent_id,
        receiver_id=AgentID("receiver"),
        message_type=MessageType.INFORM,
        content="I sent this",
    )
    received_msg = create_message(
        tick=2,
        sender_id=AgentID("sender"),
        receiver_id=agent_id,
        message_type=MessageType.REQUEST,
        content="I received this",
    )

    mailbox.record_sent(sent_msg)
    mailbox.receive(received_msg)
    mailbox.drain_inbox()

    history = mailbox.recent_history(n=10)

    assert len(history) == 2
    sent_record = next(r for r in history if r.was_sender)
    received_record = next(r for r in history if not r.was_sender)

    assert sent_record.message == sent_msg
    assert received_record.message == received_msg


# MessageBus tests


def test_messagebus_send_queues_message():
    """MessageBus: send() queues message."""
    bus = MessageBus()

    msg = create_message(
        tick=1,
        sender_id=AgentID("sender"),
        receiver_id=AgentID("receiver"),
        message_type=MessageType.INFORM,
        content="Queue test",
    )

    assert bus.pending_count == 0

    bus.send(msg)

    assert bus.pending_count == 1


def test_messagebus_send_records_in_sender_mailbox():
    """MessageBus: send() records message in sender's mailbox."""
    bus = MessageBus()
    sender_id = AgentID("sender_mailbox")

    msg = create_message(
        tick=1,
        sender_id=sender_id,
        receiver_id=AgentID("receiver"),
        message_type=MessageType.INFORM,
        content="Sender history test",
    )

    bus.send(msg)

    sender_mailbox = bus.get_or_create_mailbox(sender_id)
    history = sender_mailbox.recent_history(n=10)

    assert len(history) == 1
    assert history[0].message == msg
    assert history[0].was_sender is True


def test_messagebus_deliver_all_direct_message_within_range():
    """MessageBus: deliver_all() delivers direct message within range."""
    bus = MessageBus()
    registry = MockRegistry()
    world = MockWorld()

    sender_id = AgentID("sender")
    receiver_id = AgentID("receiver")

    sender = MockAgent(sender_id, x=5, y=5)
    receiver = MockAgent(receiver_id, x=10, y=10)

    registry.add_agent(sender)
    registry.add_agent(receiver)
    world.add_agent_at(sender_id, 5, 5)
    world.add_agent_at(receiver_id, 10, 10)

    msg = create_message(
        tick=1,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.REQUEST,
        content="Direct within range",
    )

    bus.send(msg)

    # Manhattan distance: |10-5| + |10-5| = 10, which is exactly at communication_range
    delivered = bus.deliver_all(registry, world, communication_range=10)

    assert len(delivered) == 1
    assert delivered[0] == msg

    receiver_mailbox = bus.get_or_create_mailbox(receiver_id)
    assert receiver_mailbox.inbox_count == 1


def test_messagebus_deliver_all_drops_direct_message_out_of_range():
    """MessageBus: deliver_all() drops direct message out of range."""
    bus = MessageBus()
    registry = MockRegistry()
    world = MockWorld()

    sender_id = AgentID("sender_far")
    receiver_id = AgentID("receiver_far")

    sender = MockAgent(sender_id, x=0, y=0)
    receiver = MockAgent(receiver_id, x=20, y=20)

    registry.add_agent(sender)
    registry.add_agent(receiver)
    world.add_agent_at(sender_id, 0, 0)
    world.add_agent_at(receiver_id, 20, 20)

    msg = create_message(
        tick=1,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.REQUEST,
        content="Direct out of range",
    )

    bus.send(msg)

    # Manhattan distance: |20-0| + |20-0| = 40, exceeds communication_range=10
    delivered = bus.deliver_all(registry, world, communication_range=10)

    assert len(delivered) == 0

    receiver_mailbox = bus.get_or_create_mailbox(receiver_id)
    assert receiver_mailbox.inbox_count == 0


def test_messagebus_deliver_all_broadcasts_to_agents_in_range():
    """MessageBus: deliver_all() broadcasts to all agents in range."""
    bus = MessageBus()
    registry = MockRegistry()
    world = MockWorld()

    sender_id = AgentID("broadcaster")
    listener1_id = AgentID("listener1")
    listener2_id = AgentID("listener2")
    far_id = AgentID("far_away")

    sender = MockAgent(sender_id, x=10, y=10)
    listener1 = MockAgent(listener1_id, x=12, y=12)
    listener2 = MockAgent(listener2_id, x=15, y=10)
    far_agent = MockAgent(far_id, x=30, y=30)

    registry.add_agent(sender)
    registry.add_agent(listener1)
    registry.add_agent(listener2)
    registry.add_agent(far_agent)

    world.add_agent_at(sender_id, 10, 10)
    world.add_agent_at(listener1_id, 12, 12)
    world.add_agent_at(listener2_id, 15, 10)
    world.add_agent_at(far_id, 30, 30)

    msg = create_message(
        tick=1,
        sender_id=sender_id,
        receiver_id=None,
        message_type=MessageType.BROADCAST,
        content="Broadcast test",
    )

    bus.send(msg)

    delivered = bus.deliver_all(registry, world, broadcast_range=6)

    # listener1: |12-10| + |12-10| = 4 (within range)
    # listener2: |15-10| + |10-10| = 5 (within range)
    # far_agent: |30-10| + |30-10| = 40 (out of range)
    # Should deliver to 2 agents (listener1, listener2)
    assert len(delivered) == 2

    listener1_mailbox = bus.get_or_create_mailbox(listener1_id)
    listener2_mailbox = bus.get_or_create_mailbox(listener2_id)
    far_mailbox = bus.get_or_create_mailbox(far_id)

    assert listener1_mailbox.inbox_count == 1
    assert listener2_mailbox.inbox_count == 1
    assert far_mailbox.inbox_count == 0


def test_messagebus_deliver_all_clears_pending_after_delivery():
    """MessageBus: deliver_all() clears pending after delivery."""
    bus = MessageBus()
    registry = MockRegistry()
    world = MockWorld()

    sender_id = AgentID("clear_sender")
    receiver_id = AgentID("clear_receiver")

    sender = MockAgent(sender_id, x=5, y=5)
    receiver = MockAgent(receiver_id, x=6, y=6)

    registry.add_agent(sender)
    registry.add_agent(receiver)
    world.add_agent_at(sender_id, 5, 5)
    world.add_agent_at(receiver_id, 6, 6)

    msg = create_message(
        tick=1,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.INFORM,
        content="Clear pending test",
    )

    bus.send(msg)
    assert bus.pending_count == 1

    bus.deliver_all(registry, world)

    assert bus.pending_count == 0


def test_messagebus_recent_messages_returns_history():
    """MessageBus: recent_messages() returns delivery history."""
    bus = MessageBus()
    registry = MockRegistry()
    world = MockWorld()

    sender_id = AgentID("history_sender")
    receiver_id = AgentID("history_receiver")

    sender = MockAgent(sender_id, x=5, y=5)
    receiver = MockAgent(receiver_id, x=7, y=7)

    registry.add_agent(sender)
    registry.add_agent(receiver)
    world.add_agent_at(sender_id, 5, 5)
    world.add_agent_at(receiver_id, 7, 7)

    msg1 = create_message(
        tick=1,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.INFORM,
        content="History message 1",
    )
    msg2 = create_message(
        tick=2,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.REQUEST,
        content="History message 2",
    )

    bus.send(msg1)
    bus.send(msg2)
    bus.deliver_all(registry, world)

    history = bus.recent_messages(n=10)

    assert len(history) == 2
    assert msg1 in history
    assert msg2 in history


def test_messagebus_total_messages_tracks_delivered():
    """MessageBus: total_messages property tracks delivered messages."""
    bus = MessageBus()
    registry = MockRegistry()
    world = MockWorld()

    sender_id = AgentID("total_sender")
    receiver_id = AgentID("total_receiver")

    sender = MockAgent(sender_id, x=5, y=5)
    receiver = MockAgent(receiver_id, x=6, y=6)

    registry.add_agent(sender)
    registry.add_agent(receiver)
    world.add_agent_at(sender_id, 5, 5)
    world.add_agent_at(receiver_id, 6, 6)

    assert bus.total_messages == 0

    msg1 = create_message(
        tick=1,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.INFORM,
        content="Total 1",
    )
    msg2 = create_message(
        tick=2,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.INFORM,
        content="Total 2",
    )

    bus.send(msg1)
    bus.deliver_all(registry, world)
    assert bus.total_messages == 1

    bus.send(msg2)
    bus.deliver_all(registry, world)
    assert bus.total_messages == 2
