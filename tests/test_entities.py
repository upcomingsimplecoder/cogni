"""Tests for simulation entities: Agent and AgentNeeds."""

from __future__ import annotations

import pytest

from src.agents.identity import AgentID, AgentProfile, PersonalityTraits
from src.simulation.entities import Agent, AgentNeeds


class TestAgentNeeds:
    """Tests for AgentNeeds physiological tracking."""

    def test_default_values(self):
        """AgentNeeds defaults to (80, 80, 90, 100, 37)."""
        needs = AgentNeeds()
        assert needs.hunger == 80.0
        assert needs.thirst == 80.0
        assert needs.energy == 90.0
        assert needs.health == 100.0
        assert needs.temperature == 37.0

    def test_is_alive_with_all_needs_positive(self):
        """Agent is alive when health, hunger, and thirst are all above 0."""
        needs = AgentNeeds(hunger=50, thirst=50, health=50)
        assert needs.is_alive() is True

    @pytest.mark.parametrize(
        "hunger,thirst,health,expected",
        [
            (0, 50, 50, False),  # hunger at 0
            (50, 0, 50, False),  # thirst at 0
            (50, 50, 0, False),  # health at 0
            (-1, 50, 50, False),  # hunger negative
            (50, -1, 50, False),  # thirst negative
            (50, 50, -1, False),  # health negative
            (0.1, 0.1, 0.1, True),  # all barely positive
        ],
    )
    def test_is_alive_boundary_conditions(self, hunger, thirst, health, expected):
        """Agent death occurs when any critical need reaches 0 or below."""
        needs = AgentNeeds(hunger=hunger, thirst=thirst, health=health)
        assert needs.is_alive() is expected

    def test_is_alive_ignores_energy_and_temperature(self):
        """Energy and temperature don't directly affect alive status."""
        needs = AgentNeeds(hunger=50, thirst=50, health=50, energy=0, temperature=20)
        assert needs.is_alive() is True

    def test_most_urgent_need_returns_hunger(self):
        """Returns 'hunger' when it's the lowest need."""
        needs = AgentNeeds(hunger=10, thirst=50, energy=60)
        assert needs.most_urgent_need() == "hunger"

    def test_most_urgent_need_returns_thirst(self):
        """Returns 'thirst' when it's the lowest need."""
        needs = AgentNeeds(hunger=50, thirst=5, energy=60)
        assert needs.most_urgent_need() == "thirst"

    def test_most_urgent_need_returns_energy(self):
        """Returns 'energy' when it's the lowest need."""
        needs = AgentNeeds(hunger=50, thirst=60, energy=3)
        assert needs.most_urgent_need() == "energy"

    def test_most_urgent_need_ignores_health_and_temperature(self):
        """Health and temperature are not tracked as urgent needs."""
        needs = AgentNeeds(hunger=50, thirst=60, energy=70, health=1, temperature=10)
        assert needs.most_urgent_need() == "hunger"

    @pytest.mark.parametrize(
        "value,expected_urgency",
        [
            (100, 0.0),  # fully satisfied
            (50, 0.5),  # half urgent
            (0, 1.0),  # critical
            (25, 0.75),  # mostly urgent
            (75, 0.25),  # slightly urgent
        ],
    )
    def test_urgency_calculation(self, value, expected_urgency):
        """Urgency scales linearly from 0.0 (satisfied) to 1.0 (critical)."""
        needs = AgentNeeds(hunger=value)
        assert needs.urgency("hunger") == pytest.approx(expected_urgency)

    def test_decay_reduces_needs_by_rates(self):
        """Decay reduces hunger, thirst, and energy by specified rates."""
        needs = AgentNeeds(hunger=80, thirst=80, energy=90)
        needs.decay(hunger_rate=5, thirst_rate=10, energy_rate=15)

        assert needs.hunger == 75
        assert needs.thirst == 70
        assert needs.energy == 75

    def test_decay_does_not_go_below_zero(self):
        """Needs cannot decay below 0."""
        needs = AgentNeeds(hunger=5, thirst=3, energy=2)
        needs.decay(hunger_rate=10, thirst_rate=10, energy_rate=10)

        assert needs.hunger == 0
        assert needs.thirst == 0
        assert needs.energy == 0

    def test_decay_damages_health_when_hunger_zero(self):
        """Health drops by 2.0 per tick when hunger is 0."""
        needs = AgentNeeds(hunger=0, health=100)
        needs.decay(hunger_rate=0, thirst_rate=0, energy_rate=0)
        assert needs.health == 98.0

    def test_decay_damages_health_when_thirst_zero(self):
        """Health drops by 3.0 per tick when thirst is 0."""
        needs = AgentNeeds(thirst=0, health=100)
        needs.decay(hunger_rate=0, thirst_rate=0, energy_rate=0)
        assert needs.health == 97.0

    def test_decay_damages_health_when_energy_zero(self):
        """Health drops by 0.5 per tick when energy is 0."""
        needs = AgentNeeds(energy=0, health=100)
        needs.decay(hunger_rate=0, thirst_rate=0, energy_rate=0)
        assert needs.health == 99.5

    def test_decay_multiple_critical_needs_stack_damage(self):
        """Multiple critical needs deal cumulative health damage."""
        needs = AgentNeeds(hunger=0, thirst=0, energy=0, health=100)
        needs.decay(hunger_rate=0, thirst_rate=0, energy_rate=0)
        # 2.0 (hunger) + 3.0 (thirst) + 0.5 (energy) = 5.5 damage
        assert needs.health == 94.5

    def test_decay_health_does_not_go_below_zero(self):
        """Health damage from critical needs stops at 0."""
        needs = AgentNeeds(hunger=0, thirst=0, health=3.0)
        needs.decay(hunger_rate=0, thirst_rate=0, energy_rate=0)
        # Would take 5.0 damage, but capped at 0
        assert needs.health == 0.0


class TestAgent:
    """Tests for Agent entity."""

    def test_default_position_is_center(self):
        """Agent spawns at (32, 32) by default."""
        agent = Agent()
        assert agent.x == 32
        assert agent.y == 32

    def test_custom_position(self):
        """Agent can spawn at custom position."""
        agent = Agent(x=10, y=20)
        assert agent.x == 10
        assert agent.y == 20

    def test_default_alive_status(self):
        """Agent starts alive."""
        agent = Agent()
        assert agent.alive is True

    def test_add_item_to_empty_inventory(self):
        """Adding item to empty inventory creates new entry."""
        agent = Agent()
        agent.add_item("berry", 5)
        assert agent.inventory["berry"] == 5

    def test_add_item_increments_existing(self):
        """Adding item to existing stack increments count."""
        agent = Agent()
        agent.add_item("wood", 3)
        agent.add_item("wood", 2)
        assert agent.inventory["wood"] == 5

    def test_add_item_default_count_is_one(self):
        """add_item without count adds 1."""
        agent = Agent()
        agent.add_item("stone")
        assert agent.inventory["stone"] == 1

    def test_remove_item_with_sufficient_quantity(self):
        """remove_item returns True and decrements when quantity sufficient."""
        agent = Agent()
        agent.add_item("berry", 10)
        result = agent.remove_item("berry", 3)

        assert result is True
        assert agent.inventory["berry"] == 7

    def test_remove_item_deletes_when_zero(self):
        """remove_item deletes key when quantity reaches 0."""
        agent = Agent()
        agent.add_item("berry", 5)
        agent.remove_item("berry", 5)

        assert "berry" not in agent.inventory

    def test_remove_item_insufficient_quantity(self):
        """remove_item returns False when quantity insufficient."""
        agent = Agent()
        agent.add_item("berry", 3)
        result = agent.remove_item("berry", 5)

        assert result is False
        assert agent.inventory["berry"] == 3  # unchanged

    def test_remove_item_nonexistent_item(self):
        """remove_item returns False for items not in inventory."""
        agent = Agent()
        result = agent.remove_item("phantom_item")
        assert result is False

    def test_remove_item_default_count_is_one(self):
        """remove_item without count removes 1."""
        agent = Agent()
        agent.add_item("stone", 5)
        agent.remove_item("stone")
        assert agent.inventory["stone"] == 4

    def test_has_item_returns_true_when_sufficient(self):
        """has_item returns True when quantity sufficient."""
        agent = Agent()
        agent.add_item("berry", 10)
        assert agent.has_item("berry", 5) is True

    def test_has_item_returns_false_when_insufficient(self):
        """has_item returns False when quantity insufficient."""
        agent = Agent()
        agent.add_item("berry", 3)
        assert agent.has_item("berry", 5) is False

    def test_has_item_returns_false_for_nonexistent(self):
        """has_item returns False for items not in inventory."""
        agent = Agent()
        assert agent.has_item("phantom_item") is False

    def test_has_item_default_count_is_one(self):
        """has_item without count checks for at least 1."""
        agent = Agent()
        agent.add_item("stone")
        assert agent.has_item("stone") is True

    def test_die_sets_alive_to_false(self):
        """die() marks agent as not alive."""
        agent = Agent()
        agent.die()
        assert agent.alive is False

    def test_agent_with_profile(self):
        """Agent can be initialized with a profile."""
        agent_id = AgentID("test1234")
        traits = PersonalityTraits(curiosity=0.8, risk_tolerance=0.6)
        profile = AgentProfile(
            agent_id=agent_id,
            name="TestAgent",
            archetype="explorer",
            traits=traits,
        )
        agent = Agent(profile=profile, agent_id=agent_id)

        assert agent.profile is profile
        assert agent.profile.name == "TestAgent"
        assert agent.agent_id == agent_id

    def test_agent_needs_integration(self):
        """Agent needs track health and alive status correctly."""
        agent = Agent()
        agent.needs.hunger = 0
        agent.needs.decay(0, 0, 0)

        # After decay with hunger=0, health should drop
        assert agent.needs.health < 100

        # Alive check uses needs.is_alive()
        assert agent.needs.is_alive() is False
