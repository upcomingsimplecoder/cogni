"""Tests for genetic and cultural evolution.

Tests Plan 12: Genetic/Cultural Evolution including:
- Trait inheritance and mutation
- Crossover genetics
- Fitness calculation
- Reproduction conditions
- Cultural learning and norm detection
- Lineage tracking
- Population management
"""

from __future__ import annotations

from src.agents.identity import AgentID, AgentProfile, PersonalityTraits
from src.evolution.culture import CulturalNorm, CulturalTransmission, NormDetector
from src.evolution.genetics import GeneticSystem
from src.evolution.lineage import LineageTracker
from src.evolution.population import PopulationManager
from src.evolution.reproduction import ReproductionSystem
from src.simulation.entities import Agent, AgentNeeds


class TestGeneticSystem:
    """Test genetic operations."""

    def test_inherit_with_mutation(self):
        """Test trait inheritance with mutation."""
        genetics = GeneticSystem(mutation_rate=1.0, mutation_magnitude=0.1)

        parent_traits = PersonalityTraits(
            cooperation_tendency=0.5,
            curiosity=0.5,
            risk_tolerance=0.5,
        )

        child_traits = genetics.inherit(parent_traits)

        # Traits should exist
        assert hasattr(child_traits, "cooperation_tendency")
        assert hasattr(child_traits, "curiosity")

        # Traits should be in valid range
        assert 0.0 <= child_traits.cooperation_tendency <= 1.0
        assert 0.0 <= child_traits.curiosity <= 1.0
        assert 0.0 <= child_traits.risk_tolerance <= 1.0

    def test_inherit_no_mutation(self):
        """Test inheritance with no mutation."""
        genetics = GeneticSystem(mutation_rate=0.0)

        parent_traits = PersonalityTraits(
            cooperation_tendency=0.7,
            curiosity=0.3,
        )

        child_traits = genetics.inherit(parent_traits)

        # Should be identical with no mutation
        assert child_traits.cooperation_tendency == 0.7
        assert child_traits.curiosity == 0.3

    def test_trait_bounds_clamping(self):
        """Test that mutated traits are clamped to [0.0, 1.0]."""
        genetics = GeneticSystem(mutation_rate=1.0, mutation_magnitude=2.0)

        parent_traits = PersonalityTraits(
            cooperation_tendency=0.01,  # Low value, likely to go negative
            curiosity=0.99,  # High value, likely to exceed 1.0
        )

        child_traits = genetics.inherit(parent_traits)

        # Must be clamped
        assert 0.0 <= child_traits.cooperation_tendency <= 1.0
        assert 0.0 <= child_traits.curiosity <= 1.0

    def test_crossover(self):
        """Test crossover between two parents."""
        genetics = GeneticSystem(mutation_rate=0.0)

        parent_a = PersonalityTraits(
            cooperation_tendency=1.0,
            curiosity=0.0,
        )

        parent_b = PersonalityTraits(
            cooperation_tendency=0.0,
            curiosity=1.0,
        )

        child_traits = genetics.crossover(parent_a, parent_b)

        # Child should have mix of parent traits (with no mutation)
        assert 0.0 <= child_traits.cooperation_tendency <= 1.0
        assert 0.0 <= child_traits.curiosity <= 1.0

    def test_fitness_calculation(self):
        """Test fitness scoring."""
        genetics = GeneticSystem()

        # High fitness: long-lived, well-stocked, healthy
        agent = Agent(
            ticks_alive=500,
            needs=AgentNeeds(health=90.0),
            inventory={"food": 20, "water": 15},
            alive=True,
        )

        fitness = genetics.fitness(agent)
        assert fitness > 0.5

        # Low fitness: short-lived, empty inventory
        agent = Agent(
            ticks_alive=10,
            needs=AgentNeeds(health=50.0),
            inventory={},
            alive=True,
        )

        fitness = genetics.fitness(agent)
        assert 0.0 <= fitness < 0.3

    def test_fitness_range(self):
        """Test that fitness is always in valid range."""
        genetics = GeneticSystem()

        agent = Agent(
            ticks_alive=10000,  # Extremely long-lived
            needs=AgentNeeds(health=100.0),
            inventory={"food": 1000},
            alive=True,
        )

        fitness = genetics.fitness(agent)
        assert fitness >= 0.0

    def test_dead_agent_fitness(self):
        """Test fitness calculation for dead agents."""
        genetics = GeneticSystem()

        agent = Agent(
            ticks_alive=200,
            needs=AgentNeeds(health=0.0),
            alive=False,
        )

        fitness = genetics.fitness(agent)
        assert 0.0 < fitness < 1.0


class TestReproductionSystem:
    """Test reproduction mechanics."""

    def test_can_reproduce_all_conditions_met(self):
        """Test reproduction when all conditions are met."""
        reproduction = ReproductionSystem(
            min_age=100,
            fitness_threshold=0.6,
            max_population=10,
        )

        agent = Agent(
            ticks_alive=150,
            needs=AgentNeeds(energy=80.0, hunger=70.0),
            alive=True,
        )

        can_repro = reproduction.can_reproduce(
            agent=agent,
            fitness=0.7,
            population_count=5,
            tick=200,
        )

        assert can_repro is True

    def test_cannot_reproduce_too_young(self):
        """Test that young agents cannot reproduce."""
        reproduction = ReproductionSystem(min_age=100)

        agent = Agent(
            ticks_alive=50,
            needs=AgentNeeds(energy=80.0, hunger=70.0),
            alive=True,
        )

        can_repro = reproduction.can_reproduce(
            agent=agent,
            fitness=0.8,
            population_count=5,
            tick=100,
        )

        assert can_repro is False

    def test_cannot_reproduce_low_fitness(self):
        """Test that low fitness prevents reproduction."""
        reproduction = ReproductionSystem(fitness_threshold=0.6)

        agent = Agent(
            ticks_alive=150,
            needs=AgentNeeds(energy=80.0, hunger=70.0),
            alive=True,
        )

        can_repro = reproduction.can_reproduce(
            agent=agent,
            fitness=0.4,
            population_count=5,
            tick=200,
        )

        assert can_repro is False

    def test_cannot_reproduce_population_cap(self):
        """Test that population cap prevents reproduction."""
        reproduction = ReproductionSystem(max_population=10)

        agent = Agent(
            ticks_alive=150,
            needs=AgentNeeds(energy=80.0, hunger=70.0),
            alive=True,
        )

        can_repro = reproduction.can_reproduce(
            agent=agent,
            fitness=0.8,
            population_count=10,
            tick=200,
        )

        assert can_repro is False

    def test_cannot_reproduce_insufficient_energy(self):
        """Test that low energy prevents reproduction."""
        reproduction = ReproductionSystem(energy_cost=20.0)

        agent = Agent(
            ticks_alive=150,
            needs=AgentNeeds(energy=10.0, hunger=70.0),
            alive=True,
        )

        can_repro = reproduction.can_reproduce(
            agent=agent,
            fitness=0.8,
            population_count=5,
            tick=200,
        )

        assert can_repro is False

    def test_reproduce_creates_offspring(self):
        """Test offspring creation."""
        genetics = GeneticSystem()
        reproduction = ReproductionSystem()

        parent_id = AgentID("parent1")
        parent = Agent(
            x=20,
            y=20,
            ticks_alive=200,
            needs=AgentNeeds(energy=80.0, hunger=70.0),
            alive=True,
            agent_id=parent_id,
            profile=AgentProfile(
                agent_id=parent_id,
                name="Parent",
                archetype="gatherer",
                traits=PersonalityTraits(cooperation_tendency=0.7),
            ),
        )

        offspring_data = reproduction.reproduce(parent, genetics, tick=300)

        assert offspring_data is not None
        assert "parent_id" in offspring_data
        assert "child_id" in offspring_data
        assert "child_profile" in offspring_data
        assert "position" in offspring_data
        assert offspring_data["parent_id"] == str(parent_id)

        # Child should be near parent
        child_x, child_y = offspring_data["position"]
        assert abs(child_x - parent.x) <= 2
        assert abs(child_y - parent.y) <= 2

    def test_reproduce_costs_resources(self):
        """Test that reproduction costs energy and hunger."""
        genetics = GeneticSystem()
        reproduction = ReproductionSystem(energy_cost=20.0, hunger_cost=10.0)

        parent = Agent(
            ticks_alive=200,
            needs=AgentNeeds(energy=80.0, hunger=70.0),
            alive=True,
            profile=AgentProfile(
                agent_id=AgentID("parent1"),
                name="Parent",
                archetype="gatherer",
                traits=PersonalityTraits(),
            ),
        )

        initial_energy = parent.needs.energy
        initial_hunger = parent.needs.hunger

        reproduction.reproduce(parent, genetics, tick=300)

        assert parent.needs.energy < initial_energy
        assert parent.needs.hunger < initial_hunger

    def test_reproduction_probability(self):
        """Test reproduction probability calculation."""
        reproduction = ReproductionSystem(min_age=100, fitness_threshold=0.6)

        agent = Agent(ticks_alive=150, alive=True)

        # Below threshold: 0 probability
        prob = reproduction.calculate_reproduction_probability(agent, fitness=0.5)
        assert prob == 0.0

        # Above threshold: positive probability
        prob = reproduction.calculate_reproduction_probability(agent, fitness=0.8)
        assert 0.0 < prob <= 0.05


class TestCulturalTransmission:
    """Test cultural learning."""

    def test_observe_and_learn_from_success(self):
        """Test learning from successful agent."""
        culture = CulturalTransmission(learning_rate=0.1)

        observer = PersonalityTraits(cooperation_tendency=0.3, curiosity=0.5)
        model = PersonalityTraits(cooperation_tendency=0.8, curiosity=0.7)

        new_traits = culture.observe_and_learn(
            observer_traits=observer,
            model_traits=model,
            model_success=True,
            observer_openness=1.0,
        )

        # Traits should shift toward model
        assert new_traits.cooperation_tendency > observer.cooperation_tendency
        assert new_traits.curiosity > observer.curiosity

    def test_no_learning_from_failure(self):
        """Test no learning from failed agent."""
        culture = CulturalTransmission(learning_rate=0.1)

        observer = PersonalityTraits(cooperation_tendency=0.3)
        model = PersonalityTraits(cooperation_tendency=0.8)

        new_traits = culture.observe_and_learn(
            observer_traits=observer,
            model_traits=model,
            model_success=False,
            observer_openness=1.0,
        )

        # No change
        assert new_traits.cooperation_tendency == observer.cooperation_tendency

    def test_learning_rate_effect(self):
        """Test that learning rate affects magnitude of change."""
        culture_fast = CulturalTransmission(learning_rate=0.5)
        culture_slow = CulturalTransmission(learning_rate=0.01)

        observer = PersonalityTraits(cooperation_tendency=0.3)
        model = PersonalityTraits(cooperation_tendency=0.8)

        fast_traits = culture_fast.observe_and_learn(
            observer, model, model_success=True, observer_openness=1.0
        )
        slow_traits = culture_slow.observe_and_learn(
            observer, model, model_success=True, observer_openness=1.0
        )

        # Fast learning should change more
        fast_delta = abs(fast_traits.cooperation_tendency - observer.cooperation_tendency)
        slow_delta = abs(slow_traits.cooperation_tendency - observer.cooperation_tendency)

        assert fast_delta > slow_delta

    def test_propagate_norm(self):
        """Test norm propagation to agents."""
        culture = CulturalTransmission()

        norm = CulturalNorm(
            norm_type="cooperation",
            behavior_pattern="near_food:share",
            strength=0.8,
        )

        traits = PersonalityTraits(sociability=0.9)

        adopted = culture.propagate_norm(
            norm=norm,
            agent_id="agent1",
            agent_traits=traits,
            social_influence=0.8,
        )

        # High likelihood with high sociability and strong norm
        # Note: probabilistic, so we can't assert True, but we ran the logic
        assert isinstance(adopted, bool)


class TestNormDetector:
    """Test norm detection."""

    def test_record_and_detect_norm(self):
        """Test recording actions and detecting norms."""
        detector = NormDetector(detection_threshold=3, strength_threshold=0.5)

        # Record repeated pattern
        for _ in range(5):
            detector.record_action("agent1", "share", "near_food", tick=100)
            detector.record_action("agent2", "share", "near_food", tick=100)

        norms = detector.detect(population_size=2, tick=100)

        # Should detect cooperation norm
        assert len(norms) > 0
        norm = norms[0]
        assert norm.behavior_pattern == "near_food:share"
        assert norm.strength >= 0.5

    def test_no_detection_below_threshold(self):
        """Test no detection when pattern is weak."""
        detector = NormDetector(detection_threshold=5, strength_threshold=0.5)

        # Only 2 occurrences (below threshold)
        detector.record_action("agent1", "share", "near_food", tick=100)
        detector.record_action("agent1", "share", "near_food", tick=101)

        norms = detector.detect(population_size=5, tick=100)

        # Should not detect (too few occurrences)
        assert len(norms) == 0

    def test_norm_classification(self):
        """Test norm type classification."""
        detector = NormDetector(detection_threshold=3, strength_threshold=0.3)

        # Cooperation pattern
        for _ in range(5):
            detector.record_action("agent1", "share", "near_agent", tick=100)
            detector.record_action("agent2", "share", "near_agent", tick=100)

        norms = detector.detect(population_size=2, tick=100)

        assert len(norms) > 0
        assert norms[0].norm_type == "cooperation"

    def test_clear_weak_norms(self):
        """Test removing weak norms."""
        detector = NormDetector()

        # Create norms with different strengths
        norm1 = CulturalNorm("cooperation", "pattern1", strength=0.8)
        norm2 = CulturalNorm("aggression", "pattern2", strength=0.05)

        detector._detected_norms = [norm1, norm2]

        detector.clear_weak_norms(min_strength=0.1)

        # Only strong norm should remain
        assert len(detector._detected_norms) == 1
        assert detector._detected_norms[0].norm_type == "cooperation"


class TestLineageTracker:
    """Test lineage and ancestry tracking."""

    def test_record_birth(self):
        """Test recording agent birth."""
        tracker = LineageTracker()

        traits = PersonalityTraits(cooperation_tendency=0.7)

        tracker.record_birth(
            agent_id="agent1",
            parent_id=None,
            traits=traits,
            birth_tick=0,
        )

        # Should be recorded
        assert "agent1" in tracker._lineages
        node = tracker._lineages["agent1"]
        assert node.agent_id == "agent1"
        assert node.parent_id is None
        assert node.birth_tick == 0

    def test_record_birth_with_parent(self):
        """Test recording birth with parent."""
        tracker = LineageTracker()

        parent_traits = PersonalityTraits(cooperation_tendency=0.6)
        child_traits = PersonalityTraits(cooperation_tendency=0.7)

        tracker.record_birth("parent1", None, parent_traits, birth_tick=0)
        tracker.record_birth("child1", "parent1", child_traits, birth_tick=100)

        # Parent should have child
        parent_node = tracker._lineages["parent1"]
        assert "child1" in parent_node.children

        # Child should have parent
        child_node = tracker._lineages["child1"]
        assert child_node.parent_id == "parent1"

    def test_record_death(self):
        """Test recording agent death."""
        tracker = LineageTracker()

        traits = PersonalityTraits()
        tracker.record_birth("agent1", None, traits, birth_tick=0)

        tracker.record_death("agent1", death_tick=500, final_fitness=0.8)

        node = tracker._lineages["agent1"]
        assert node.death_tick == 500
        assert node.fitness == 0.8

    def test_get_lineage(self):
        """Test retrieving complete lineage."""
        tracker = LineageTracker()

        traits = PersonalityTraits()

        tracker.record_birth("gen0", None, traits, birth_tick=0)
        tracker.record_birth("gen1", "gen0", traits, birth_tick=100)
        tracker.record_birth("gen2", "gen1", traits, birth_tick=200)

        lineage = tracker.get_lineage("gen2")

        assert len(lineage) == 3
        assert lineage[0].agent_id == "gen0"
        assert lineage[1].agent_id == "gen1"
        assert lineage[2].agent_id == "gen2"

    def test_get_descendants(self):
        """Test retrieving all descendants."""
        tracker = LineageTracker()

        traits = PersonalityTraits()

        tracker.record_birth("parent", None, traits, birth_tick=0)
        tracker.record_birth("child1", "parent", traits, birth_tick=100)
        tracker.record_birth("child2", "parent", traits, birth_tick=100)
        tracker.record_birth("grandchild", "child1", traits, birth_tick=200)

        descendants = tracker.get_descendants("parent")

        assert len(descendants) == 3
        assert "child1" in descendants
        assert "child2" in descendants
        assert "grandchild" in descendants

    def test_trait_drift(self):
        """Test tracking trait evolution."""
        tracker = LineageTracker()

        traits_gen0 = PersonalityTraits(cooperation_tendency=0.5)
        traits_gen1 = PersonalityTraits(cooperation_tendency=0.6)
        traits_gen2 = PersonalityTraits(cooperation_tendency=0.7)

        tracker.record_birth("gen0", None, traits_gen0, birth_tick=0)
        tracker.record_birth("gen1", "gen0", traits_gen1, birth_tick=100)
        tracker.record_birth("gen2", "gen1", traits_gen2, birth_tick=200)

        drift = tracker.trait_drift("gen2", "cooperation_tendency")

        assert len(drift) == 3
        assert drift[0] == (0, 0.5)
        assert drift[1] == (100, 0.6)
        assert drift[2] == (200, 0.7)

    def test_most_successful_lineage(self):
        """Test finding most successful lineage."""
        tracker = LineageTracker()

        traits = PersonalityTraits()

        # Lineage 1: 1 child
        tracker.record_birth("root1", None, traits, birth_tick=0)
        tracker.record_birth("child1", "root1", traits, birth_tick=100)

        # Lineage 2: 3 children
        tracker.record_birth("root2", None, traits, birth_tick=0)
        tracker.record_birth("child2a", "root2", traits, birth_tick=100)
        tracker.record_birth("child2b", "root2", traits, birth_tick=100)
        tracker.record_birth("grandchild2", "child2a", traits, birth_tick=200)

        lineage = tracker.most_successful_lineage()

        # Should be lineage 2
        assert len(lineage) > 0
        assert lineage[0].agent_id == "root2"

    def test_generation_stats(self):
        """Test generation statistics."""
        tracker = LineageTracker()

        traits1 = PersonalityTraits(cooperation_tendency=0.6, curiosity=0.4)
        traits2 = PersonalityTraits(cooperation_tendency=0.8, curiosity=0.6)

        tracker.record_birth("agent1", None, traits1, birth_tick=0)
        tracker.record_birth("agent2", None, traits2, birth_tick=0)

        stats = tracker.get_generation_stats(tick=100)

        assert stats["population"] == 2
        assert "avg_traits" in stats
        assert "trait_diversity" in stats

        # Average cooperation should be 0.7
        assert 0.6 <= stats["avg_traits"]["cooperation_tendency"] <= 0.8


class TestPopulationManager:
    """Test population lifecycle management."""

    def test_population_manager_creation(self):
        """Test creating population manager."""
        genetics = GeneticSystem()
        reproduction = ReproductionSystem()
        manager = PopulationManager(genetics, reproduction)

        assert manager.genetics is genetics
        assert manager.reproduction is reproduction

    def test_get_population_stats(self):
        """Test retrieving population statistics."""
        genetics = GeneticSystem()
        reproduction = ReproductionSystem()
        manager = PopulationManager(genetics, reproduction)

        stats = manager.get_population_stats()

        assert "total_births" in stats
        assert "total_deaths" in stats
        assert "current_population" in stats

    def test_population_history_tracking(self):
        """Test that population history is tracked."""
        genetics = GeneticSystem()
        reproduction = ReproductionSystem()
        manager = PopulationManager(genetics, reproduction)

        history = manager.get_population_history()

        assert isinstance(history, list)
