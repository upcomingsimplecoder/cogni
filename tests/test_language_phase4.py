"""Comprehensive tests for Phase 4: Emergent Language system.

Tests cover:
- Symbol, SymbolMeaning, SymbolAssociation basic operations
- Lexicon: production, comprehension, decay, capacity, agreement
- LanguageEngine: innovation, communication, grounding, evolution, conventions
- LinguisticEvolutionAnalyzer: metrics and reporting
- Integration with SimulationEngine (engine wiring)
- Edge cases: empty lexicons, dead agents, zero vocabulary, bandwidth limits
"""

from __future__ import annotations

import random

import pytest

from src.analysis.linguistic import LinguisticEvolutionAnalyzer
from src.communication.language import (
    CommunicationOutcome,
    Lexicon,
    MeaningType,
    SharedConvention,
    Symbol,
    SymbolAssociation,
    SymbolMeaning,
    SymbolMessage,
)
from src.communication.language_engine import (
    ACTION_MEANINGS,
    CONTEXT_MEANINGS,
    DIRECTION_MEANINGS,
    LanguageEngine,
    LanguageSnapshot,
)

# ========================================================================
# Symbol Tests
# ========================================================================


class TestSymbol:
    """Tests for the Symbol dataclass."""

    def test_create_symbol(self):
        sym = Symbol(form="grp", created_tick=10, creator_id="agent-1")
        assert sym.form == "grp"
        assert sym.created_tick == 10
        assert sym.creator_id == "agent-1"
        assert sym.times_used == 0
        assert sym.times_understood == 0

    def test_success_rate_no_usage(self):
        sym = Symbol(form="foo")
        assert sym.success_rate == 0.5  # uninformative prior

    def test_success_rate_partial(self):
        sym = Symbol(form="foo", times_used=10, times_understood=7)
        assert sym.success_rate == pytest.approx(0.7)

    def test_success_rate_perfect(self):
        sym = Symbol(form="bar", times_used=5, times_understood=5)
        assert sym.success_rate == 1.0

    def test_success_rate_zero(self):
        sym = Symbol(form="baz", times_used=5, times_understood=0)
        assert sym.success_rate == 0.0

    def test_generate_novel(self):
        sym = Symbol.generate_novel(tick=5, creator_id="agent-2", length=3)
        assert len(sym.form) == 3
        assert sym.created_tick == 5
        assert sym.creator_id == "agent-2"

    def test_generate_novel_length(self):
        sym = Symbol.generate_novel(tick=0, creator_id="a", length=5)
        assert len(sym.form) == 5

    def test_generate_novel_randomness(self):
        """Two generated symbols should usually differ."""
        random.seed(42)
        s1 = Symbol.generate_novel(tick=0, creator_id="a")
        s2 = Symbol.generate_novel(tick=0, creator_id="a")
        # Very unlikely to be the same (20*5*20 = 2000 combos for 3 chars)
        # but not impossible; just check they're valid
        assert len(s1.form) == 3
        assert len(s2.form) == 3


# ========================================================================
# SymbolMeaning Tests
# ========================================================================


class TestSymbolMeaning:
    """Tests for the SymbolMeaning dataclass."""

    def test_create_meaning(self):
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        assert m.meaning_type == MeaningType.ACTION
        assert m.referent == "gather"
        assert m.confidence == 0.5

    def test_matches_same(self):
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.ACTION, "gather")
        assert m1.matches(m2)

    def test_matches_different_referent(self):
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.ACTION, "move")
        assert not m1.matches(m2)

    def test_matches_different_type(self):
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.CONTEXT, "gather")
        assert not m1.matches(m2)

    def test_similarity_exact(self):
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.ACTION, "gather")
        assert m1.similarity(m2) == 1.0

    def test_similarity_same_type(self):
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.ACTION, "move")
        assert m1.similarity(m2) == 0.5

    def test_similarity_different(self):
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.LOCATION, "north")
        assert m1.similarity(m2) == 0.0


# ========================================================================
# SymbolAssociation Tests
# ========================================================================


class TestSymbolAssociation:
    """Tests for the SymbolAssociation dataclass."""

    def test_reinforce(self):
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        assoc = SymbolAssociation(symbol=sym, meaning=meaning, strength=0.5)

        assoc.reinforce(amount=0.1, tick=10)
        assert assoc.strength == pytest.approx(0.6)
        assert assoc.times_reinforced == 1
        assert assoc.last_used_tick == 10

    def test_reinforce_cap(self):
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        assoc = SymbolAssociation(symbol=sym, meaning=meaning, strength=0.95)

        assoc.reinforce(amount=0.2)
        assert assoc.strength == 1.0  # capped

    def test_weaken(self):
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        assoc = SymbolAssociation(symbol=sym, meaning=meaning, strength=0.5)

        assoc.weaken(amount=0.1)
        assert assoc.strength == pytest.approx(0.4)
        assert assoc.times_weakened == 1

    def test_weaken_floor(self):
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        assoc = SymbolAssociation(symbol=sym, meaning=meaning, strength=0.02)

        assoc.weaken(amount=0.1)
        assert assoc.strength == 0.0  # floored

    def test_decay(self):
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        assoc = SymbolAssociation(symbol=sym, meaning=meaning, strength=0.5)

        assoc.decay(rate=0.01)
        assert assoc.strength == pytest.approx(0.49)


# ========================================================================
# Lexicon Tests
# ========================================================================


class TestLexicon:
    """Tests for the Lexicon class."""

    def test_empty_lexicon(self):
        lex = Lexicon("agent-1")
        assert lex.vocabulary_size() == 0
        assert lex.convention_count() == 0
        assert lex.all_symbols() == []

    def test_add_association(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")

        assoc = lex.add_association(sym, meaning, initial_strength=0.6)
        assert assoc.strength == 0.6
        assert lex.vocabulary_size() == 1
        assert lex.knows_symbol("grp")

    def test_add_duplicate_reinforces(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")

        lex.add_association(sym, meaning, initial_strength=0.5)
        assoc = lex.add_association(sym, meaning, initial_strength=0.5)
        assert assoc.times_reinforced == 1  # reinforced, not duplicated
        assert lex.vocabulary_size() == 1

    def test_produce(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        lex.add_association(sym, meaning, initial_strength=0.7)

        result = lex.produce(meaning)
        assert result is not None
        assert result.form == "grp"

    def test_produce_no_match(self):
        lex = Lexicon("agent-1")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        assert lex.produce(meaning) is None

    def test_produce_weak_association_returns_none(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        lex.add_association(sym, meaning, initial_strength=0.05)

        assert lex.produce(meaning) is None  # too weak

    def test_comprehend(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        lex.add_association(sym, meaning, initial_strength=0.7)

        result = lex.comprehend("grp")
        assert result is not None
        assert result.referent == "gather"

    def test_comprehend_unknown_symbol(self):
        lex = Lexicon("agent-1")
        assert lex.comprehend("xyz") is None

    def test_produce_selects_strongest(self):
        lex = Lexicon("agent-1")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")

        sym1 = Symbol(form="grp")
        sym2 = Symbol(form="gat")

        lex.add_association(sym1, meaning, initial_strength=0.3)
        lex.add_association(sym2, meaning, initial_strength=0.8)

        result = lex.produce(meaning)
        assert result is not None
        assert result.form == "gat"  # stronger association

    def test_comprehend_selects_strongest(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp")

        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.ACTION, "move")

        lex.add_association(sym, m1, initial_strength=0.3)
        lex.add_association(sym, m2, initial_strength=0.8)

        result = lex.comprehend("grp")
        assert result is not None
        assert result.referent == "move"  # stronger association

    def test_decay_all_removes_weak(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        _assoc = lex.add_association(sym, meaning, initial_strength=0.02)

        removed = lex.decay_all(rate=0.02)
        assert removed == 1
        assert lex.vocabulary_size() == 0

    def test_decay_all_preserves_strong(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        lex.add_association(sym, meaning, initial_strength=0.8)

        removed = lex.decay_all(rate=0.005)
        assert removed == 0
        assert lex.vocabulary_size() == 1

    def test_convention_count(self):
        lex = Lexicon("agent-1")
        sym1 = Symbol(form="grp")
        sym2 = Symbol(form="mvv")

        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.ACTION, "move")

        lex.add_association(sym1, m1, initial_strength=0.7)  # convention
        lex.add_association(sym2, m2, initial_strength=0.3)  # not convention

        assert lex.convention_count() == 1

    def test_shared_symbols_with(self):
        lex1 = Lexicon("agent-1")
        lex2 = Lexicon("agent-2")

        sym_a = Symbol(form="grp")
        sym_b = Symbol(form="mvv")
        sym_c = Symbol(form="xyz")

        m = SymbolMeaning(MeaningType.ACTION, "gather")

        lex1.add_association(sym_a, m, initial_strength=0.5)
        lex1.add_association(sym_b, m, initial_strength=0.5)
        lex2.add_association(sym_a, m, initial_strength=0.5)
        lex2.add_association(sym_c, m, initial_strength=0.5)

        shared = lex1.shared_symbols_with(lex2)
        assert shared == {"grp"}

    def test_agreement_score_full(self):
        lex1 = Lexicon("agent-1")
        lex2 = Lexicon("agent-2")

        sym = Symbol(form="grp")
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")

        lex1.add_association(sym, meaning, initial_strength=0.7)
        lex2.add_association(sym, meaning, initial_strength=0.7)

        assert lex1.agreement_score(lex2) == 1.0

    def test_agreement_score_disagreement(self):
        lex1 = Lexicon("agent-1")
        lex2 = Lexicon("agent-2")

        sym = Symbol(form="grp")
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.ACTION, "move")

        lex1.add_association(sym, m1, initial_strength=0.7)
        lex2.add_association(sym, m2, initial_strength=0.7)

        assert lex1.agreement_score(lex2) == 0.0

    def test_agreement_score_no_shared(self):
        lex1 = Lexicon("agent-1")
        lex2 = Lexicon("agent-2")
        assert lex1.agreement_score(lex2) == 0.0

    def test_capacity_enforcement(self):
        lex = Lexicon("agent-1")
        for i in range(250):
            sym = Symbol(form=f"s{i:04d}")
            meaning = SymbolMeaning(MeaningType.ACTION, f"action_{i}")
            lex.add_association(sym, meaning, initial_strength=random.uniform(0.1, 1.0))

        assert len(lex._associations) <= Lexicon.MAX_ASSOCIATIONS

    def test_to_dict(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp", times_used=5)
        meaning = SymbolMeaning(MeaningType.ACTION, "gather")
        lex.add_association(sym, meaning, initial_strength=0.7)

        d = lex.to_dict()
        assert d["vocabulary_size"] == 1
        assert d["convention_count"] == 1
        assert "grp" in d["symbols"]

    def test_get_associations_for_symbol(self):
        lex = Lexicon("agent-1")
        sym = Symbol(form="grp")
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.CONTEXT, "hungry")

        lex.add_association(sym, m1, initial_strength=0.5)
        lex.add_association(sym, m2, initial_strength=0.3)

        assocs = lex.get_associations_for_symbol("grp")
        assert len(assocs) == 2


# ========================================================================
# SymbolMessage & CommunicationOutcome Tests
# ========================================================================


class TestSymbolMessage:
    """Tests for SymbolMessage."""

    def test_create_message(self):
        msg = SymbolMessage(
            sender_id="a1",
            receiver_id="a2",
            symbols=("grp", "nrth"),
            context="low_hunger",
            tick=10,
        )
        assert msg.symbol_count == 2
        assert msg.sender_id == "a1"
        assert msg.energy_cost == 0.5

    def test_broadcast_message(self):
        msg = SymbolMessage(
            sender_id="a1",
            receiver_id=None,
            symbols=("grp",),
            context="danger",
            tick=5,
        )
        assert msg.receiver_id is None
        assert msg.symbol_count == 1


class TestCommunicationOutcome:
    """Tests for CommunicationOutcome."""

    def test_successful_outcome(self):
        msg = SymbolMessage("a1", "a2", ("grp",), "ctx", 10)
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        outcome = CommunicationOutcome(
            message=msg,
            intended_meanings=[m],
            interpreted_meanings=[m],
            success=True,
            tick=10,
        )
        assert outcome.interpretation_accuracy == 1.0

    def test_failed_outcome(self):
        msg = SymbolMessage("a1", "a2", ("grp",), "ctx", 10)
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.ACTION, "move")
        outcome = CommunicationOutcome(
            message=msg,
            intended_meanings=[m1],
            interpreted_meanings=[m2],
            success=False,
            tick=10,
        )
        assert outcome.interpretation_accuracy == 0.0

    def test_partial_outcome(self):
        msg = SymbolMessage("a1", "a2", ("grp", "nrth"), "ctx", 10)
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        m2 = SymbolMeaning(MeaningType.LOCATION, "north")
        outcome = CommunicationOutcome(
            message=msg,
            intended_meanings=[m1, m2],
            interpreted_meanings=[m1, None],
            success=True,
            tick=10,
        )
        assert outcome.interpretation_accuracy == pytest.approx(0.5)

    def test_empty_outcome(self):
        msg = SymbolMessage("a1", "a2", (), "ctx", 10)
        outcome = CommunicationOutcome(
            message=msg,
            intended_meanings=[],
            interpreted_meanings=[],
            success=False,
            tick=10,
        )
        assert outcome.interpretation_accuracy == 0.0


# ========================================================================
# SharedConvention Tests
# ========================================================================


class TestSharedConvention:
    """Tests for SharedConvention."""

    def test_adoption_count(self):
        conv = SharedConvention(
            symbol_form="grp",
            meaning=SymbolMeaning(MeaningType.ACTION, "gather"),
            adopters={"a1", "a2", "a3"},
        )
        assert conv.adoption_count == 3

    def test_is_established(self):
        conv = SharedConvention(
            symbol_form="grp",
            meaning=SymbolMeaning(MeaningType.ACTION, "gather"),
            adopters={"a1", "a2", "a3"},
        )
        assert conv.is_established(min_adopters=3)
        assert not conv.is_established(min_adopters=4)

    def test_is_established_empty(self):
        conv = SharedConvention(
            symbol_form="grp",
            meaning=SymbolMeaning(MeaningType.ACTION, "gather"),
        )
        assert not conv.is_established()


# ========================================================================
# LanguageEngine Tests
# ========================================================================


class TestLanguageEngine:
    """Tests for the LanguageEngine orchestrator."""

    def test_init_defaults(self):
        engine = LanguageEngine()
        assert engine.innovation_rate == 0.05
        assert engine.bandwidth_limit == 3
        assert engine.communication_range == 8

    def test_register_agent(self):
        engine = LanguageEngine()
        engine.register_agent("agent-1")
        assert engine.get_lexicon("agent-1") is not None
        assert engine.get_lexicon("agent-1").vocabulary_size() == 0

    def test_register_agent_idempotent(self):
        engine = LanguageEngine()
        engine.register_agent("agent-1")
        lex = engine.get_lexicon("agent-1")

        # Add something to lexicon
        sym = Symbol(form="grp")
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        lex.add_association(sym, m, initial_strength=0.5)

        # Re-register should not reset
        engine.register_agent("agent-1")
        assert engine.get_lexicon("agent-1").vocabulary_size() == 1

    def test_get_lexicon_unknown(self):
        engine = LanguageEngine()
        assert engine.get_lexicon("nonexistent") is None

    def test_get_communication_success_rate_no_history(self):
        engine = LanguageEngine()
        engine.register_agent("agent-1")
        assert engine.get_communication_success_rate("agent-1") == 0.0

    def test_get_language_stats_empty(self):
        engine = LanguageEngine()
        stats = engine.get_language_stats()
        assert stats["total_vocabulary"] == 0
        assert stats["unique_symbols"] == 0
        assert stats["established_conventions"] == 0

    def test_get_language_stats_with_data(self):
        engine = LanguageEngine()
        engine.register_agent("a1")
        engine.register_agent("a2")

        lex1 = engine.get_lexicon("a1")
        sym = Symbol(form="grp")
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        lex1.add_association(sym, m, initial_strength=0.7)
        engine._all_symbols["grp"] = sym

        stats = engine.get_language_stats()
        assert stats["total_vocabulary"] == 1
        assert stats["agents_tracked"] == 2

    def test_get_established_conventions_empty(self):
        engine = LanguageEngine()
        assert engine.get_established_conventions() == []

    def test_convention_tracking(self):
        engine = LanguageEngine()
        sym = Symbol(form="grp")
        m = SymbolMeaning(MeaningType.ACTION, "gather")

        # Register 4 agents all with same symbol-meaning
        for i in range(4):
            aid = f"agent-{i}"
            engine.register_agent(aid)
            lex = engine.get_lexicon(aid)
            lex.add_association(sym, m, initial_strength=0.7)

        engine._update_conventions(tick=10)
        established = engine.get_established_conventions()
        assert len(established) == 1
        assert established[0].symbol_form == "grp"
        assert established[0].adoption_count == 4

    def test_convention_removed_when_strength_drops(self):
        engine = LanguageEngine()
        sym = Symbol(form="grp")
        m = SymbolMeaning(MeaningType.ACTION, "gather")

        for i in range(3):
            aid = f"agent-{i}"
            engine.register_agent(aid)
            lex = engine.get_lexicon(aid)
            lex.add_association(sym, m, initial_strength=0.7)

        engine._update_conventions(tick=10)
        assert len(engine.get_established_conventions()) == 1

        # Drop strength below threshold for all agents
        for i in range(3):
            lex = engine.get_lexicon(f"agent-{i}")
            for assoc in lex._associations:
                assoc.strength = 0.1

        engine._update_conventions(tick=20)
        assert len(engine.get_established_conventions()) == 0

    def test_dialect_detection_single_group(self):
        engine = LanguageEngine()
        sym = Symbol(form="grp")
        m = SymbolMeaning(MeaningType.ACTION, "gather")

        for i in range(4):
            aid = f"agent-{i}"
            engine.register_agent(aid)
            lex = engine.get_lexicon(aid)
            lex.add_association(sym, m, initial_strength=0.7)

        count = engine._detect_dialect_count()
        assert count == 1  # all share same vocabulary

    def test_dialect_detection_multiple_groups(self):
        engine = LanguageEngine()

        # Group 1: shares "grp"
        sym1 = Symbol(form="grp")
        m1 = SymbolMeaning(MeaningType.ACTION, "gather")
        for i in range(3):
            aid = f"group1-{i}"
            engine.register_agent(aid)
            lex = engine.get_lexicon(aid)
            lex.add_association(sym1, m1, initial_strength=0.7)

        # Group 2: shares "mvv" (different symbol, same meaning)
        sym2 = Symbol(form="mvv")
        for i in range(3):
            aid = f"group2-{i}"
            engine.register_agent(aid)
            lex = engine.get_lexicon(aid)
            lex.add_association(sym2, m1, initial_strength=0.7)

        count = engine._detect_dialect_count()
        assert count == 2  # two distinct vocabulary groups

    def test_process_grounding_success_reinforces(self):
        engine = LanguageEngine()
        engine.register_agent("sender")
        engine.register_agent("listener")

        sym = Symbol(form="grp")
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        engine._all_symbols["grp"] = sym

        sender_lex = engine.get_lexicon("sender")
        listener_lex = engine.get_lexicon("listener")
        sender_lex.add_association(sym, m, initial_strength=0.5)
        listener_lex.add_association(sym, m, initial_strength=0.5)

        msg = SymbolMessage("sender", "listener", ("grp",), "ctx", 10)
        outcome = CommunicationOutcome(
            message=msg,
            intended_meanings=[m],
            interpreted_meanings=[m],
            success=True,
            tick=10,
        )

        engine._process_grounding(outcome, 10)

        # Both should be reinforced
        sender_assoc = sender_lex._find_association("grp", m)
        listener_assoc = listener_lex._find_association("grp", m)
        assert sender_assoc.strength > 0.5
        assert listener_assoc.strength > 0.5

    def test_process_grounding_failure_weakens_and_teaches(self):
        engine = LanguageEngine()
        engine.register_agent("sender")
        engine.register_agent("listener")

        sym = Symbol(form="grp")
        m_intended = SymbolMeaning(MeaningType.ACTION, "gather")
        _m_wrong = SymbolMeaning(MeaningType.ACTION, "move")
        engine._all_symbols["grp"] = sym

        sender_lex = engine.get_lexicon("sender")
        sender_lex.add_association(sym, m_intended, initial_strength=0.5)

        listener_lex = engine.get_lexicon("listener")
        # Listener doesn't know this symbol yet

        msg = SymbolMessage("sender", "listener", ("grp",), "ctx", 10)
        outcome = CommunicationOutcome(
            message=msg,
            intended_meanings=[m_intended],
            interpreted_meanings=[None],
            success=False,
            tick=10,
        )

        engine._process_grounding(outcome, 10)

        # Sender should be weakened
        sender_assoc = sender_lex._find_association("grp", m_intended)
        assert sender_assoc.strength < 0.5

        # Listener should have learned the symbol
        assert listener_lex.knows_symbol("grp")

    def test_messages_this_tick_property(self):
        engine = LanguageEngine()
        assert engine.messages_this_tick == []

    def test_outcomes_this_tick_property(self):
        engine = LanguageEngine()
        assert engine.outcomes_this_tick == []

    def test_snapshots_property(self):
        engine = LanguageEngine()
        assert engine.snapshots == []


# ========================================================================
# Meaning Maps Tests
# ========================================================================


class TestMeaningMaps:
    """Tests for the context/action/direction meaning maps."""

    def test_context_meanings_exist(self):
        assert "low_hunger" in CONTEXT_MEANINGS
        assert "danger" in CONTEXT_MEANINGS
        assert "alone" in CONTEXT_MEANINGS

    def test_action_meanings_exist(self):
        assert "gather" in ACTION_MEANINGS
        assert "move" in ACTION_MEANINGS
        assert "attack" in ACTION_MEANINGS

    def test_direction_meanings_exist(self):
        assert "north" in DIRECTION_MEANINGS
        assert "south" in DIRECTION_MEANINGS

    def test_context_meanings_type(self):
        for m in CONTEXT_MEANINGS.values():
            assert m.meaning_type == MeaningType.CONTEXT

    def test_action_meanings_type(self):
        for _key, m in ACTION_MEANINGS.items():
            assert m.meaning_type in (MeaningType.ACTION, MeaningType.SOCIAL)


# ========================================================================
# LanguageSnapshot Tests
# ========================================================================


class TestLanguageSnapshot:
    """Tests for LanguageSnapshot dataclass."""

    def test_create_snapshot(self):
        snap = LanguageSnapshot(
            tick=10,
            total_vocabulary=15,
            unique_symbols=8,
            convention_count=3,
            avg_agreement=0.65,
            communication_success_rate=0.5,
            dialect_count=2,
            innovations_this_tick=1,
            adoptions_this_tick=2,
            deaths_this_tick=0,
        )
        assert snap.tick == 10
        assert snap.total_vocabulary == 15
        assert snap.convention_count == 3


# ========================================================================
# LinguisticEvolutionAnalyzer Tests
# ========================================================================


class TestLinguisticEvolutionAnalyzer:
    """Tests for the linguistic evolution analysis module."""

    def _make_snapshot(
        self,
        tick,
        vocab=5,
        symbols=3,
        conventions=1,
        agreement=0.5,
        success=0.4,
        dialects=1,
        innovations=1,
        adoptions=0,
        deaths=0,
    ):
        """Helper to create LanguageSnapshot."""
        return LanguageSnapshot(
            tick=tick,
            total_vocabulary=vocab,
            unique_symbols=symbols,
            convention_count=conventions,
            avg_agreement=agreement,
            communication_success_rate=success,
            dialect_count=dialects,
            innovations_this_tick=innovations,
            adoptions_this_tick=adoptions,
            deaths_this_tick=deaths,
        )

    def test_empty_analyzer(self):
        analyzer = LinguisticEvolutionAnalyzer()
        assert analyzer.vocabulary_growth_curve() == []
        assert analyzer.convention_formation_timeline() == []
        assert analyzer.communication_success_over_time() == []
        assert analyzer.dialect_divergence_over_time() == []
        assert analyzer.innovation_rate_over_time() == []

    def test_empty_report(self):
        analyzer = LinguisticEvolutionAnalyzer()
        report = analyzer.generate_report()
        assert "No data recorded" in report

    def test_vocabulary_growth_curve(self):
        analyzer = LinguisticEvolutionAnalyzer()

        # Simulate recording ticks with growing vocabulary
        for tick in range(5):
            snapshot = self._make_snapshot(tick, symbols=tick + 1)
            analyzer._history.append(
                {
                    "tick": tick,
                    "snapshots": [snapshot],
                    "established_conventions": [],
                    "all_conventions": [],
                    "language_stats": {},
                }
            )

        curve = analyzer.vocabulary_growth_curve()
        assert len(curve) == 5
        assert curve[0] == (0, 1)
        assert curve[4] == (4, 5)

    def test_communication_success_over_time(self):
        analyzer = LinguisticEvolutionAnalyzer()

        for tick in range(3):
            success = 0.2 * (tick + 1)
            snapshot = self._make_snapshot(tick, success=success)
            analyzer._history.append(
                {
                    "tick": tick,
                    "snapshots": [snapshot],
                    "established_conventions": [],
                    "all_conventions": [],
                    "language_stats": {},
                }
            )

        curve = analyzer.communication_success_over_time()
        assert len(curve) == 3
        assert curve[0] == (0, pytest.approx(0.2))
        assert curve[2] == (2, pytest.approx(0.6))

    def test_dialect_divergence_over_time(self):
        analyzer = LinguisticEvolutionAnalyzer()

        for tick in range(3):
            snapshot = self._make_snapshot(tick, dialects=tick + 1)
            analyzer._history.append(
                {
                    "tick": tick,
                    "snapshots": [snapshot],
                    "established_conventions": [],
                    "all_conventions": [],
                    "language_stats": {},
                }
            )

        curve = analyzer.dialect_divergence_over_time()
        assert len(curve) == 3
        assert curve[0] == (0, 1)
        assert curve[2] == (2, 3)

    def test_innovation_rate_over_time(self):
        analyzer = LinguisticEvolutionAnalyzer()

        for tick in range(3):
            snapshot = self._make_snapshot(tick, innovations=tick)
            analyzer._history.append(
                {
                    "tick": tick,
                    "snapshots": [snapshot],
                    "established_conventions": [],
                    "all_conventions": [],
                    "language_stats": {},
                }
            )

        curve = analyzer.innovation_rate_over_time()
        assert len(curve) == 3
        assert curve[0] == (0, 0)
        assert curve[2] == (2, 2)

    def test_convention_formation_timeline(self):
        analyzer = LinguisticEvolutionAnalyzer()
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        conv = SharedConvention(
            symbol_form="grp",
            meaning=m,
            adopters={"a1", "a2", "a3"},
            first_observed_tick=5,
        )

        analyzer._history.append(
            {
                "tick": 10,
                "snapshots": [],
                "established_conventions": [conv],
                "all_conventions": [conv],
                "language_stats": {},
            }
        )

        timeline = analyzer.convention_formation_timeline()
        assert len(timeline) == 1
        assert timeline[0]["symbol"] == "grp"
        assert timeline[0]["tick"] == 5
        assert timeline[0]["adopters"] == 3

    def test_convention_formation_deduplication(self):
        """Same convention should only appear once in timeline."""
        analyzer = LinguisticEvolutionAnalyzer()
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        conv = SharedConvention(
            symbol_form="grp",
            meaning=m,
            adopters={"a1", "a2", "a3"},
            first_observed_tick=5,
        )

        # Record same convention in two different ticks
        for tick in [10, 20]:
            analyzer._history.append(
                {
                    "tick": tick,
                    "snapshots": [],
                    "established_conventions": [conv],
                    "all_conventions": [conv],
                    "language_stats": {},
                }
            )

        timeline = analyzer.convention_formation_timeline()
        assert len(timeline) == 1  # deduplicated

    def test_generate_report_with_data(self):
        analyzer = LinguisticEvolutionAnalyzer()
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        conv = SharedConvention(
            symbol_form="grp",
            meaning=m,
            adopters={"a1", "a2", "a3"},
            first_observed_tick=5,
        )

        for tick in range(10):
            snapshot = self._make_snapshot(
                tick,
                symbols=tick + 1,
                success=0.3 + tick * 0.05,
                dialects=1,
                innovations=max(0, 2 - tick),
            )
            analyzer._history.append(
                {
                    "tick": tick,
                    "snapshots": [snapshot],
                    "established_conventions": [conv] if tick >= 5 else [],
                    "all_conventions": [conv] if tick >= 5 else [],
                    "language_stats": {},
                }
            )

        report = analyzer.generate_report()
        assert "# Linguistic Evolution Report" in report
        assert "Vocabulary Growth" in report
        assert "Convention Formation" in report
        assert "Communication Efficiency" in report
        assert "Dialect Formation" in report
        assert "Innovation" in report
        assert "Summary" in report

    def test_to_dict(self):
        analyzer = LinguisticEvolutionAnalyzer()
        d = analyzer.to_dict()
        assert "vocabulary_growth" in d
        assert "convention_timeline" in d
        assert "communication_success" in d
        assert "dialect_divergence" in d
        assert "innovation_rate" in d
        assert "raw_history" in d


# ========================================================================
# Integration Tests
# ========================================================================


class TestLanguageEngineIntegration:
    """Integration tests for the language engine with SimulationEngine."""

    def test_language_engine_created_when_enabled(self):
        from src.config import SimulationConfig

        config = SimulationConfig(language_enabled=True)
        from src.simulation.engine import SimulationEngine

        engine = SimulationEngine(config)
        assert engine.language_engine is not None

    def test_language_engine_not_created_when_disabled(self):
        from src.config import SimulationConfig

        config = SimulationConfig(language_enabled=False)
        from src.simulation.engine import SimulationEngine

        engine = SimulationEngine(config)
        assert engine.language_engine is None

    def test_language_config_wired(self):
        from src.config import SimulationConfig

        config = SimulationConfig(
            language_enabled=True,
            language_innovation_rate=0.1,
            language_bandwidth_limit=5,
            language_communication_range=12,
        )
        from src.simulation.engine import SimulationEngine

        engine = SimulationEngine(config)
        assert engine.language_engine.innovation_rate == 0.1
        assert engine.language_engine.bandwidth_limit == 5
        assert engine.language_engine.communication_range == 12

    def test_agents_registered_with_language_engine(self):
        from src.config import SimulationConfig

        config = SimulationConfig(
            language_enabled=True,
            num_agents=3,
            max_ticks=10,
            seed=42,
        )
        from src.simulation.engine import SimulationEngine

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # All agents should be registered
        living = engine.registry.living_agents()
        for agent in living:
            lex = engine.language_engine.get_lexicon(str(agent.agent_id))
            assert lex is not None

    def test_step_all_calls_language_tick(self):
        from src.config import SimulationConfig

        config = SimulationConfig(
            language_enabled=True,
            num_agents=3,
            max_ticks=10,
            seed=42,
        )
        from src.simulation.engine import SimulationEngine

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run a few ticks
        for _ in range(5):
            if engine.is_over():
                break
            engine.step_all()

        # Language engine should have been called
        assert len(engine.language_engine.snapshots) > 0

    def test_tick_to_json_includes_language_data(self):
        from src.config import SimulationConfig

        config = SimulationConfig(
            language_enabled=True,
            num_agents=3,
            max_ticks=10,
            seed=42,
        )
        from src.simulation.engine import SimulationEngine
        from src.visualization.realtime import tick_to_json

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        assert "language" in data
        assert "unique_symbols" in data["language"]
        assert "established_conventions" in data["language"]

    def test_tick_to_json_no_language_when_disabled(self):
        from src.config import SimulationConfig

        config = SimulationConfig(
            language_enabled=False,
            num_agents=3,
            max_ticks=10,
            seed=42,
        )
        from src.simulation.engine import SimulationEngine
        from src.visualization.realtime import tick_to_json

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        assert data["language"] == {}

    def test_language_config_defaults(self):
        from src.config import SimulationConfig

        config = SimulationConfig()
        assert config.language_enabled is False
        assert config.language_innovation_rate == 0.05
        assert config.language_bandwidth_limit == 3


# ========================================================================
# Edge Case Tests
# ========================================================================


class TestLanguageEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_lexicon_produce(self):
        lex = Lexicon("agent-1")
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        assert lex.produce(m) is None

    def test_empty_lexicon_comprehend(self):
        lex = Lexicon("agent-1")
        assert lex.comprehend("grp") is None

    def test_empty_lexicon_agreement(self):
        lex1 = Lexicon("a1")
        lex2 = Lexicon("a2")
        assert lex1.agreement_score(lex2) == 0.0

    def test_single_agent_dialect(self):
        engine = LanguageEngine()
        engine.register_agent("lonely")
        assert engine._detect_dialect_count() == 1

    def test_no_agents_dialect(self):
        engine = LanguageEngine()
        assert engine._detect_dialect_count() == 0

    def test_convention_update_empty_lexicons(self):
        engine = LanguageEngine()
        engine.register_agent("a1")
        engine._update_conventions(tick=0)
        assert engine.get_established_conventions() == []

    def test_symbol_generate_novel_deterministic(self):
        random.seed(123)
        s1 = Symbol.generate_novel(0, "a")
        random.seed(123)
        s2 = Symbol.generate_novel(0, "a")
        assert s1.form == s2.form

    def test_lexicon_remove_last_association(self):
        lex = Lexicon("a1")
        sym = Symbol(form="grp")
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        _assoc = lex.add_association(sym, m, initial_strength=0.02)

        removed = lex.decay_all(rate=0.03)
        assert removed == 1
        assert lex.vocabulary_size() == 0
        assert not lex.knows_symbol("grp")

    def test_grounding_with_missing_sender(self):
        engine = LanguageEngine()
        # No sender registered
        msg = SymbolMessage("ghost", "listener", ("grp",), "ctx", 10)
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        outcome = CommunicationOutcome(
            message=msg,
            intended_meanings=[m],
            interpreted_meanings=[None],
            success=False,
            tick=10,
        )
        # Should not crash
        engine._process_grounding(outcome, 10)

    def test_grounding_with_broadcast_message(self):
        engine = LanguageEngine()
        engine.register_agent("sender")
        sym = Symbol(form="grp")
        m = SymbolMeaning(MeaningType.ACTION, "gather")
        engine._all_symbols["grp"] = sym
        engine.get_lexicon("sender").add_association(sym, m, initial_strength=0.5)

        msg = SymbolMessage("sender", None, ("grp",), "ctx", 10)
        outcome = CommunicationOutcome(
            message=msg,
            intended_meanings=[m],
            interpreted_meanings=[None],
            success=False,
            tick=10,
        )
        # Should not crash (broadcast — no listener to teach)
        engine._process_grounding(outcome, 10)

    def test_meaning_type_enum_values(self):
        """Ensure all MeaningType enum values are accessible."""
        assert MeaningType.ACTION.value == "action"
        assert MeaningType.CONTEXT.value == "context"
        assert MeaningType.LOCATION.value == "location"
        assert MeaningType.AGENT.value == "agent"
        assert MeaningType.RESOURCE.value == "resource"
        assert MeaningType.SOCIAL.value == "social"
        assert MeaningType.COMPOSITE.value == "composite"

    def test_analyzer_to_dict_with_data(self):
        analyzer = LinguisticEvolutionAnalyzer()
        snap = LanguageSnapshot(
            tick=1,
            total_vocabulary=5,
            unique_symbols=3,
            convention_count=1,
            avg_agreement=0.5,
            communication_success_rate=0.4,
            dialect_count=1,
            innovations_this_tick=1,
            adoptions_this_tick=0,
            deaths_this_tick=0,
        )
        analyzer._history.append(
            {
                "tick": 1,
                "snapshots": [snap],
                "established_conventions": [],
                "all_conventions": [],
                "language_stats": {},
            }
        )
        d = analyzer.to_dict()
        assert len(d["vocabulary_growth"]) == 1
        assert len(d["innovation_rate"]) == 1
