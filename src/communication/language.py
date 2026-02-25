"""Emergent language system for agent communication.

Phase 4 of the AUTOCOG product vision. Implements:
- Symbol: Arbitrary tokens that gain meaning through use
- Lexicon: Agent's personal vocabulary (symbol → meaning mappings)
- SharedConvention: A symbol-meaning pair that multiple agents agree on
- SymbolMeaning: What a symbol refers to (context, action, location, etc.)

Symbols acquire meaning through referential games and repeated interaction.
Bandwidth pressure and communication cost drive compression/shorthand.

This is the deep tech moat — first emergent language system combining
LLM cognition + cultural transmission + ToM-grounded pragmatics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MeaningType(Enum):
    """Categories of what a symbol can mean."""

    ACTION = "action"  # A type of action (e.g., "gather", "move")
    CONTEXT = "context"  # A situational context (e.g., "low_hunger", "danger")
    LOCATION = "location"  # A spatial reference (e.g., "north", "food_area")
    AGENT = "agent"  # A reference to another agent
    RESOURCE = "resource"  # A resource type (e.g., "food", "water")
    SOCIAL = "social"  # Social intent (e.g., "help", "trade", "warning")
    COMPOSITE = "composite"  # Combination of meanings (emerging compositionality)


@dataclass
class SymbolMeaning:
    """What a symbol refers to.

    A symbol can map to one or more meanings. The referent is the
    concrete thing being referred to, while meaning_type classifies it.
    """

    meaning_type: MeaningType
    referent: str  # e.g., "gather", "north", "food"
    context: str = ""  # optional context qualifier
    confidence: float = 0.5  # how sure the agent is about this meaning

    def matches(self, other: SymbolMeaning) -> bool:
        """Check if two meanings refer to the same thing.

        Matches on type + referent. Context is optional qualifier.
        """
        return self.meaning_type == other.meaning_type and self.referent == other.referent

    def similarity(self, other: SymbolMeaning) -> float:
        """Compute semantic similarity between two meanings.

        Returns:
            1.0 for exact match, 0.5 for same type, 0.0 for unrelated.
        """
        if self.matches(other):
            return 1.0
        if self.meaning_type == other.meaning_type:
            return 0.5
        return 0.0


@dataclass
class Symbol:
    """An arbitrary token that gains meaning through use.

    Symbols start as novel tokens and acquire meaning via referential
    games and interaction feedback. Each symbol has a unique form
    (the "word") and associated meanings for the agent that holds it.
    """

    form: str  # The symbol's surface form (e.g., "grp", "nrth", "zq")
    created_tick: int = 0
    creator_id: str = ""
    times_used: int = 0
    times_understood: int = 0  # receiver correctly interpreted

    @property
    def success_rate(self) -> float:
        """How often this symbol is correctly interpreted."""
        if self.times_used == 0:
            return 0.5  # uninformative prior
        return self.times_understood / self.times_used

    @staticmethod
    def generate_novel(tick: int, creator_id: str, length: int = 3) -> Symbol:
        """Generate a novel symbol with random form.

        Creates short, pronounceable-ish tokens by alternating
        consonants and vowels.

        Args:
            tick: Current simulation tick
            creator_id: ID of the creating agent
            length: Number of characters (default 3)

        Returns:
            A new Symbol with random form
        """
        consonants = "bcdfghjklmnpqrstvwxz"
        vowels = "aeiou"
        form = ""
        for i in range(length):
            if i % 2 == 0:
                form += random.choice(consonants)
            else:
                form += random.choice(vowels)
        return Symbol(form=form, created_tick=tick, creator_id=creator_id)


@dataclass
class SymbolAssociation:
    """An agent's association between a symbol and a meaning.

    Strength is reinforced by successful communication and
    weakened by failures. Multiple meanings can associate
    with the same symbol (polysemy) or multiple symbols with
    the same meaning (synonymy).
    """

    symbol: Symbol
    meaning: SymbolMeaning
    strength: float = 0.5  # [0, 1] — bidirectional association strength
    times_reinforced: int = 0
    times_weakened: int = 0
    last_used_tick: int = 0

    def reinforce(self, amount: float = 0.1, tick: int = 0) -> None:
        """Strengthen association from successful communication."""
        self.strength = min(1.0, self.strength + amount)
        self.times_reinforced += 1
        self.last_used_tick = tick

    def weaken(self, amount: float = 0.05) -> None:
        """Weaken association from failed communication."""
        self.strength = max(0.0, self.strength - amount)
        self.times_weakened += 1

    def decay(self, rate: float = 0.005) -> None:
        """Natural decay of unused associations."""
        self.strength = max(0.0, self.strength - rate)


class Lexicon:
    """An agent's personal vocabulary: symbol ↔ meaning associations.

    Supports:
    - Production: meaning → symbol (what symbol to use for a meaning)
    - Comprehension: symbol → meaning (what a received symbol means)
    - Multiple associations per symbol/meaning (polysemy, synonymy)
    - Strength-based selection (strongest association wins)
    - Natural decay of unused associations
    """

    MAX_ASSOCIATIONS = 200

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._associations: list[SymbolAssociation] = []
        # Indexes for fast lookup
        self._by_symbol: dict[str, list[SymbolAssociation]] = {}
        self._by_meaning: dict[str, list[SymbolAssociation]] = {}

    def _meaning_key(self, meaning: SymbolMeaning) -> str:
        """Create a hashable key for a meaning."""
        return f"{meaning.meaning_type.value}:{meaning.referent}"

    def add_association(
        self,
        symbol: Symbol,
        meaning: SymbolMeaning,
        initial_strength: float = 0.5,
        tick: int = 0,
    ) -> SymbolAssociation:
        """Add or update a symbol-meaning association.

        If the association already exists, reinforces it instead.

        Args:
            symbol: The symbol
            meaning: The meaning
            initial_strength: Starting strength for new associations
            tick: Current tick

        Returns:
            The new or existing SymbolAssociation
        """
        # Check for existing association
        existing = self._find_association(symbol.form, meaning)
        if existing:
            existing.reinforce(tick=tick)
            return existing

        # Create new association
        assoc = SymbolAssociation(
            symbol=symbol,
            meaning=meaning,
            strength=initial_strength,
            last_used_tick=tick,
        )
        self._associations.append(assoc)

        # Update indexes
        if symbol.form not in self._by_symbol:
            self._by_symbol[symbol.form] = []
        self._by_symbol[symbol.form].append(assoc)

        mkey = self._meaning_key(meaning)
        if mkey not in self._by_meaning:
            self._by_meaning[mkey] = []
        self._by_meaning[mkey].append(assoc)

        # Enforce capacity limit
        self._enforce_capacity()

        return assoc

    def _find_association(
        self, symbol_form: str, meaning: SymbolMeaning
    ) -> SymbolAssociation | None:
        """Find an existing association between symbol and meaning."""
        candidates = self._by_symbol.get(symbol_form, [])
        for assoc in candidates:
            if assoc.meaning.matches(meaning):
                return assoc
        return None

    def produce(self, meaning: SymbolMeaning) -> Symbol | None:
        """Find the best symbol for a meaning (production).

        Returns the symbol with the strongest association to this meaning.

        Args:
            meaning: The meaning to express

        Returns:
            Best symbol, or None if no association exists
        """
        mkey = self._meaning_key(meaning)
        candidates = self._by_meaning.get(mkey, [])
        if not candidates:
            return None

        # Pick strongest association
        best = max(candidates, key=lambda a: a.strength)
        if best.strength < 0.1:
            return None  # too weak to use
        return best.symbol

    def comprehend(self, symbol_form: str) -> SymbolMeaning | None:
        """Interpret a received symbol (comprehension).

        Returns the meaning with the strongest association to this symbol.

        Args:
            symbol_form: The received symbol form

        Returns:
            Best meaning, or None if symbol is unknown
        """
        candidates = self._by_symbol.get(symbol_form, [])
        if not candidates:
            return None

        # Pick strongest association
        best = max(candidates, key=lambda a: a.strength)
        if best.strength < 0.1:
            return None  # too weak to interpret
        return best.meaning

    def all_symbols(self) -> list[str]:
        """Get all known symbol forms."""
        return list(self._by_symbol.keys())

    def vocabulary_size(self) -> int:
        """Number of distinct symbols in lexicon."""
        return len(self._by_symbol)

    def convention_count(self) -> int:
        """Number of strong associations (strength >= 0.5)."""
        return sum(1 for a in self._associations if a.strength >= 0.5)

    def knows_symbol(self, symbol_form: str) -> bool:
        """Check if agent has any association for this symbol."""
        return symbol_form in self._by_symbol and len(self._by_symbol[symbol_form]) > 0

    def get_associations_for_symbol(self, symbol_form: str) -> list[SymbolAssociation]:
        """Get all associations for a symbol form."""
        return self._by_symbol.get(symbol_form, [])

    def decay_all(self, rate: float = 0.005) -> int:
        """Decay all associations. Returns number removed."""
        removed = 0
        dead: list[SymbolAssociation] = []
        for assoc in self._associations:
            assoc.decay(rate)
            if assoc.strength <= 0.01:
                dead.append(assoc)

        for assoc in dead:
            self._remove_association(assoc)
            removed += 1

        return removed

    def _remove_association(self, assoc: SymbolAssociation) -> None:
        """Remove an association from all indexes."""
        if assoc in self._associations:
            self._associations.remove(assoc)

        sym_list = self._by_symbol.get(assoc.symbol.form, [])
        if assoc in sym_list:
            sym_list.remove(assoc)
        if not sym_list and assoc.symbol.form in self._by_symbol:
            del self._by_symbol[assoc.symbol.form]

        mkey = self._meaning_key(assoc.meaning)
        m_list = self._by_meaning.get(mkey, [])
        if assoc in m_list:
            m_list.remove(assoc)
        if not m_list and mkey in self._by_meaning:
            del self._by_meaning[mkey]

    def _enforce_capacity(self) -> None:
        """Remove weakest associations if over capacity."""
        while len(self._associations) > self.MAX_ASSOCIATIONS:
            weakest = min(self._associations, key=lambda a: a.strength)
            self._remove_association(weakest)

    def shared_symbols_with(self, other: Lexicon) -> set[str]:
        """Find symbols both agents know."""
        return set(self._by_symbol.keys()) & set(other._by_symbol.keys())

    def agreement_score(self, other: Lexicon) -> float:
        """Measure how much two lexicons agree on symbol meanings.

        For each shared symbol, check if both agents assign the same meaning.

        Returns:
            Agreement score in [0, 1], or 0.0 if no shared symbols.
        """
        shared = self.shared_symbols_with(other)
        if not shared:
            return 0.0

        agreements = 0
        for sym_form in shared:
            my_meaning = self.comprehend(sym_form)
            their_meaning = other.comprehend(sym_form)
            if my_meaning and their_meaning and my_meaning.matches(their_meaning):
                agreements += 1

        return agreements / len(shared)

    def to_dict(self) -> dict[str, Any]:
        """Serialize lexicon for trajectory recording."""
        result: dict[str, Any] = {
            "vocabulary_size": self.vocabulary_size(),
            "convention_count": self.convention_count(),
            "symbols": {},
        }
        for sym_form, assocs in self._by_symbol.items():
            best = max(assocs, key=lambda a: a.strength) if assocs else None
            if best and best.strength >= 0.2:
                result["symbols"][sym_form] = {
                    "meaning": f"{best.meaning.meaning_type.value}:{best.meaning.referent}",
                    "strength": round(best.strength, 3),
                    "times_used": best.symbol.times_used,
                }
        return result


@dataclass(frozen=True)
class SymbolMessage:
    """A message containing symbols sent between agents.

    Extends the existing message system with symbol-based content.
    Messages have a fixed-type component (for backward compatibility)
    and a symbols component (for emergent language).
    """

    sender_id: str
    receiver_id: str | None  # None = broadcast
    symbols: tuple[str, ...]  # sequence of symbol forms
    context: str  # situational context (e.g., "low_hunger")
    tick: int
    energy_cost: float = 0.5

    @property
    def symbol_count(self) -> int:
        """Number of symbols in message."""
        return len(self.symbols)


@dataclass
class CommunicationOutcome:
    """Result of a symbol-based communication attempt.

    Records whether the receiver correctly interpreted the sender's
    intended meaning, for reinforcement learning.
    """

    message: SymbolMessage
    intended_meanings: list[SymbolMeaning]
    interpreted_meanings: list[SymbolMeaning | None]
    success: bool  # did receiver understand the intended meaning?
    tick: int

    @property
    def interpretation_accuracy(self) -> float:
        """Fraction of symbols correctly interpreted."""
        if not self.intended_meanings:
            return 0.0
        correct = sum(
            1
            for intended, interpreted in zip(
                self.intended_meanings, self.interpreted_meanings, strict=False
            )
            if interpreted and intended.matches(interpreted)
        )
        return correct / len(self.intended_meanings)


@dataclass
class SharedConvention:
    """A symbol-meaning pair that multiple agents agree on.

    Tracked at the population level. A convention is "shared" when
    N+ agents have the same symbol→meaning mapping with strength >= threshold.
    """

    symbol_form: str
    meaning: SymbolMeaning
    adopters: set[str] = field(default_factory=set)  # agent IDs
    first_observed_tick: int = 0
    stability_score: float = 0.0  # how stable this convention has been

    @property
    def adoption_count(self) -> int:
        """Number of agents sharing this convention."""
        return len(self.adopters)

    def is_established(self, min_adopters: int = 3) -> bool:
        """Check if convention has enough adopters to be considered established."""
        return len(self.adopters) >= min_adopters
