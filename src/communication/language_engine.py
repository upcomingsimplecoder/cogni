"""Language engine: orchestrates emergent language dynamics in agent societies.

Phase 4 of the AUTOCOG product vision. Manages the full language cycle each tick:
1. Innovation: Agents create new symbols for novel contexts
2. Communication: Agents exchange symbol-based messages
3. Grounding: Referential games create/reinforce symbol-meaning associations
4. ToM Pragmatics: Agents adapt utterances to listener's lexicon
5. Evolution: Conventions spread, drift, and die based on usage
6. Metrics: Track vocabulary growth, convention stability, dialect formation

This is the deep tech moat — first emergent language system combining
LLM cognition + cultural transmission dynamics + ToM-grounded pragmatics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.communication.language import (
    CommunicationOutcome,
    Lexicon,
    MeaningType,
    SharedConvention,
    Symbol,
    SymbolMeaning,
    SymbolMessage,
)

if TYPE_CHECKING:
    from src.simulation.engine import AgentTickRecord, SimulationEngine


# --- Context-to-meaning mapping ---
# Maps context tags and action types to SymbolMeaning instances.
# This is how the grounding system connects simulation state to language.

CONTEXT_MEANINGS: dict[str, SymbolMeaning] = {
    "low_hunger": SymbolMeaning(MeaningType.CONTEXT, "hungry"),
    "low_thirst": SymbolMeaning(MeaningType.CONTEXT, "thirsty"),
    "low_energy": SymbolMeaning(MeaningType.CONTEXT, "tired"),
    "danger": SymbolMeaning(MeaningType.CONTEXT, "danger"),
    "crowded": SymbolMeaning(MeaningType.CONTEXT, "crowded"),
    "alone": SymbolMeaning(MeaningType.CONTEXT, "alone"),
    "resource_nearby": SymbolMeaning(MeaningType.CONTEXT, "food_nearby"),
}

ACTION_MEANINGS: dict[str, SymbolMeaning] = {
    "gather": SymbolMeaning(MeaningType.ACTION, "gather"),
    "move": SymbolMeaning(MeaningType.ACTION, "move"),
    "rest": SymbolMeaning(MeaningType.ACTION, "rest"),
    "eat": SymbolMeaning(MeaningType.ACTION, "eat"),
    "drink": SymbolMeaning(MeaningType.ACTION, "drink"),
    "attack": SymbolMeaning(MeaningType.ACTION, "attack"),
    "give": SymbolMeaning(MeaningType.SOCIAL, "help"),
    "share": SymbolMeaning(MeaningType.SOCIAL, "share"),
    "wait": SymbolMeaning(MeaningType.ACTION, "wait"),
}

DIRECTION_MEANINGS: dict[str, SymbolMeaning] = {
    "north": SymbolMeaning(MeaningType.LOCATION, "north"),
    "south": SymbolMeaning(MeaningType.LOCATION, "south"),
    "east": SymbolMeaning(MeaningType.LOCATION, "east"),
    "west": SymbolMeaning(MeaningType.LOCATION, "west"),
}


@dataclass
class LanguageSnapshot:
    """Snapshot of the language state at a point in time."""

    tick: int
    total_vocabulary: int  # sum of all agents' vocabulary sizes
    unique_symbols: int  # total distinct symbols across all agents
    convention_count: int  # number of established shared conventions
    avg_agreement: float  # average pairwise lexicon agreement
    communication_success_rate: float  # recent communication success
    dialect_count: int  # number of dialect groups detected
    innovations_this_tick: int
    adoptions_this_tick: int
    deaths_this_tick: int


class LanguageEngine:
    """Orchestrates emergent language dynamics across the agent population.

    Called once per tick (Phase 8.9 in engine.step_all) after metacognition.

    Architecture follows the same pattern as CulturalTransmissionEngine
    and MetacognitiveEngine: per-agent state, tick-based processing,
    snapshot generation for visualization and analysis.
    """

    def __init__(
        self,
        innovation_rate: float = 0.05,
        bandwidth_limit: int = 3,
        symbol_energy_cost: float = 0.3,
        grounding_reinforcement: float = 0.15,
        grounding_weakening: float = 0.08,
        decay_rate: float = 0.003,
        convention_threshold: float = 0.5,
        convention_min_adopters: int = 3,
        communication_range: int = 8,
    ):
        """Initialize the language engine.

        Args:
            innovation_rate: Probability per tick of creating a new symbol
            bandwidth_limit: Max symbols per message (drives compression)
            symbol_energy_cost: Energy cost per symbol in message
            grounding_reinforcement: Strength increase on successful communication
            grounding_weakening: Strength decrease on failed communication
            decay_rate: Per-tick decay of unused symbol associations
            convention_threshold: Min association strength to count as convention
            convention_min_adopters: Min agents sharing convention to be "established"
            communication_range: Max manhattan distance for symbol messages
        """
        self.innovation_rate = innovation_rate
        self.bandwidth_limit = bandwidth_limit
        self.symbol_energy_cost = symbol_energy_cost
        self.grounding_reinforcement = grounding_reinforcement
        self.grounding_weakening = grounding_weakening
        self.decay_rate = decay_rate
        self.convention_threshold = convention_threshold
        self.convention_min_adopters = convention_min_adopters
        self.communication_range = communication_range

        # Per-agent state
        self._lexicons: dict[str, Lexicon] = {}
        self._communication_history: dict[str, list[CommunicationOutcome]] = {}

        # Global state
        self._conventions: dict[str, SharedConvention] = {}  # key: "symbol:meaning_key"
        self._all_symbols: dict[str, Symbol] = {}  # form -> Symbol
        self._messages_this_tick: list[SymbolMessage] = []
        self._outcomes_this_tick: list[CommunicationOutcome] = []
        self._snapshots: list[LanguageSnapshot] = []

        # Per-tick counters
        self._innovations_this_tick = 0
        self._adoptions_this_tick = 0
        self._deaths_this_tick = 0

    def register_agent(self, agent_id: str) -> None:
        """Register a new agent with the language system.

        Creates an empty lexicon and communication history.

        Args:
            agent_id: String agent identifier
        """
        if agent_id not in self._lexicons:
            self._lexicons[agent_id] = Lexicon(agent_id)
            self._communication_history[agent_id] = []

    def unregister_agent(self, agent_id: str) -> None:
        """Mark agent as inactive. Keep data for analysis."""
        pass  # Intentionally keep data for linguistic evolution analysis

    def get_lexicon(self, agent_id: str) -> Lexicon | None:
        """Get an agent's lexicon."""
        return self._lexicons.get(agent_id)

    def tick(
        self,
        engine: SimulationEngine,
        tick: int,
        agent_records: list[AgentTickRecord],
    ) -> LanguageSnapshot:
        """Process language dynamics for this tick.

        Steps:
        1. Reset per-tick counters
        2. Innovation: agents create symbols for novel contexts
        3. Communication: agents send symbol messages to nearby agents
        4. Grounding: evaluate communication outcomes, reinforce/weaken
        5. Evolution: decay unused symbols, update conventions
        6. Build snapshot

        Args:
            engine: SimulationEngine instance
            tick: Current tick number
            agent_records: List of AgentTickRecord from this tick

        Returns:
            LanguageSnapshot for this tick
        """
        self._innovations_this_tick = 0
        self._adoptions_this_tick = 0
        self._deaths_this_tick = 0
        self._messages_this_tick = []
        self._outcomes_this_tick = []

        living = engine.registry.living_agents()

        # Step 1: Innovation — agents create symbols for contexts they lack words for
        for agent in living:
            aid = str(agent.agent_id)
            if aid not in self._lexicons:
                continue
            self._try_innovation(aid, tick, engine, agent)

        # Step 2: Communication — agents send symbol messages
        for record in agent_records:
            aid = str(record.agent_id)
            if aid not in self._lexicons:
                continue
            sender_agent = engine.registry.get(record.agent_id)
            if not sender_agent or not sender_agent.alive:
                continue
            self._try_communication(aid, tick, engine, sender_agent, record)

        # Step 3: Grounding — evaluate outcomes and reinforce/weaken
        for outcome in self._outcomes_this_tick:
            self._process_grounding(outcome, tick)

        # Step 4: Evolution — decay and convention tracking
        for aid in list(self._lexicons.keys()):
            living_agent: Any = None
            for a in living:
                if str(a.agent_id) == aid:
                    living_agent = a
                    break
            if living_agent and living_agent.alive:
                deaths = self._lexicons[aid].decay_all(self.decay_rate)
                self._deaths_this_tick += deaths

        self._update_conventions(tick)

        # Step 5: Build snapshot
        snapshot = self._build_snapshot(tick, engine)
        self._snapshots.append(snapshot)
        return snapshot

    def _try_innovation(
        self,
        agent_id: str,
        tick: int,
        engine: Any,
        agent: Any,
    ) -> None:
        """Probabilistically create new symbols for contexts the agent lacks words for.

        Innovation happens when:
        1. Agent encounters a context/action it has no symbol for
        2. Random roll passes the innovation rate
        """
        if random.random() > self.innovation_rate:
            return

        lexicon = self._lexicons[agent_id]

        # Get current context from awareness loop
        loop = engine.registry.get_awareness_loop(agent.agent_id)
        if not loop or not loop.last_sensation:
            return

        from src.awareness.context import ContextTag

        context_tag = ContextTag.extract_primary(loop.last_sensation)

        # Check if there's a meaning for this context that lacks a symbol
        meaning = CONTEXT_MEANINGS.get(context_tag)
        if not meaning:
            return

        # Only innovate if we don't already have a symbol for this meaning
        existing = lexicon.produce(meaning)
        if existing:
            return

        # Create a new symbol
        symbol = Symbol.generate_novel(tick, agent_id)

        # Avoid collision with existing symbols
        attempts = 0
        while symbol.form in self._all_symbols and attempts < 5:
            symbol = Symbol.generate_novel(tick, agent_id)
            attempts += 1

        # Register globally
        self._all_symbols[symbol.form] = symbol

        # Add to agent's lexicon with moderate initial strength
        lexicon.add_association(symbol, meaning, initial_strength=0.6, tick=tick)
        self._innovations_this_tick += 1

    def _try_communication(
        self,
        agent_id: str,
        tick: int,
        engine: Any,
        agent: Any,
        record: Any,
    ) -> None:
        """Attempt symbol-based communication with nearby agents.

        Agents communicate when they have something to say and a nearby listener.
        Messages are adapted to the listener's lexicon via ToM (if available).
        """
        lexicon = self._lexicons[agent_id]
        if lexicon.vocabulary_size() == 0:
            return

        # Get awareness loop for context
        loop = engine.registry.get_awareness_loop(agent.agent_id)
        if not loop or not loop.last_sensation:
            return

        # Only communicate sometimes (not every tick)
        if random.random() > 0.15:
            return

        # Find nearby agents
        nearby = []
        for other in engine.registry.living_agents():
            if str(other.agent_id) == agent_id:
                continue
            dist = abs(agent.x - other.x) + abs(agent.y - other.y)
            if dist <= self.communication_range:
                nearby.append(other)

        if not nearby:
            return

        # Pick a listener (nearest)
        listener = min(nearby, key=lambda a: abs(a.x - agent.x) + abs(a.y - agent.y))
        listener_id = str(listener.agent_id)

        # Determine what to communicate (context + action intent)
        from src.awareness.context import ContextTag

        context_tag = ContextTag.extract_primary(loop.last_sensation)
        meanings_to_express: list[SymbolMeaning] = []

        # Add context meaning
        ctx_meaning = CONTEXT_MEANINGS.get(context_tag)
        if ctx_meaning:
            meanings_to_express.append(ctx_meaning)

        # Add action meaning
        if record.action:
            act_meaning = ACTION_MEANINGS.get(record.action.type.value)
            if act_meaning:
                meanings_to_express.append(act_meaning)

        if not meanings_to_express:
            return

        # Apply bandwidth limit
        meanings_to_express = meanings_to_express[: self.bandwidth_limit]

        # ToM-grounded pragmatics: adapt to listener's known symbols
        symbols_to_send: list[str] = []
        intended_meanings: list[SymbolMeaning] = []

        for meaning in meanings_to_express:
            symbol = self._select_symbol_for_listener(agent_id, listener_id, meaning, tick)
            if symbol:
                symbols_to_send.append(symbol.form)
                intended_meanings.append(meaning)
                symbol.times_used += 1

        if not symbols_to_send:
            return

        # Create message
        msg = SymbolMessage(
            sender_id=agent_id,
            receiver_id=listener_id,
            symbols=tuple(symbols_to_send),
            context=context_tag,
            tick=tick,
            energy_cost=self.symbol_energy_cost * len(symbols_to_send),
        )
        self._messages_this_tick.append(msg)

        # Route to engine message bus for cross-system visibility
        from src.communication.protocol import MessageType, create_message

        bus_msg = create_message(
            tick=tick,
            sender_id=agent.agent_id,
            receiver_id=listener.agent_id,
            message_type=MessageType.INFORM,
            content=f"[lang] {' '.join(symbols_to_send)}",
            payload={"symbols": list(symbols_to_send), "context": context_tag},
        )
        engine.message_bus.send(bus_msg)

        # Listener interprets
        listener_lexicon = self._lexicons.get(listener_id)
        interpreted: list[SymbolMeaning | None] = []
        if listener_lexicon:
            for sym_form in symbols_to_send:
                interpreted.append(listener_lexicon.comprehend(sym_form))
        else:
            interpreted = [None] * len(symbols_to_send)

        # Evaluate success
        success = False
        if intended_meanings and interpreted:
            # Success if at least one symbol was correctly interpreted
            for intended, interp in zip(intended_meanings, interpreted, strict=False):
                if interp and intended.matches(interp):
                    success = True
                    break

        outcome = CommunicationOutcome(
            message=msg,
            intended_meanings=intended_meanings,
            interpreted_meanings=interpreted,
            success=success,
            tick=tick,
        )
        self._outcomes_this_tick.append(outcome)

        # Store in history
        if agent_id in self._communication_history:
            self._communication_history[agent_id].append(outcome)
            # Keep bounded
            if len(self._communication_history[agent_id]) > 100:
                self._communication_history[agent_id] = self._communication_history[agent_id][-100:]

    def _select_symbol_for_listener(
        self,
        sender_id: str,
        listener_id: str,
        meaning: SymbolMeaning,
        tick: int,
    ) -> Symbol | None:
        """Select the best symbol for a meaning, considering listener's lexicon.

        Gricean pragmatics:
        - Prefer symbols the listener knows (maxim of manner)
        - Prefer established conventions (maxim of quantity)
        - Fall back to sender's strongest association

        Args:
            sender_id: Sending agent
            listener_id: Receiving agent
            meaning: Meaning to express
            tick: Current tick

        Returns:
            Best symbol to use, or None
        """
        sender_lexicon = self._lexicons.get(sender_id)
        listener_lexicon = self._lexicons.get(listener_id)

        if not sender_lexicon:
            return None

        # Get sender's symbol for this meaning
        sender_symbol = sender_lexicon.produce(meaning)
        if not sender_symbol:
            return None

        # If no listener lexicon, just use sender's best
        if not listener_lexicon:
            return sender_symbol

        # ToM check: does listener know this symbol?
        if listener_lexicon.knows_symbol(sender_symbol.form):
            return sender_symbol

        # Try to find a symbol both know that maps to this meaning
        shared = sender_lexicon.shared_symbols_with(listener_lexicon)
        for sym_form in shared:
            sender_meaning = sender_lexicon.comprehend(sym_form)
            if sender_meaning and sender_meaning.matches(meaning):
                # Both know this symbol and sender maps it to the right meaning
                return (
                    sender_lexicon._by_symbol[sym_form][0].symbol
                    if sym_form in sender_lexicon._by_symbol
                    else None
                )

        # Fall back to sender's best (listener will learn)
        return sender_symbol

    def _process_grounding(
        self,
        outcome: CommunicationOutcome,
        tick: int,
    ) -> None:
        """Process communication outcome: reinforce or weaken associations.

        Successful communication strengthens associations in both lexicons.
        Failed communication weakens sender's association and creates
        a new association in listener's lexicon (learning).

        Args:
            outcome: The communication outcome to process
            tick: Current tick
        """
        sender_lexicon = self._lexicons.get(outcome.message.sender_id)
        listener_lexicon = (
            self._lexicons.get(outcome.message.receiver_id) if outcome.message.receiver_id else None
        )

        if not sender_lexicon:
            return

        for i, (sym_form, intended) in enumerate(
            zip(outcome.message.symbols, outcome.intended_meanings, strict=False)
        ):
            interpreted = (
                outcome.interpreted_meanings[i] if i < len(outcome.interpreted_meanings) else None
            )

            if interpreted and intended.matches(interpreted):
                # Success — reinforce in both lexicons
                sender_assoc = sender_lexicon._find_association(sym_form, intended)
                if sender_assoc:
                    sender_assoc.reinforce(self.grounding_reinforcement, tick)

                if listener_lexicon:
                    listener_assoc = listener_lexicon._find_association(sym_form, intended)
                    if listener_assoc:
                        listener_assoc.reinforce(self.grounding_reinforcement, tick)

                # Track on symbol
                symbol = self._all_symbols.get(sym_form)
                if symbol:
                    symbol.times_understood += 1

            else:
                # Failure — weaken sender's association slightly
                sender_assoc = sender_lexicon._find_association(sym_form, intended)
                if sender_assoc:
                    sender_assoc.weaken(self.grounding_weakening)

                # Listener learns: create new association from context
                if listener_lexicon and outcome.message.receiver_id:
                    symbol = self._all_symbols.get(sym_form)
                    if symbol:
                        # Listener infers meaning from context
                        listener_lexicon.add_association(
                            symbol, intended, initial_strength=0.3, tick=tick
                        )
                        self._adoptions_this_tick += 1

    def _update_conventions(self, tick: int) -> None:
        """Detect and track shared conventions across the population.

        A convention exists when multiple agents have the same
        symbol→meaning mapping above the threshold strength.
        """
        # Build convention candidates from all lexicons
        convention_candidates: dict[str, dict[str, set[str]]] = {}
        # Key: symbol_form, Value: {meaning_key: set of agent_ids}

        for agent_id, lexicon in self._lexicons.items():
            for sym_form, assocs in lexicon._by_symbol.items():
                for assoc in assocs:
                    if assoc.strength < self.convention_threshold:
                        continue
                    mkey = lexicon._meaning_key(assoc.meaning)
                    if sym_form not in convention_candidates:
                        convention_candidates[sym_form] = {}
                    if mkey not in convention_candidates[sym_form]:
                        convention_candidates[sym_form][mkey] = set()
                    convention_candidates[sym_form][mkey].add(agent_id)

        # Update conventions
        active_keys: set[str] = set()
        for sym_form, meaning_map in convention_candidates.items():
            for mkey, adopters in meaning_map.items():
                conv_key = f"{sym_form}:{mkey}"
                active_keys.add(conv_key)

                if conv_key in self._conventions:
                    # Update existing convention
                    conv = self._conventions[conv_key]
                    conv.adopters = adopters
                else:
                    # Parse meaning from key
                    parts = mkey.split(":", 1)
                    if len(parts) == 2:
                        try:
                            meaning = SymbolMeaning(
                                meaning_type=MeaningType(parts[0]),
                                referent=parts[1],
                            )
                        except ValueError:
                            continue
                    else:
                        continue

                    self._conventions[conv_key] = SharedConvention(
                        symbol_form=sym_form,
                        meaning=meaning,
                        adopters=adopters,
                        first_observed_tick=tick,
                    )

        # Remove dead conventions (no longer shared by anyone)
        dead_keys = set(self._conventions.keys()) - active_keys
        for key in dead_keys:
            del self._conventions[key]

    def _build_snapshot(self, tick: int, engine: Any) -> LanguageSnapshot:
        """Build a snapshot of the language state for analysis."""
        # Vocabulary stats
        total_vocab = sum(lex.vocabulary_size() for lex in self._lexicons.values())
        all_sym_forms: set[str] = set()
        for lex in self._lexicons.values():
            all_sym_forms.update(lex.all_symbols())
        unique_symbols = len(all_sym_forms)

        # Convention count (established only)
        convention_count = sum(
            1 for c in self._conventions.values() if c.is_established(self.convention_min_adopters)
        )

        # Average pairwise agreement (sample if too many agents)
        agent_ids = list(self._lexicons.keys())
        avg_agreement = 0.0
        if len(agent_ids) >= 2:
            pairs = 0
            total_agreement = 0.0
            # Sample up to 20 pairs
            for i in range(min(len(agent_ids), 10)):
                for j in range(i + 1, min(len(agent_ids), 10)):
                    a = self._lexicons[agent_ids[i]]
                    b = self._lexicons[agent_ids[j]]
                    total_agreement += a.agreement_score(b)
                    pairs += 1
            if pairs > 0:
                avg_agreement = total_agreement / pairs

        # Communication success rate (recent)
        recent_outcomes = self._outcomes_this_tick
        success_rate = 0.0
        if recent_outcomes:
            success_rate = sum(1 for o in recent_outcomes if o.success) / len(recent_outcomes)

        # Dialect detection (simple clustering by shared vocabulary)
        dialect_count = self._detect_dialect_count()

        return LanguageSnapshot(
            tick=tick,
            total_vocabulary=total_vocab,
            unique_symbols=unique_symbols,
            convention_count=convention_count,
            avg_agreement=round(avg_agreement, 3),
            communication_success_rate=round(success_rate, 3),
            dialect_count=dialect_count,
            innovations_this_tick=self._innovations_this_tick,
            adoptions_this_tick=self._adoptions_this_tick,
            deaths_this_tick=self._deaths_this_tick,
        )

    def _detect_dialect_count(self) -> int:
        """Simple dialect detection via vocabulary overlap clustering.

        Returns approximate number of dialect groups.
        """
        agent_ids = list(self._lexicons.keys())
        if len(agent_ids) < 2:
            return 1 if agent_ids else 0

        # Simple connected-component approach:
        # Two agents are in the same dialect if they share 50%+ vocabulary
        groups: list[set[str]] = []
        assigned: set[str] = set()

        for aid in agent_ids:
            if aid in assigned:
                continue
            group = {aid}
            assigned.add(aid)
            # BFS for connected agents
            queue = [aid]
            while queue:
                current = queue.pop(0)
                current_lex = self._lexicons[current]
                for other_id in agent_ids:
                    if other_id in assigned:
                        continue
                    other_lex = self._lexicons[other_id]
                    agreement = current_lex.agreement_score(other_lex)
                    if agreement >= 0.5:
                        group.add(other_id)
                        assigned.add(other_id)
                        queue.append(other_id)
            groups.append(group)

        return len(groups)

    # =========================================================================
    # Accessors (for visualization, metrics, testing)
    # =========================================================================

    def get_established_conventions(self) -> list[SharedConvention]:
        """Get all conventions with enough adopters."""
        return [
            c for c in self._conventions.values() if c.is_established(self.convention_min_adopters)
        ]

    def get_all_conventions(self) -> list[SharedConvention]:
        """Get all tracked conventions."""
        return list(self._conventions.values())

    def get_communication_success_rate(self, agent_id: str) -> float:
        """Get an agent's recent communication success rate."""
        history = self._communication_history.get(agent_id, [])
        if not history:
            return 0.0
        recent = history[-20:]
        return sum(1 for o in recent if o.success) / len(recent)

    def get_language_stats(self) -> dict[str, Any]:
        """Aggregate language statistics for metrics and visualization."""
        total_vocab = sum(lex.vocabulary_size() for lex in self._lexicons.values())
        all_sym_forms: set[str] = set()
        for lex in self._lexicons.values():
            all_sym_forms.update(lex.all_symbols())

        established = self.get_established_conventions()

        return {
            "total_vocabulary": total_vocab,
            "unique_symbols": len(all_sym_forms),
            "agents_tracked": len(self._lexicons),
            "established_conventions": len(established),
            "total_conventions_tracked": len(self._conventions),
            "messages_this_tick": len(self._messages_this_tick),
            "outcomes_this_tick": len(self._outcomes_this_tick),
            "innovations_this_tick": self._innovations_this_tick,
            "convention_list": [
                {
                    "symbol": c.symbol_form,
                    "meaning": f"{c.meaning.meaning_type.value}:{c.meaning.referent}",
                    "adopters": c.adoption_count,
                    "established": c.is_established(self.convention_min_adopters),
                }
                for c in sorted(
                    self._conventions.values(),
                    key=lambda c: c.adoption_count,
                    reverse=True,
                )[:10]  # top 10
            ],
        }

    def pairwise_similarity(self, agent_ids: list[str]) -> dict[str, Any]:
        """Compute Jaccard similarity between each pair of agent lexicons.

        Args:
            agent_ids: List of agent IDs to compute similarity for

        Returns:
            Dict with 'agent_ids' and 'matrix' where matrix[i][j] is the
            Jaccard similarity between agent i and agent j's lexicon
        """
        n = len(agent_ids)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue

                lex_i = self._lexicons.get(agent_ids[i])
                lex_j = self._lexicons.get(agent_ids[j])

                if not lex_i or not lex_j:
                    matrix[i][j] = 0.0
                    continue

                # Jaccard similarity: |intersection| / |union|
                symbols_i = set(lex_i.all_symbols())
                symbols_j = set(lex_j.all_symbols())

                if not symbols_i and not symbols_j:
                    matrix[i][j] = 1.0  # Both empty
                elif not symbols_i or not symbols_j:
                    matrix[i][j] = 0.0  # One empty
                else:
                    intersection = len(symbols_i & symbols_j)
                    union = len(symbols_i | symbols_j)
                    matrix[i][j] = round(intersection / union, 3) if union > 0 else 0.0

        return {
            "agent_ids": agent_ids,
            "matrix": matrix,
        }

    @property
    def messages_this_tick(self) -> list[SymbolMessage]:
        """Messages sent this tick."""
        return self._messages_this_tick

    @property
    def outcomes_this_tick(self) -> list[CommunicationOutcome]:
        """Communication outcomes this tick."""
        return self._outcomes_this_tick

    @property
    def snapshots(self) -> list[LanguageSnapshot]:
        """All language snapshots across the simulation."""
        return self._snapshots
