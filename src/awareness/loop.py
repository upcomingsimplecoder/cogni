"""Awareness loop orchestrator: the SRIE pipeline for agent decision-making."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.awareness.types import Expression, Intention, Reflection, Sensation

if TYPE_CHECKING:
    from src.awareness.protocols import DeliberationStrategy, EvaluationStrategy, PerceptionStrategy
    from src.awareness.reflection import ReflectionModule
    from src.awareness.sensation import SensationModule
    from src.cognition.strategies.base import DecisionStrategy
    from src.simulation.entities import Agent


class AwarenessLoop:
    """Orchestrates the Sensation-Reflection-Intention-Expression pipeline.

    Wraps a DecisionStrategy with shared perception and reflection infrastructure.
    The strategy only needs to implement intention formation + expression.
    Sensation and Reflection are handled here (shared across all strategies).

    Now supports pluggable perception, evaluation, and deliberation strategies.
    """

    def __init__(
        self,
        agent: Agent,
        strategy: DecisionStrategy,
        sensation_module: SensationModule | None = None,
        reflection_module: ReflectionModule | None = None,
        perception: PerceptionStrategy | None = None,
        evaluation: EvaluationStrategy | None = None,
        deliberation: DeliberationStrategy | None = None,
    ):
        """Initialize the awareness loop.

        Args:
            agent: The agent this loop belongs to
            strategy: Decision strategy (forms intentions + expressions)
            sensation_module: DEPRECATED — use perception instead
            reflection_module: DEPRECATED — use evaluation instead
            perception: Perception strategy (defaults to SensationModule)
            evaluation: Evaluation strategy (defaults to ReflectionModule)
            deliberation: Optional deliberation strategy (System 2 thinking)
        """
        from src.awareness.reflection import ReflectionModule as RM
        from src.awareness.sensation import SensationModule as SM

        self.agent = agent
        self.strategy = strategy

        # Support both old and new API for backward compatibility
        if perception is not None:
            self._perception = perception
        elif sensation_module is not None:
            self._perception = sensation_module
        else:
            self._perception = SM()

        if evaluation is not None:
            self._evaluation = evaluation
        elif reflection_module is not None:
            self._evaluation = reflection_module
        else:
            self._evaluation = RM()

        self._deliberation = deliberation

        # Keep old names as aliases for backward compatibility
        self._sensation = self._perception
        self._reflection = self._evaluation

        self._last_sensation: Sensation | None = None
        self._last_reflection: Reflection | None = None
        self._last_intention: Intention | None = None

    def tick(self, engine: Any) -> Expression:
        """Run one full SRIE cycle. Returns the agent's action + optional message.

        1. Sense: Build Sensation from world state
        2. Reflect: Evaluate recent history
        3. Deliberate (optional): System 2 thinking if needed
        4. Intend: Strategy forms goals
        5. Express: Strategy produces action + message
        """
        # 1. Sensation
        sensation: Sensation = self._perception.perceive(self.agent, engine)
        self._last_sensation = sensation

        # 2. Reflection
        reflection: Reflection = self._evaluation.evaluate(self.agent, engine, sensation)

        # 3. Deliberation (optional System 2)
        if self._deliberation is not None and self._deliberation.should_escalate(
            sensation, reflection
        ):
            reflection = self._deliberation.deliberate(self.agent, sensation, reflection)

        self._last_reflection = reflection

        # 4. Intention (strategy-specific)
        intention: Intention = self.strategy.form_intention(sensation, reflection)
        self._last_intention = intention

        # 5. Expression (strategy-specific)
        expression = self.strategy.express(sensation, reflection, intention)

        # Annotate expression with internal monologue for visualization
        expression.internal_monologue = (
            f"Goal: {intention.primary_goal} "
            f"(conf: {intention.confidence:.1f}) | "
            f"Threat: {reflection.threat_level:.1f} | "
            f"Opp: {reflection.opportunity_score:.1f}"
        )

        # Fill in sender_id on message if present
        if (
            expression.message is not None
            and hasattr(expression.message, "sender_id")
            and expression.message.sender_id is None
        ):
            from src.communication.protocol import create_message

            old = expression.message
            expression.message = create_message(
                tick=old.tick,
                sender_id=self.agent.agent_id,
                receiver_id=old.receiver_id,
                message_type=old.message_type,
                content=old.content,
                payload=dict(old.payload) if old.payload else {},
                energy_cost=old.energy_cost,
            )

        return expression

    @property
    def last_sensation(self):
        """Most recent sensation (for debugging/visualization)."""
        return self._last_sensation

    @property
    def last_reflection(self):
        """Most recent reflection (for debugging/visualization)."""
        return self._last_reflection

    @property
    def last_intention(self):
        """Most recent intention (for debugging/visualization)."""
        return self._last_intention

    def srie_cache_to_dict(self) -> dict | None:
        """Serialize the SRIE cache to a dict for checkpoint storage.

        Returns None if all cached values are None. Otherwise returns a dict with:
        - last_sensation: summarized (no full visible_tiles/agents)
        - last_reflection: fully serialized
        - last_intention: fully serialized

        Note: Sensation is summarized to avoid bloat — visible_tiles/agents are
        replaced with counts. This is sufficient since cache is overwritten on next tick.
        """
        if (
            self._last_sensation is None
            and self._last_reflection is None
            and self._last_intention is None
        ):
            return None

        result: dict[str, Any] = {}

        # Serialize Sensation as SUMMARY
        if self._last_sensation is not None:
            sensation = self._last_sensation
            result["last_sensation"] = {
                "tick": sensation.tick,
                "own_needs": sensation.own_needs,
                "own_position": list(sensation.own_position),
                "own_inventory": sensation.own_inventory,
                "visible_agent_count": len(sensation.visible_agents),
                "visible_resource_count": sum(1 for t in sensation.visible_tiles if t.resources),
                "total_resources": sum(
                    qty for t in sensation.visible_tiles for _, qty in t.resources
                ),
                "incoming_message_count": len(sensation.incoming_messages),
                "time_of_day": sensation.time_of_day,
                "own_traits": sensation.own_traits,
            }
        else:
            result["last_sensation"] = None

        # Serialize Reflection fully
        if self._last_reflection is not None:
            reflection = self._last_reflection
            result["last_reflection"] = {
                "last_action_succeeded": reflection.last_action_succeeded,
                "need_trends": dict(reflection.need_trends),
                "threat_level": reflection.threat_level,
                "opportunity_score": reflection.opportunity_score,
            }
        else:
            result["last_reflection"] = None

        # Serialize Intention fully
        if self._last_intention is not None:
            intention = self._last_intention
            result["last_intention"] = {
                "primary_goal": intention.primary_goal,
                "target_position": (
                    list(intention.target_position) if intention.target_position else None
                ),
                "target_agent_id": (
                    str(intention.target_agent_id) if intention.target_agent_id else None
                ),
                "planned_actions": intention.planned_actions,
                "confidence": intention.confidence,
            }
        else:
            result["last_intention"] = None

        return result

    def srie_cache_from_dict(self, data: dict) -> None:
        """Restore the SRIE cache from serialized checkpoint data.

        Args:
            data: Dict from srie_cache_to_dict()

        Note: Sensation restoration is minimal (empty visible_tiles/agents) since
        we only have summary data. This is intentional — the cache will be
        overwritten on the next tick anyway.
        """
        from src.awareness.types import Intention, Reflection, Sensation

        # Restore Sensation (minimal reconstruction from summary)
        if data.get("last_sensation") is not None:
            s = data["last_sensation"]
            sensation_obj: Sensation = Sensation(
                tick=s["tick"],
                own_needs=s["own_needs"],
                own_position=tuple(s["own_position"]),
                own_inventory=s["own_inventory"],
                visible_tiles=[],  # Summary doesn't preserve this
                visible_agents=[],  # Summary doesn't preserve this
                incoming_messages=[],  # Summary doesn't preserve this
                time_of_day=s["time_of_day"],
                own_traits=s["own_traits"],
            )
            self._last_sensation = sensation_obj
        else:
            self._last_sensation = None

        # Restore Reflection (minimal reconstruction)
        if data.get("last_reflection") is not None:
            r = data["last_reflection"]
            reflection_obj: Reflection = Reflection(
                last_action_succeeded=r["last_action_succeeded"],
                need_trends=dict(r["need_trends"]),
                recent_interaction_outcomes=[],  # Not serialized in summary
                threat_level=r["threat_level"],
                opportunity_score=r["opportunity_score"],
            )
            self._last_reflection = reflection_obj
        else:
            self._last_reflection = None

        # Restore Intention (full reconstruction)
        if data.get("last_intention") is not None:
            i = data["last_intention"]
            intention_obj: Intention = Intention(
                primary_goal=i["primary_goal"],
                target_position=tuple(i["target_position"]) if i["target_position"] else None,
                target_agent_id=i["target_agent_id"],  # Stored as string
                planned_actions=i["planned_actions"],
                confidence=i["confidence"],
            )
            self._last_intention = intention_obj
        else:
            self._last_intention = None
