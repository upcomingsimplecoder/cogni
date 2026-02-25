"""Personality-driven decision strategy with trait-biased behavior."""

from __future__ import annotations

import math

from src.awareness.types import (
    AgentSummary,
    Expression,
    Intention,
    Reflection,
    Sensation,
)
from src.communication.protocol import MessageType, create_message
from src.simulation.actions import Action, ActionType, Direction

NEED_TO_RESOURCE = {
    "hunger": "berry_bush",
    "thirst": "water_source",
}

NEED_TO_ITEM = {
    "hunger": "berries",
    "thirst": "water",
}

NEED_TO_ACTION = {
    "hunger": ActionType.EAT,
    "thirst": ActionType.DRINK,
}


class PersonalityStrategy:
    """Trait-biased decision making.

    Same fundamental survival logic as HardcodedStrategy, but every decision
    point is modulated by personality traits:

    - cooperation_tendency → weighs helping others vs self-interest
    - curiosity → weighs explore vs exploit
    - risk_tolerance → threshold at which needs become "urgent"
    - resource_sharing → willingness to GIVE items
    - aggression → tendency to ATTACK vs FLEE vs negotiate
    - sociability → tendency to move toward other agents + send messages
    """

    def form_intention(self, sensation: Sensation, reflection: Reflection) -> Intention:
        """Form intention modulated by personality traits."""
        traits = sensation.own_traits
        needs = sensation.own_needs

        # Risk tolerance shifts urgency thresholds
        risk_tol = traits.get("risk_tolerance", 0.5)
        urgency_threshold = 30 - (risk_tol * 20)  # range: 10-30

        # Priority 0: High threat overrides — flee if danger is assessed as severe
        # This is where cognitive architecture differences manifest:
        # cautious/social architectures inflate threat_level, making agents flee sooner.
        threat = reflection.threat_level
        if threat > 0.6 and risk_tol < 0.7:
            # Non-bold agents flee from high assessed threat
            return Intention(
                primary_goal="flee",
                planned_actions=["flee"],
                confidence=min(0.95, 0.6 + threat * 0.3),
            )

        # Priority 1: Rest if energy critically low (universal survival)
        if needs.get("energy", 0) < max(10, urgency_threshold * 0.5):
            return Intention(
                primary_goal="rest",
                planned_actions=["rest"],
                confidence=0.9,
            )

        # Priority 2: Agent interactions (sociability/aggression-gated)
        sociability = traits.get("sociability", 0.5)
        aggression_val = traits.get("aggression", 0.5)

        visible_agents = sensation.visible_agents

        if visible_agents:
            # Aggressive agents attack if on same tile
            if aggression_val > 0.7 and self._any_agent_on_same_tile(sensation):
                target = self._agent_on_same_tile(sensation)
                if target:
                    return Intention(
                        primary_goal="attack",
                        target_agent_id=target.agent_id,
                        target_position=target.position,
                        planned_actions=["attack_target"],
                        confidence=0.7,
                    )

            # Aggressive agents approach and threaten when they see others
            if aggression_val > 0.5:
                nearest = self._nearest_agent(sensation)
                if nearest:
                    return Intention(
                        primary_goal="approach_threaten",
                        target_agent_id=nearest.agent_id,
                        target_position=nearest.position,
                        planned_actions=["move_toward_agent", "threaten"],
                        confidence=0.6,
                    )

            # Sharing-first for generous agents adjacent to others
            if traits.get("resource_sharing", 0.5) > 0.4 and sensation.own_inventory:
                for agent_sum in visible_agents:
                    dist = abs(agent_sum.position[0] - sensation.own_position[0]) + abs(
                        agent_sum.position[1] - sensation.own_position[1]
                    )
                    sharing = traits.get("resource_sharing", 0.5)
                    if dist <= 1 and (
                        agent_sum.apparent_health in ("injured", "critical") or sharing > 0.6
                    ):
                        return Intention(
                            primary_goal="share_resources",
                            target_agent_id=agent_sum.agent_id,
                            target_position=agent_sum.position,
                            planned_actions=["give_item"],
                            confidence=0.7,
                        )

            # Social agents approach and communicate (only if NOT adjacent)
            if sociability > 0.5:
                nearest = self._nearest_agent(sensation)
                if nearest:
                    dist = abs(nearest.position[0] - sensation.own_position[0]) + abs(
                        nearest.position[1] - sensation.own_position[1]
                    )
                    if dist > 1:
                        # Move toward distant agent
                        return Intention(
                            primary_goal="socialize",
                            target_agent_id=nearest.agent_id,
                            target_position=nearest.position,
                            planned_actions=["move_toward_agent", "communicate"],
                            confidence=0.6,
                        )
                    # Already adjacent — fall through to gather/explore
                    # so we can build inventory for future sharing

        # Priority 3: Address urgent needs (threshold modulated by risk_tolerance)
        survival_needs = {k: v for k, v in needs.items() if k in ("hunger", "thirst", "energy")}
        urgent_needs = {k: v for k, v in survival_needs.items() if v < urgency_threshold}

        if urgent_needs:
            most_urgent = min(urgent_needs, key=lambda k: urgent_needs[k])

            # Try inventory first
            item_needed = NEED_TO_ITEM.get(most_urgent)
            if item_needed and sensation.own_inventory.get(item_needed, 0) > 0:
                return Intention(
                    primary_goal=f"consume_{most_urgent}",
                    planned_actions=["eat_or_drink"],
                    confidence=0.9,
                )

            # Find resource
            resource_kind = NEED_TO_RESOURCE.get(most_urgent)
            if resource_kind:
                target_pos: tuple[int, int] | None = self._find_nearest_resource(
                    sensation, resource_kind
                )
                if target_pos is not None:
                    dist = abs(target_pos[0] - sensation.own_position[0]) + abs(
                        target_pos[1] - sensation.own_position[1]
                    )
                    # Adjacent (dist ≤ 1) counts as "at resource" for gather —
                    # critical for water (can't walk on water tiles)
                    at_resource = dist <= 1
                    return Intention(
                        primary_goal=f"{'gather' if at_resource else 'seek'}_{most_urgent}",
                        target_position=target_pos,
                        planned_actions=["gather" if at_resource else "move_toward_resource"],
                        confidence=0.8 if at_resource else 0.6,
                    )

        # Priority 4: Exploration (curiosity-gated, or when needs are declining)
        curiosity = traits.get("curiosity", 0.5)
        opportunity = reflection.opportunity_score
        needs_declining = (
            any(v == "declining" for v in reflection.need_trends.values())
            if reflection.need_trends
            else False
        )

        # High opportunity score (from architecture evaluation) lowers curiosity bar
        effective_curiosity = curiosity + opportunity * 0.2

        # Explore if curious, OR if needs are declining (must find resources)
        if effective_curiosity > 0.6 or needs_declining:
            return Intention(
                primary_goal="explore",
                planned_actions=["move_explore"],
                confidence=0.5,
            )

        # Priority 5: Gather whatever is around (own tile or adjacent)
        for tile in sensation.visible_tiles:
            tx, ty = tile.x, tile.y
            dist = abs(tx - sensation.own_position[0]) + abs(ty - sensation.own_position[1])
            if dist <= 1 and tile.resources:
                kind, qty = tile.resources[0]
                if qty > 0:
                    return Intention(
                        primary_goal="gather_opportunistic",
                        target_position=(tx, ty),
                        planned_actions=["gather"],
                        confidence=0.5,
                    )

        # Default: explore if energy allows, otherwise rest
        if needs.get("energy", 100) > 30:
            return Intention(
                primary_goal="explore", planned_actions=["move_explore"], confidence=0.3
            )
        if needs.get("energy", 100) < 50:
            return Intention(primary_goal="rest", planned_actions=["rest"], confidence=0.4)
        return Intention(primary_goal="wait", planned_actions=["wait"], confidence=0.2)

    def express(
        self, sensation: Sensation, reflection: Reflection, intention: Intention
    ) -> Expression:
        """Convert intention to concrete action + optional message."""
        traits = sensation.own_traits

        match intention.primary_goal:
            case "rest":
                return Expression(action=Action(type=ActionType.REST))

            case "wait":
                return Expression(action=Action(type=ActionType.WAIT))

            case "flee":
                # Flee away from visible agents (or random direction if none)
                if sensation.visible_agents:
                    nearest = self._nearest_agent(sensation)
                    if nearest:
                        direction = self._direction_away(sensation.own_position, nearest.position)
                        if direction:
                            return Expression(
                                action=Action(type=ActionType.MOVE, direction=direction),
                                internal_monologue="Threat detected — fleeing!",
                            )
                # Fallback: move in exploration pattern
                return self._express_explore(sensation)

            case goal if goal.startswith("consume_"):
                need = goal.removeprefix("consume_")
                action_type = NEED_TO_ACTION.get(need, ActionType.WAIT)
                return Expression(action=Action(type=action_type))

            case goal if goal.startswith("gather"):
                resource_kind = None
                for tile in sensation.visible_tiles:
                    if (tile.x, tile.y) == sensation.own_position and tile.resources:
                        resource_kind = tile.resources[0][0]
                        break
                expr = Expression(action=Action(type=ActionType.GATHER, target=resource_kind))
                # Share resource location with nearby agents
                if (
                    sensation.visible_agents
                    and traits.get("sociability", 0.5) > 0.3
                    and intention.target_position
                ):
                    nearest = self._nearest_agent(sensation)
                    if nearest:
                        pos = intention.target_position
                        msg = create_message(
                            tick=sensation.tick,
                            sender_id=None,  # Filled by awareness loop
                            receiver_id=nearest.agent_id,
                            message_type=MessageType.INFORM,
                            content=f"Resource at ({pos[0]}, {pos[1]})",
                            payload={"resource_pos": list(intention.target_position)},
                        )
                        expr.message = msg
                return expr

            case goal if goal.startswith("seek_"):
                return self._express_move_toward(sensation, intention)

            case "explore":
                return self._express_explore(sensation)

            case "socialize":
                expr = self._express_move_toward(sensation, intention)
                # Add a friendly message if sociable
                if traits.get("sociability", 0.5) > 0.5 and intention.target_agent_id:
                    msg = create_message(
                        tick=sensation.tick,
                        sender_id=None,  # Filled by awareness loop
                        receiver_id=intention.target_agent_id,
                        message_type=MessageType.INFORM,
                        content="Hello! I'm nearby.",
                    )
                    expr.message = msg
                return expr

            case "approach_threaten":
                expr = self._express_move_toward(sensation, intention)
                if intention.target_agent_id:
                    msg = create_message(
                        tick=sensation.tick,
                        sender_id=None,
                        receiver_id=intention.target_agent_id,
                        message_type=MessageType.THREAT,
                        content="Leave this area!",
                        payload={"demand": "leave"},
                    )
                    expr.message = msg
                return expr

            case "attack":
                if intention.target_agent_id:
                    return Expression(
                        action=Action(
                            type=ActionType.ATTACK,
                            target_agent_id=intention.target_agent_id,
                        ),
                        internal_monologue=f"Attacking agent {intention.target_agent_id}",
                    )
                return Expression(action=Action(type=ActionType.WAIT))

            case "share_resources":
                if intention.target_agent_id and sensation.own_inventory:
                    item = next(iter(sensation.own_inventory))
                    return Expression(
                        action=Action(
                            type=ActionType.GIVE,
                            target=item,
                            target_agent_id=intention.target_agent_id,
                            quantity=1,
                        ),
                        internal_monologue=f"Giving {item} to {intention.target_agent_id}",
                    )
                return Expression(action=Action(type=ActionType.WAIT))

            case _:
                return Expression(action=Action(type=ActionType.WAIT))

    def _express_move_toward(self, sensation: Sensation, intention: Intention) -> Expression:
        """Move toward intention's target position."""
        if intention.target_position:
            direction = self._direction_toward(sensation.own_position, intention.target_position)
            if direction:
                return Expression(action=Action(type=ActionType.MOVE, direction=direction))
        return Expression(action=Action(type=ActionType.WAIT))

    def _express_explore(self, sensation: Sensation) -> Expression:
        """Move in an exploration direction, avoiding water."""
        directions = list(Direction)
        idx = sensation.tick % len(directions)
        ax, ay = sensation.own_position

        for i in range(len(directions)):
            d = directions[(idx + i) % len(directions)]
            dx, dy = d.value
            tx, ty = ax + dx, ay + dy
            for tile in sensation.visible_tiles:
                if (tile.x, tile.y) == (tx, ty) and tile.tile_type != "water":
                    return Expression(action=Action(type=ActionType.MOVE, direction=d))

        return Expression(action=Action(type=ActionType.WAIT))

    def _find_nearest_resource(
        self, sensation: Sensation, resource_kind: str
    ) -> tuple[int, int] | None:
        """Find nearest tile with desired resource (includes agent's own tile)."""
        ax, ay = sensation.own_position
        best_dist = float("inf")
        best_pos = None

        for tile in sensation.visible_tiles:
            if tile.tile_type == "water" and resource_kind != "water_source":
                continue
            for kind, qty in tile.resources:
                if kind == resource_kind and qty > 0:
                    dist = math.sqrt((tile.x - ax) ** 2 + (tile.y - ay) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (tile.x, tile.y)

        return best_pos

    def _direction_toward(
        self, from_pos: tuple[int, int], to_pos: tuple[int, int]
    ) -> Direction | None:
        """Get cardinal direction toward target."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if dx == 0 and dy == 0:
            return None
        if abs(dx) >= abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if dy > 0 else Direction.NORTH

    def _direction_away(
        self, from_pos: tuple[int, int], threat_pos: tuple[int, int]
    ) -> Direction | None:
        """Get cardinal direction away from threat."""
        dx = from_pos[0] - threat_pos[0]
        dy = from_pos[1] - threat_pos[1]
        if dx == 0 and dy == 0:
            # On same tile — flee north as fallback
            return Direction.NORTH
        if abs(dx) >= abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if dy > 0 else Direction.NORTH

    def _any_agent_on_same_tile(self, sensation: Sensation) -> bool:
        """Check if any visible agent is on our tile."""
        for tile in sensation.visible_tiles:
            if (tile.x, tile.y) == sensation.own_position:
                return len(tile.occupants) > 0
        return False

    def _agent_on_same_tile(self, sensation: Sensation) -> AgentSummary | None:
        """Get first visible agent on our tile."""
        for agent in sensation.visible_agents:
            if agent.position == sensation.own_position:
                return agent
        return None

    def _nearest_agent(self, sensation: Sensation) -> AgentSummary | None:
        """Find nearest visible agent."""
        ax, ay = sensation.own_position
        best_dist = float("inf")
        best = None
        for agent in sensation.visible_agents:
            dist = abs(agent.position[0] - ax) + abs(agent.position[1] - ay)
            if dist < best_dist:
                best_dist = dist
                best = agent
        return best
