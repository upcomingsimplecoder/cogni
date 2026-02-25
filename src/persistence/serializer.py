"""State serialization: convert SimulationEngine to/from dict.

Handles full engine state including world, agents, memories, and awareness loops.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.simulation.engine import SimulationEngine


class StateSerializer:
    """Serialize and deserialize full simulation state."""

    SCHEMA_VERSION = 2

    def serialize(self, engine: SimulationEngine) -> dict:
        """Serialize full engine state to dict.

        Args:
            engine: The simulation engine to serialize

        Returns:
            Dictionary containing all engine state
        """
        # Serialize world
        world_data = {
            "width": engine.world.width,
            "height": engine.world.height,
            "seed": engine.world.seed,
            "tiles": self._serialize_tiles(engine),
        }

        # Serialize agents (living + dead)
        living_agents = []
        for agent in engine.registry.living_agents():
            living_agents.append(self._serialize_agent(agent))

        dead_agents = []
        for agent in engine.registry._dead.values():
            dead_agents.append(self._serialize_agent(agent))

        death_causes = {
            str(agent_id): cause for agent_id, cause in engine.registry._death_causes.items()
        }

        agents_data = {
            "living": living_agents,
            "dead": dead_agents,
            "death_causes": death_causes,
        }

        # Serialize memories
        memories_data = {}
        for agent_id, (episodic, social) in engine.registry._memories.items():
            memories_data[str(agent_id)] = {
                "episodic": self._serialize_episodic_memory(episodic),
                "social": self._serialize_social_memory(social),
            }

        # Serialize emergence detector state (basic event counts)
        # Only serialize basic stats, not full tracking history
        emergence_data = {
            "event_count": engine.emergence_detector.event_count,
        }

        # Serialize mailbox histories
        mailboxes_data = {}
        for agent_id_obj, mailbox in engine.message_bus._mailboxes.items():
            mailboxes_data[str(agent_id_obj)] = self._serialize_mailbox(mailbox)

        # Serialize awareness loop SRIE caches
        awareness_loops_data = {}
        for agent in engine.registry.living_agents():
            loop = engine.registry.get_awareness_loop(agent.agent_id)
            if loop:
                cache = loop.srie_cache_to_dict()
                if cache is not None:
                    awareness_loops_data[str(agent.agent_id)] = cache

        # Serialize trait evolution history
        trait_evolution_data = {
            "history": engine.trait_evolution.export_history(),
        }

        # Serialize lineage tree (if available)
        lineage_data: dict[str, Any] = {"roots": [], "nodes": {}}
        if (
            engine.population_manager
            and hasattr(engine.population_manager, "lineage_tracker")
            and engine.population_manager.lineage_tracker
        ):
            lineage_data = engine.population_manager.lineage_tracker.export_tree()

        # Serialize effectiveness engine state (if available)
        effectiveness_data: dict[str, Any] = {
            "nudge_scores": [],
            "router_records": [],
            "classifier_records": [],
            "quality_scores": {},
            "classifier_accuracy": {},
        }
        if hasattr(engine, "effectiveness_engine") and engine.effectiveness_engine:
            effectiveness_data = engine.effectiveness_engine.export_state()

        # Build full state
        return {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
            "config": engine.config.model_dump(),
            "state": {
                "tick": engine.state.tick,
                "day": engine.state.day,
                "time_of_day": engine.state.time_of_day,
            },
            "world": world_data,
            "agents": agents_data,
            "memories": memories_data,
            "emergence": emergence_data,
            "mailboxes": mailboxes_data,
            "awareness_loops": awareness_loops_data,
            "trait_evolution": trait_evolution_data,
            "lineage": lineage_data,
            "effectiveness": effectiveness_data,
        }

    def deserialize(self, data: dict, config_override: dict | None = None) -> SimulationEngine:
        """Reconstruct engine from serialized dict.

        Args:
            data: Serialized state dictionary
            config_override: Optional config fields to override (for branching)

        Returns:
            Reconstructed SimulationEngine
        """
        # Validate schema version
        version = data.get("schema_version", 1)
        if version < self.SCHEMA_VERSION:
            from src.persistence.migration import SchemaMigration

            data = SchemaMigration.migrate(data)

        # Reconstruct config
        from src.config import SimulationConfig

        config_dict = data["config"].copy()
        if config_override:
            config_dict.update(config_override)
        config = SimulationConfig(**config_dict)

        # Create engine
        from src.simulation.engine import SimulationEngine

        engine = SimulationEngine(config)

        # Rebuild world tiles
        self._deserialize_tiles(engine, data["world"])

        # Spawn agents
        self._deserialize_agents(engine, data["agents"])

        # Rebuild memories
        self._deserialize_memories(engine, data["memories"])

        # Rebuild awareness loops for living agents
        self._rebuild_awareness_loops(engine)

        # Restore mailbox histories
        if "mailboxes" in data:
            self._deserialize_mailboxes(engine, data["mailboxes"])

        # Restore SRIE caches
        if "awareness_loops" in data:
            self._restore_srie_caches(engine, data["awareness_loops"])

        # Restore trait evolution history
        if "trait_evolution" in data:
            engine.trait_evolution.import_history(data["trait_evolution"].get("history", []))

        # Restore lineage tree
        if (
            "lineage" in data
            and data["lineage"].get("nodes")
            and engine.population_manager
            and hasattr(engine.population_manager, "lineage_tracker")
            and engine.population_manager.lineage_tracker
        ):
            engine.population_manager.lineage_tracker.import_tree(data["lineage"])

        # Restore effectiveness engine state
        if (
            "effectiveness" in data
            and hasattr(engine, "effectiveness_engine")
            and engine.effectiveness_engine
        ):
            engine.effectiveness_engine.import_state(data["effectiveness"])

        # Restore simulation state
        engine.state.tick = data["state"]["tick"]
        engine.state.day = data["state"]["day"]
        engine.state.time_of_day = data["state"]["time_of_day"]

        # Restore emergence detector state (event count only — tracking state resets)
        # The detector's internal tracking (_active_clusters, _sharing_log, etc.)
        # is not serialized since it rebuilds naturally during simulation.

        return engine

    def _serialize_tiles(self, engine: SimulationEngine) -> list[list[dict]]:
        """Serialize world tiles as 2D array."""
        tiles = []
        for row in engine.world.tiles:
            tile_row = []
            for tile in row:
                tile_data = {
                    "type": tile.type.value,
                    "x": tile.x,
                    "y": tile.y,
                    "resources": [
                        {
                            "kind": res.kind,
                            "quantity": res.quantity,
                            "max_quantity": res.max_quantity,
                            "regen_rate": res.regen_rate,
                        }
                        for res in tile.resources
                    ],
                    "occupants": [str(occ) for occ in tile.occupants],
                }
                tile_row.append(tile_data)
            tiles.append(tile_row)
        return tiles

    def _deserialize_tiles(self, engine: SimulationEngine, world_data: dict) -> None:
        """Rebuild world tiles from serialized data."""
        from src.simulation.world import Resource, Tile, TileType

        tiles_data = world_data["tiles"]
        engine.world.tiles = []

        for row_data in tiles_data:
            row = []
            for tile_data in row_data:
                tile_type = TileType(tile_data["type"])
                tile = Tile(type=tile_type, x=tile_data["x"], y=tile_data["y"])

                # Restore resources
                for res_data in tile_data["resources"]:
                    resource = Resource(
                        kind=res_data["kind"],
                        quantity=res_data["quantity"],
                        max_quantity=res_data["max_quantity"],
                        regen_rate=res_data["regen_rate"],
                    )
                    tile.resources.append(resource)

                # Occupants will be restored when agents are spawned
                tile.occupants = set()

                row.append(tile)
            engine.world.tiles.append(row)

    def _serialize_agent(self, agent: Any) -> dict:
        """Serialize a single agent."""
        return {
            "agent_id": str(agent.agent_id),
            "name": agent.profile.name if agent.profile else "agent",
            "archetype": agent.profile.archetype if agent.profile else "survivalist",
            "x": agent.x,
            "y": agent.y,
            "alive": agent.alive,
            "ticks_alive": agent.ticks_alive,
            "color": agent.color,
            "needs": {
                "hunger": agent.needs.hunger,
                "thirst": agent.needs.thirst,
                "energy": agent.needs.energy,
                "health": agent.needs.health,
                "temperature": agent.needs.temperature,
            },
            "inventory": dict(agent.inventory),
            "traits": (agent.profile.traits.as_dict() if agent.profile else {}),
        }

    def _deserialize_agents(self, engine: SimulationEngine, agents_data: dict) -> None:
        """Spawn agents from serialized data."""
        from src.agents.identity import AgentID, AgentProfile, PersonalityTraits

        # Spawn living agents
        for agent_data in agents_data["living"]:
            # Create profile
            agent_id_obj = AgentID(agent_data["agent_id"])
            traits = PersonalityTraits(**agent_data["traits"])
            profile = AgentProfile(
                agent_id=agent_id_obj,
                name=agent_data["name"],
                archetype=agent_data["archetype"],
                traits=traits,
            )

            # Spawn agent
            agent = engine.registry.spawn(
                profile,
                agent_data["x"],
                agent_data["y"],
                color=agent_data.get("color", "white"),
            )

            # Restore needs
            agent.needs.hunger = agent_data["needs"]["hunger"]
            agent.needs.thirst = agent_data["needs"]["thirst"]
            agent.needs.energy = agent_data["needs"]["energy"]
            agent.needs.health = agent_data["needs"]["health"]
            agent.needs.temperature = agent_data["needs"]["temperature"]

            # Restore inventory
            agent.inventory = agent_data["inventory"].copy()

            # Restore ticks_alive
            agent.ticks_alive = agent_data["ticks_alive"]

        # Restore dead agents
        for agent_data in agents_data["dead"]:
            # Create profile
            agent_id = AgentID(agent_data["agent_id"])
            traits = PersonalityTraits(**agent_data["traits"])
            profile = AgentProfile(
                agent_id=agent_id,
                name=agent_data["name"],
                archetype=agent_data["archetype"],
                traits=traits,
            )

            # Create agent manually (not spawned)
            from src.simulation.entities import Agent

            agent = Agent(
                x=agent_data["x"],
                y=agent_data["y"],
                agent_id=agent_id,
                profile=profile,
                alive=False,
                ticks_alive=agent_data["ticks_alive"],
                color=agent_data.get("color", "white"),
            )

            # Restore needs
            agent.needs.hunger = agent_data["needs"]["hunger"]
            agent.needs.thirst = agent_data["needs"]["thirst"]
            agent.needs.energy = agent_data["needs"]["energy"]
            agent.needs.health = agent_data["needs"]["health"]
            agent.needs.temperature = agent_data["needs"]["temperature"]

            # Restore inventory
            agent.inventory = agent_data["inventory"].copy()

            # Add to dead registry
            engine.registry._dead[agent_id] = agent

        # Restore death causes
        for agent_id_str, cause in agents_data["death_causes"].items():
            # Find the AgentID in dead agents
            found_agent_id: AgentID | None = None
            for aid in engine.registry._dead:
                if str(aid) == agent_id_str:
                    found_agent_id = aid
                    break
            if found_agent_id:
                engine.registry._death_causes[found_agent_id] = cause

    def _serialize_episodic_memory(self, episodic: Any) -> dict:
        """Serialize episodic memory."""
        episodes = []
        for episode in episodic._episodes:
            episodes.append(
                {
                    "tick": episode.tick,
                    "action_type": episode.action_type,
                    "target": episode.target,
                    "success": episode.success,
                    "needs_delta": episode.needs_delta,
                    "location": list(episode.location),
                    "involved_agent": str(episode.involved_agent)
                    if episode.involved_agent
                    else None,
                }
            )

        interactions = []
        for interaction in episodic._interaction_log:
            interactions.append(
                {
                    "other_agent_id": str(interaction.other_agent_id),
                    "tick": interaction.tick,
                    "was_positive": interaction.was_positive,
                    "interaction_type": interaction.interaction_type,
                }
            )

        return {
            "episodes": episodes,
            "interactions": interactions,
        }

    def _serialize_social_memory(self, social: Any) -> dict:
        """Serialize social memory."""
        relationships = {}
        for agent_id, rel in social._relationships.items():
            relationships[str(agent_id)] = {
                "trust": rel.trust,
                "interaction_count": rel.interaction_count,
                "last_interaction_tick": rel.last_interaction_tick,
                "net_resources_given": rel.net_resources_given,
                "was_attacked_by": rel.was_attacked_by,
                "was_helped_by": rel.was_helped_by,
            }
        return {"relationships": relationships}

    def _deserialize_memories(self, engine: SimulationEngine, memories_data: dict) -> None:
        """Rebuild memory systems from serialized data."""
        from src.memory.episodic import Episode, EpisodicMemory, InteractionOutcome
        from src.memory.social import Relationship, SocialMemory

        for agent_id_str, memory_data in memories_data.items():
            # Find the agent
            agent_id: Any = None
            for aid in list(engine.registry._agents.keys()) + list(engine.registry._dead.keys()):
                if str(aid) == agent_id_str:
                    agent_id = aid
                    break

            if agent_id is None:
                continue

            # Rebuild episodic memory
            episodic = EpisodicMemory()
            episodic_data: dict[str, Any] = {}
            if isinstance(memory_data, dict):
                episodic_data = memory_data.get("episodic", {})
            for ep_data in episodic_data.get("episodes", []):
                # Find involved agent if present
                involved_agent = None
                if ep_data["involved_agent"]:
                    all_aids = list(engine.registry._agents) + list(engine.registry._dead)
                    for aid in all_aids:
                        if str(aid) == ep_data["involved_agent"]:
                            involved_agent = aid
                            break

                episode = Episode(
                    tick=ep_data["tick"],
                    action_type=ep_data["action_type"],
                    target=ep_data["target"],
                    success=ep_data["success"],
                    needs_delta=ep_data["needs_delta"],
                    location=tuple(ep_data["location"]),
                    involved_agent=involved_agent,
                )
                episodic._episodes.append(episode)

            effectiveness_data_val: dict[str, Any] = {}
            if isinstance(memory_data, dict):
                effectiveness_data_val = memory_data.get("episodic", {})
            for int_data in effectiveness_data_val.get("interactions", []):
                # Find other agent
                other_agent_id = None
                all_aids = list(engine.registry._agents) + list(engine.registry._dead)
                for aid in all_aids:
                    if str(aid) == int_data["other_agent_id"]:
                        other_agent_id = aid
                        break

                if other_agent_id:
                    interaction = InteractionOutcome(
                        other_agent_id=other_agent_id,
                        tick=int_data["tick"],
                        was_positive=int_data["was_positive"],
                        interaction_type=int_data["interaction_type"],
                    )
                    episodic._interaction_log.append(interaction)

            # Rebuild social memory
            social = SocialMemory()
            for other_id_str, rel_data in memory_data["social"]["relationships"].items():
                # Find other agent
                other_agent_id = None
                all_aids = list(engine.registry._agents) + list(engine.registry._dead)
                for aid in all_aids:
                    if str(aid) == other_id_str:
                        other_agent_id = aid
                        break

                if other_agent_id:
                    relationship = Relationship(
                        other_agent_id=other_agent_id,
                        trust=rel_data["trust"],
                        interaction_count=rel_data["interaction_count"],
                        last_interaction_tick=rel_data["last_interaction_tick"],
                        net_resources_given=rel_data["net_resources_given"],
                        was_attacked_by=rel_data["was_attacked_by"],
                        was_helped_by=rel_data["was_helped_by"],
                    )
                    social._relationships[other_agent_id] = relationship

            # Register memories
            engine.registry.register_memory(agent_id, episodic, social)

    def _rebuild_awareness_loops(self, engine: SimulationEngine) -> None:
        """Rebuild awareness loops for living agents."""
        from src.awareness.architecture import build_awareness_loop

        for agent in engine.registry.living_agents():
            # Use the agent's archetype to determine architecture
            architecture_name = engine.config.default_architecture

            # Build awareness loop
            loop = build_awareness_loop(agent, architecture_name)
            engine.registry.register_awareness_loop(agent.agent_id, loop)

    def _serialize_mailbox(self, mailbox: Any) -> dict:
        """Serialize a single mailbox's history."""
        records = []
        for record in mailbox._history:
            msg = record.message
            records.append(
                {
                    "message": {
                        "id": msg.id,
                        "tick": msg.tick,
                        "sender_id": str(msg.sender_id) if msg.sender_id else None,
                        "receiver_id": str(msg.receiver_id) if msg.receiver_id else None,
                        "message_type": msg.message_type.value,
                        "content": msg.content,
                        "payload": msg.payload,
                        "energy_cost": msg.energy_cost,
                    },
                    "was_sender": record.was_sender,
                    "processed": record.processed,
                }
            )
        return {"history": records}

    def _deserialize_mailboxes(self, engine: SimulationEngine, data: dict) -> None:
        """Restore mailbox histories from serialized data."""
        from src.communication.protocol import Message, MessageRecord, MessageType

        for agent_id_str, mailbox_data in data.items():
            # Find the matching AgentID
            agent_id = None
            for aid in list(engine.registry._agents.keys()) + list(engine.registry._dead.keys()):
                if str(aid) == agent_id_str:
                    agent_id = aid
                    break

            if agent_id is None:
                continue

            # Get or create the mailbox
            mailbox = engine.message_bus.get_or_create_mailbox(agent_id)

            # Restore history
            for record_data in mailbox_data.get("history", []):
                msg_data = record_data["message"]
                # Reconstruct Message (frozen dataclass)
                message = Message(
                    id=msg_data["id"],
                    tick=msg_data["tick"],
                    sender_id=msg_data["sender_id"],
                    receiver_id=msg_data["receiver_id"],
                    message_type=MessageType(msg_data["message_type"]),
                    content=msg_data["content"],
                    payload=msg_data.get("payload", {}),
                    energy_cost=msg_data.get("energy_cost", 0.5),
                )
                record = MessageRecord(
                    message=message,
                    was_sender=record_data["was_sender"],
                    processed=record_data.get("processed", False),
                )
                mailbox._history.append(record)

    def _restore_srie_caches(self, engine: SimulationEngine, data: dict) -> None:
        """Restore SRIE cached state for awareness loops."""
        for agent_id_str, cache_data in data.items():
            # Find the agent
            for agent in engine.registry.living_agents():
                if str(agent.agent_id) == agent_id_str:
                    loop = engine.registry.get_awareness_loop(agent.agent_id)
                    if loop:
                        loop.srie_cache_from_dict(cache_data)
                    break
