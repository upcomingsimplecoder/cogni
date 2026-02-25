"""Behavioral fingerprints and pattern analysis.

Analyze action patterns and detect behavioral shifts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schema import TrajectoryDataset


class BehavioralAnalyzer:
    """Analyze action patterns and behavioral signatures."""

    def behavioral_fingerprint(self, dataset: TrajectoryDataset, agent_id: str) -> dict:
        """Compute behavioral fingerprint for an agent.

        Returns: {
            "action_distribution": {"move": 0.15, "gather": 0.50, ...},
            "social_ratio": 0.12,
            "exploration_ratio": 0.25,
            "risk_index": 0.67,
            "consistency": 0.85
        }
        """
        agent_snapshots = [s for s in dataset.agent_snapshots if s.agent_id == agent_id]
        if not agent_snapshots:
            return {
                "action_distribution": {},
                "social_ratio": 0.0,
                "exploration_ratio": 0.0,
                "risk_index": 0.0,
                "consistency": 0.0,
            }

        # Sort by tick
        agent_snapshots.sort(key=lambda s: s.tick)

        # Action distribution
        action_counts: dict[str, int] = {}
        for snapshot in agent_snapshots:
            action = snapshot.action_type
            action_counts[action] = action_counts.get(action, 0) + 1

        total_actions = len(agent_snapshots)
        action_distribution = {a: c / total_actions for a, c in action_counts.items()}

        # Social ratio: proportion of social actions (give, send_message, attack)
        social_actions = ["give", "send_message", "attack"]
        social_count = sum(action_counts.get(a, 0) for a in social_actions)
        social_ratio = social_count / total_actions if total_actions > 0 else 0.0

        # Exploration ratio: proportion of move actions
        exploration_count = action_counts.get("move", 0)
        exploration_ratio = exploration_count / total_actions if total_actions > 0 else 0.0

        # Risk index: weighted by risky actions (attack, aggression trait)
        risky_actions = ["attack", "fight"]
        risky_count = sum(action_counts.get(a, 0) for a in risky_actions)

        # Also factor in average aggression trait
        avg_aggression = sum(s.traits.get("aggression", 0.5) for s in agent_snapshots) / len(
            agent_snapshots
        )

        risk_index = (
            (risky_count / total_actions) * 0.7 + avg_aggression * 0.3 if total_actions > 0 else 0.0
        )

        # Consistency: inverse of action entropy (more consistent = lower entropy)
        consistency = 1.0 - self._action_entropy(action_distribution)

        return {
            "action_distribution": action_distribution,
            "social_ratio": social_ratio,
            "exploration_ratio": exploration_ratio,
            "risk_index": risk_index,
            "consistency": consistency,
        }

    def behavioral_shift_detection(
        self, dataset: TrajectoryDataset, agent_id: str, window: int = 50
    ) -> list[dict]:
        """Detect when agent behavior changes significantly.

        Uses sliding window comparison.

        Returns: [{
            "tick": 100,
            "shift_magnitude": 0.45,
            "before_fingerprint": {...},
            "after_fingerprint": {...}
        }, ...]
        """
        agent_snapshots = [s for s in dataset.agent_snapshots if s.agent_id == agent_id]
        if len(agent_snapshots) < window * 2:
            return []

        # Sort by tick
        agent_snapshots.sort(key=lambda s: s.tick)

        shifts = []

        # Slide window through timeline
        for i in range(window, len(agent_snapshots) - window):
            # Get action distributions for before and after windows
            before_window = agent_snapshots[i - window : i]
            after_window = agent_snapshots[i : i + window]

            before_dist = self._action_distribution(before_window)
            after_dist = self._action_distribution(after_window)

            # Compute shift magnitude (Euclidean distance between distributions)
            shift_mag = self._distribution_distance(before_dist, after_dist)

            # Only record significant shifts (threshold: 0.3)
            if shift_mag > 0.3:
                shifts.append(
                    {
                        "tick": agent_snapshots[i].tick,
                        "shift_magnitude": shift_mag,
                        "before_distribution": before_dist,
                        "after_distribution": after_dist,
                    }
                )

        return shifts

    def archetype_fidelity(self, dataset: TrajectoryDataset) -> dict:
        """How closely does each agent's actual behavior match archetype template?

        Returns: {
            "agent_id_1": 0.85,
            "agent_id_2": 0.62,
            ...
        }
        """
        # Define expected behavior for each archetype
        archetype_templates = {
            "gatherer": {"gather": 0.5, "eat": 0.15, "drink": 0.15, "rest": 0.1, "move": 0.1},
            "explorer": {"move": 0.5, "scout": 0.2, "gather": 0.2, "rest": 0.1},
            "cooperator": {
                "give": 0.3,
                "send_message": 0.2,
                "gather": 0.3,
                "move": 0.1,
                "rest": 0.1,
            },
            "aggressive": {"attack": 0.3, "fight": 0.2, "move": 0.2, "gather": 0.2, "rest": 0.1},
        }

        # Get all unique agent IDs
        agent_ids = set(s.agent_id for s in dataset.agent_snapshots)

        result = {}

        for agent_id in agent_ids:
            agent_snapshots = [s for s in dataset.agent_snapshots if s.agent_id == agent_id]
            if not agent_snapshots:
                continue

            # Get archetype
            archetype = agent_snapshots[0].archetype

            # Get actual action distribution
            actual_dist = self._action_distribution(agent_snapshots)

            # Get expected distribution for archetype
            expected_dist = archetype_templates.get(archetype, {})

            # Compute fidelity (1 - distance)
            if expected_dist:
                distance = self._distribution_distance(actual_dist, expected_dist)
                fidelity = max(0.0, 1.0 - distance)
            else:
                fidelity = 0.0

            result[agent_id] = fidelity

        return result

    def _action_distribution(self, snapshots: list) -> dict[str, float]:
        """Compute action distribution from snapshots."""
        action_counts: dict[str, int] = {}
        for snapshot in snapshots:
            action = snapshot.action_type
            action_counts[action] = action_counts.get(action, 0) + 1

        total = len(snapshots)
        if total == 0:
            return {}

        return {a: c / total for a, c in action_counts.items()}

    def _action_entropy(self, distribution: dict[str, float]) -> float:
        """Compute normalized entropy of action distribution (0-1 scale)."""
        if not distribution:
            return 0.0

        import math

        entropy = 0.0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)

        # Normalize by maximum entropy (log2 of number of actions)
        max_entropy = math.log2(len(distribution)) if len(distribution) > 1 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _distribution_distance(self, dist1: dict[str, float], dist2: dict[str, float]) -> float:
        """Compute Euclidean distance between two action distributions."""
        # Get all actions
        all_actions = set(dist1.keys()) | set(dist2.keys())

        # Compute squared differences
        squared_diff = 0.0
        for action in all_actions:
            p1 = dist1.get(action, 0.0)
            p2 = dist2.get(action, 0.0)
            squared_diff += (p1 - p2) ** 2

        return float(squared_diff**0.5)
