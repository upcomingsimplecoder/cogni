"""Plugin hooks documentation for AUTOCOG extension points.

This module documents the extension points available in AUTOCOG for plugins
to hook into and extend the system's behavior.

## Extension Points

### 1. Decision Strategies

Strategies control how agents make decisions. Register via `@PluginRegistry.register_strategy`.

**Interface:**
```python
class CustomStrategy:
    def form_intention(
        self, sensation: Sensation, reflection: Reflection
    ) -> Intention:
        '''Decide what the agent wants to do based on perception.'''
        pass

    def express(
        self,
        sensation: Sensation,
        reflection: Reflection,
        intention: Intention
    ) -> Expression:
        '''Convert intention into concrete action.'''
        pass
```

**Example:**
```python
from src.plugins import PluginRegistry
from src.awareness.types import Sensation, Reflection, Intention, Expression
from src.simulation.actions import Action, ActionType

@PluginRegistry.register_strategy("cautious_gatherer")
class CautiousGathererStrategy:
    def form_intention(self, sensation, reflection):
        # Prioritize safety over resource gathering
        if reflection.threat_level > 0.3:
            return Intention(primary_goal="flee", confidence=0.9)
        return Intention(primary_goal="gather_food", confidence=0.7)

    def express(self, sensation, reflection, intention):
        if intention.primary_goal == "flee":
            action = Action(type=ActionType.FLEE)
        else:
            action = Action(type=ActionType.GATHER, target="berries")
        return Expression(action=action)
```

### 2. Action Handlers

Add new action types beyond the built-in ones. Register via `@PluginRegistry.register_action`.

**Interface:**
```python
def custom_action_handler(
    action: Action,
    agent: Agent,
    world: World,
    registry: object
) -> ActionResult:
    '''Execute custom action and return result.'''
    pass
```

**Example:**
```python
from src.plugins import PluginRegistry
from src.simulation.actions import ActionResult

@PluginRegistry.register_action("meditate")
def meditate_handler(action, agent, world, registry):
    # Restore mental state (hypothetical)
    return ActionResult(
        action=action,
        success=True,
        message="Agent meditated and regained focus.",
        needs_delta={"energy": 5.0}
    )
```

### 3. Agent Archetypes

Define new personality profiles. Register via `PluginRegistry.register_archetype`.

**Interface:**
```python
PluginRegistry.register_archetype(
    name: str,
    traits: dict[str, float],  # cooperation_tendency, curiosity, etc.
    color: str = "white",
    symbol: str = "?"
)
```

**Example:**
```python
from src.plugins import PluginRegistry

PluginRegistry.register_archetype(
    name="scientist",
    traits={
        "cooperation_tendency": 0.6,
        "curiosity": 0.95,
        "risk_tolerance": 0.5,
        "resource_sharing": 0.7,
        "aggression": 0.1,
        "sociability": 0.4,
    },
    color="purple",
    symbol="S"
)
```

### 4. Evaluation Strategies

Custom reflection logic for evaluating agent state. Register via
`@PluginRegistry.register_evaluation`.

**Interface:**
```python
class CustomEvaluation:
    def evaluate(
        self,
        agent: Agent,
        engine: Any,
        sensation: Sensation
    ) -> Reflection:
        '''Analyze agent's situation and return reflection.'''
        pass
```

**Example:**
```python
from src.plugins import PluginRegistry
from src.awareness.types import Reflection

@PluginRegistry.register_evaluation("paranoid")
class ParanoidEvaluation:
    def evaluate(self, agent, engine, sensation):
        # Always assume high threat
        return Reflection(
            last_action_succeeded=True,
            need_trends={},
            threat_level=0.8,  # Always paranoid
            opportunity_score=0.2
        )
```

### 5. Metrics

Custom metrics for simulation analysis. Register via `@PluginRegistry.register_metric`.

**Interface:**
```python
def custom_metric(engine: Any) -> float:
    '''Extract custom metric from simulation state.'''
    pass
```

**Example:**
```python
from src.plugins import PluginRegistry

@PluginRegistry.register_metric("cooperation_index")
def cooperation_index(engine):
    # Calculate average cooperation tendency across all agents
    total = 0.0
    count = 0
    for agent in engine.registry.all_agents():
        total += agent.profile.traits.cooperation_tendency
        count += 1
    return total / count if count > 0 else 0.0
```

## Plugin Discovery

Plugins are automatically discovered and loaded from the `plugins/` directory
at the project root. Place your plugin files there and they will be loaded
when the simulation starts.

Use `PluginLoader.load_all()` to manually trigger plugin loading:

```python
from src.plugins import PluginLoader

stats = PluginLoader.load_all()
print(f"Loaded {stats['strategies']} strategies")
```

## Best Practices

1. **Single Responsibility**: One plugin file per extension type
2. **Documentation**: Include docstrings explaining plugin behavior
3. **Error Handling**: Plugins should handle errors gracefully
4. **Testing**: Write tests for custom plugins
5. **Naming**: Use descriptive names for registrations

## Type Imports

```python
from src.awareness.types import Sensation, Reflection, Intention, Expression
from src.simulation.actions import Action, ActionType, ActionResult
from src.simulation.entities import Agent
from src.simulation.world import World
from src.plugins import PluginRegistry
```
"""

from __future__ import annotations

# No implementation needed - this is documentation only
