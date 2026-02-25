# Real-Time WebSocket Server for AUTOCOG

This module provides real-time visualization of the AUTOCOG simulation via WebSocket.

## Features

- **WebSocket Streaming**: Broadcasts tick data to connected browser clients in real-time
- **Live Dashboard**: Web-based UI showing agent positions, needs, actions, and messages
- **Thread-Safe**: Runs in a daemon thread, doesn't block the main simulation loop
- **Auto-Reconnect**: Browser client automatically reconnects if connection drops

## Dependencies

This module requires FastAPI and uvicorn (not included in base dependencies):

```bash
pip install fastapi uvicorn[standard]
```

## Usage

### Basic Setup

```python
from src.visualization.realtime import LiveServer, tick_to_json
from src.simulation.engine import SimulationEngine
from src.config import SimulationConfig

# Create engine and server
config = SimulationConfig(num_agents=5)
engine = SimulationEngine(config)
engine.setup_multi_agent()

server = LiveServer(port=8001, open_browser=True)
server.set_engine(engine)
server.start()

# Simulation loop
while not engine.is_over():
    tick_record = engine.step_all()

    # Broadcast to live clients
    tick_data = tick_to_json(engine, tick_record)
    server.broadcast(tick_data)

server.stop()
```

### API Endpoints

- `GET /` - Serves the live dashboard HTML
- `GET /api/config` - Returns simulation configuration (world size, agent archetypes, etc.)
- `WebSocket /ws` - WebSocket endpoint for receiving tick data

### Tick Data Format

The `tick_to_json()` function serializes engine state to:

```json
{
  "tick": 42,
  "day": 1,
  "time_of_day": "dawn",
  "living_count": 5,
  "dead_count": 0,
  "agents": [
    {
      "id": "abc123",
      "name": "gatherer_0",
      "archetype": "gatherer",
      "color": "#22c55e",
      "position": {"x": 32, "y": 28},
      "needs": {
        "hunger": 75.2,
        "thirst": 68.5,
        "energy": 82.1,
        "health": 100.0
      },
      "alive": true,
      "action": {
        "type": "gather",
        "success": true
      },
      "intention": {
        "primary_goal": "find_food",
        "confidence": 0.85
      },
      "internal_monologue": "Found berries nearby, gathering resources",
      "inventory": {"berries": 5},
      "traits": {
        "cooperation_tendency": 0.3,
        "curiosity": 0.2,
        ...
      }
    }
  ],
  "emergent_events": [
    "Cluster detected: 3 agents at (30,30)"
  ],
  "messages": [
    {
      "id": "msg123",
      "tick": 42,
      "sender": "abc123",
      "receiver": "def456",
      "type": "inform",
      "content": "Food at (10, 20)"
    }
  ]
}
```

## Architecture

- **LiveServer**: Manages FastAPI app and WebSocket connections
- **tick_to_json()**: Serializes TickRecord + engine state to JSON
- **Broadcast Queue**: Thread-safe queue for communication between simulation and WebSocket threads
- **Daemon Thread**: Server runs in background, doesn't prevent program exit

## Notes

- The server runs on `127.0.0.1` (localhost only) by default
- WebSocket clients automatically reconnect if disconnected
- Recent messages are limited to last 10 for performance
- Canvas renders at 8px per tile for clean visualization
