# AUTOCOG Visualization System — Architecture Overview

## Overview

The AUTOCOG visualization system has been refactored from monolithic HTML templates into a modular ES module architecture that supports multiple "lenses" for viewing the simulation from different perspectives.

## Key Concepts

### Lenses
A **lens** is a pluggable visualization mode that determines:
- How agents are styled (color, shape, size, glow)
- What canvas layers are rendered (grid, overlays, connections)
- What the side panel displays when an agent is selected
- What timeline markers appear
- Custom keyboard shortcuts

**Built-in Lenses:**
1. **Physical** (active) — Physical world view with resources, actions, health
2. **Social** (future) — Trust networks, coalitions, Theory of Mind
3. **Cognitive** (future) — SRIE cascade, metacognition, calibration
4. **Cultural** (future) — Cultural groups, transmission networks, learning styles
5. **Temporal** (future) — Agent trails, metric sparklines, emergence timeline

### Modes
- **Live Mode**: WebSocket connection to running simulation (`/ws`)
- **Dashboard Mode**: Replay from embedded JSON trajectory data

## Architecture

```
app.js (orchestrator)
  ├── core/
  │   ├── data-source.js      # WebSocket or embedded JSON
  │   ├── temporal-buffer.js  # Ring buffer of last N ticks
  │   ├── spatial-index.js    # Fast neighbor queries
  │   ├── network-graph.js    # Relationship graph
  │   ├── time-series-store.js # Metric history
  │   ├── event-bus.js        # Pub/sub messaging
  │   ├── colors.js           # Color palettes
  │   └── config.js           # Constants
  │
  ├── renderers/
  │   ├── canvas-compositor.js # 10-layer rendering system
  │   ├── grid-renderer.js     # Grid + resources
  │   ├── agent-renderer.js    # Agents + decorations
  │   └── overlay-renderer.js  # Lines, arrows, hulls
  │
  ├── lenses/
  │   ├── lens-base.js        # Abstract interface
  │   ├── physical-lens.js    # Physical world view
  │   ├── social-lens.js      # (future)
  │   ├── cognitive-lens.js   # (future)
  │   ├── cultural-lens.js    # (future)
  │   └── temporal-lens.js    # (future)
  │
  ├── panels/
  │   ├── inspector-panel.js      # Agent detail panel
  │   ├── timeline-panel.js       # Scrubber + markers
  │   └── side-panel-manager.js   # Routes to active lens
  │
  └── utils/
      └── export.js           # Screenshot + JSON export
```

## Canvas Layer System

The canvas uses a 10-layer compositing system with try-catch isolation per layer:

| Layer | Name | Content |
|-------|------|---------|
| 0 | Background | Dark fill, time-of-day tint |
| 1 | Grid | Grid lines |
| 2 | Resources | Resource tiles |
| 3 | Heatmap | Density overlays (future) |
| 4 | Territory | Coalition/cultural group hulls (future) |
| 5 | Connections | Trust, cultural, proximity edges |
| 6 | Agents | Agent bodies (styled per lens) |
| 7 | Decorations | Health bars, belief bubbles, action icons |
| 8 | Annotations | Emergence labels, callouts (future) |
| 9 | UI Overlay | Selection highlight, tooltips, range circles |

Each lens can provide its own layers or customize existing ones.

## Data Flow

### Live Mode
```
Python Server (FastAPI)
  ↓ WebSocket /ws
DataSource (live)
  ↓ onTick
app.handleTickData()
  ├→ TemporalBuffer.addTick()
  ├→ SpatialIndex.update()
  ├→ NetworkGraph.update()
  └→ TimeSeriesStore.record()
  ↓
Compositor.render(renderCtx)
  ├→ activeLens.getCanvasLayers()
  └→ forEach layer: layer.draw(renderCtx)
```

### Dashboard Mode
```
Embedded __DASHBOARD_DATA__
  ↓
DataSource (dashboard)
  ↓ getTickData(index)
app.loadTick()
  ├→ TemporalBuffer.addTick()
  ├→ SpatialIndex.update()
  └→ render()
```

## RenderContext Object

Every render function receives a `renderCtx` object:

```javascript
{
  // Canvas
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  config: {worldWidth, worldHeight, cellSize, offsetX, offsetY},

  // State
  mode: 'live' | 'dashboard',
  currentTick: number,
  activeLens: LensInstance,
  agents: Agent[],
  selectedAgentId: string | null,

  // Data structures
  temporalBuffer: TemporalBuffer,
  spatialIndex: SpatialIndex,
  networkGraph: NetworkGraph,
  timeSeriesStore: TimeSeriesStore,

  // Events
  emergenceEvents: string[],

  // Toggles
  toggles: {grid, heatmap, trails, ...}
}
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1-5` | Switch lens (Physical, Social, Cognitive, Cultural, Temporal) |
| `Space` | Toggle pause |
| `←/→` | Previous/next tick (dashboard mode) |
| `p` | Screenshot (clipboard or download) |
| `e` | Export JSON state |
| `g` | Toggle grid |
| `h` | Toggle heatmap |
| `t` | Toggle trails |
| `Esc` | Clear selection |

## Adding a New Lens

1. Create `lenses/my-lens.js`:
```javascript
import { LensBase } from './lens-base.js';

export class MyLens extends LensBase {
  constructor() {
    super('my-lens-name');
  }

  getCanvasLayers(renderCtx) {
    return [
      { name: 'my-layer', zIndex: 5, draw: (ctx) => { /* draw */ } }
    ];
  }

  getAgentStyle(agent, renderCtx) {
    return {
      color: '#ff0000',
      shape: 'circle',
      size: 0.3
    };
  }

  renderSidePanel(container, renderCtx) {
    container.innerHTML = `<div>Custom panel content</div>`;
  }
}
```

2. Register in `app.js`:
```javascript
import { MyLens } from './lenses/my-lens.js';

_initializeLenses() {
  this.lenses.set('physical', new PhysicalLens());
  this.lenses.set('my-lens', new MyLens());
}
```

3. Add keyboard binding in `core/config.js`:
```javascript
KEYBOARD_BINDINGS = {
  '6': 'switchLens:my-lens',
  // ...
}
```

## Testing

### Live Mode
1. Start simulation with live server:
```python
from src.simulation.engine import SimulationEngine
from src.visualization.realtime import LiveServer

engine = SimulationEngine(config)
server = LiveServer(port=8001, open_browser=True)
server.set_engine(engine)
server.start()

while not engine.is_over():
    tick_record = engine.step_all()
    server.broadcast(tick_to_json(engine, tick_record))

server.stop()
```

2. Open browser to `http://localhost:8001`
3. Verify:
   - WebSocket connects (green indicator)
   - Agents render and move
   - Click agent → inspector updates
   - Keys `1-5` switch lenses (2-5 show placeholder)
   - `P` key screenshots canvas

### Dashboard Mode
1. Generate trajectory:
```python
from src.visualization.dashboard import DashboardGenerator

gen = DashboardGenerator.from_file('outputs/run_ABC/trajectory.jsonl')
gen.generate('outputs/run_ABC/dashboard.html')
```

2. Serve dashboard:
```python
from src.visualization.server import start_server

start_server('outputs/run_ABC/dashboard.html', port=8000)
```

3. Verify:
   - Scrubber moves through ticks
   - Play/pause works
   - All Physical lens features work

## Performance

- **Target**: 60 FPS at 100 agents
- **Spatial Index**: O(1) neighbor queries via grid
- **Temporal Buffer**: Rolling 500-tick window
- **Layer Isolation**: Failed layer doesn't crash render loop
- **Module Loading**: Native ES modules (no bundler needed)

## Troubleshooting

### "Failed to load module"
- Ensure `StaticFiles` mounted in FastAPI (realtime.py line 96)
- Check server serves from visualization root
- Verify MIME type for `.js` is `application/javascript`

### "WebSocket disconnected"
- Check `/ws` endpoint is available
- Verify engine is set via `server.set_engine(engine)`
- Look for Python exceptions in server logs

### "Agent style is undefined"
- Ensure active lens implements `getAgentStyle()`
- Check renderCtx has valid agent data
- Verify lens was registered in `_initializeLenses()`

### Canvas rendering is blank
- Open browser console for JavaScript errors
- Check `renderCtx.agents` is populated
- Verify `worldConfig` has correct dimensions
- Ensure canvas size matches world dimensions

## Future Enhancements

### Phase 1.2: Social Lens
- Trust network edges (weighted, colored by strength)
- Coalition boundaries (convex hulls)
- Theory of Mind prediction arrows
- Social relationship heatmap

### Phase 1.3: Cognitive Lens
- SRIE cascade visualization (sensation → reflection → intention → execution)
- Metacognition state indicator (System 1/2)
- Calibration curve overlays
- Confidence glow on agents

### Phase 1.4: Cultural Lens
- Cultural group territories (Voronoi-style)
- Variant flow diagram
- Learning style badges
- Transmission bias network

### Phase 1.5: Temporal Lens
- Agent movement trails (fading)
- Metric sparklines (health, hunger, thirst, energy)
- Emergence event timeline
- Population-wide statistics graphs

## Design Principles

1. **Modularity**: One responsibility per module
2. **Fail-safe**: Try-catch around every layer
3. **Extensibility**: New lenses don't touch core
4. **Zero framework**: Vanilla JS ES modules only
5. **No build step**: Browser-native loading
6. **Backward compatible**: Old templates still work
