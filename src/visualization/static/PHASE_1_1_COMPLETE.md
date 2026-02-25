# Stream 1 Phase 1.1 Complete: JS Foundation + Physical Lens

## What Was Built

### Core Data Structures
- **`temporal-buffer.js`**: Ring buffer storing last 500 ticks, provides agent trails and metric time series
- **`spatial-index.js`**: Grid-based spatial queries for fast neighbor lookups
- **`network-graph.js`**: Weighted directed graph for relationships (trust, cultural, coalitions)
- **`time-series-store.js`**: Per-agent metric history for sparklines
- **`event-bus.js`**: Pub/sub system for cross-component communication
- **`data-source.js`**: Unified interface for WebSocket (live) and embedded JSON (dashboard) modes
- **`colors.js`**: Color palettes and interpolation utilities
- **`config.js`**: Central configuration constants

### Renderers
- **`canvas-compositor.js`**: 10-layer rendering system with per-layer try-catch isolation
- **`grid-renderer.js`**: Grid lines + resource tiles
- **`agent-renderer.js`**: Agent shapes (circle/diamond/hex/triangle/square), health bars, action icons
- **`overlay-renderer.js`**: Lines, arrows, convex hulls, heatmaps

### Lens System
- **`lens-base.js`**: Abstract base class defining lens interface
- **`physical-lens.js`**: **Physical world view** — extracted all existing rendering behavior:
  - Agent colors by archetype
  - Health bars
  - Action glyphs (⛏ gather, → move, ↗ give, ✕ attack, … rest, ○ talk)
  - Cultural transmission lines with decay
  - Cultural group rings
  - Resource tiles with opacity-based density
  - Full inspector panel (needs, traits, intention, monologue, inventory)

### Panels
- **`inspector-panel.js`**: Routes to active lens for agent details
- **`timeline-panel.js`**: Scrubber with live/dashboard mode support
- **`side-panel-manager.js`**: Manages sidebar content + emergence events feed

### Utilities
- **`export.js`**: Screenshot to clipboard (`P` key) + JSON state export

### Main Orchestrator
- **`app.js`**: Entry point that:
  - Auto-detects mode (live vs dashboard)
  - Initializes all core modules
  - Manages lens switching (keys 1-5)
  - Handles canvas interactions (click to select agent)
  - Keyboard shortcuts (grid, heatmap, trails, export)
  - Render loop with requestAnimationFrame
  - Updates spatial index, network graph, time series

### HTML Templates
- **`live_new.html`**: Thin template (250 lines) for WebSocket mode
- **`dashboard_new.html`**: Thin template (260 lines) for replay mode
- Both load `app.js` as ES module

### Python Integration
- Updated `realtime.py` to serve static JS files via FastAPI StaticFiles
- Updated `dashboard.py` to use new template (with fallback to old)
- Updated `server.py` to serve from visualization root with correct MIME types

## Architecture

### Layer Draw Order
```
0: Background (dark fill)
1: Grid lines
2: Resource tiles
3: (Heatmap - future)
4: (Territory hulls - future)
5: Transmission edges (cultural)
6: Agent bodies (styled per lens)
7: Agent decorations (health bars, action icons)
8: (Annotations - future)
9: (UI overlay - future)
```

### RenderContext Structure
```javascript
{
  ctx, canvas, config,
  mode: 'live' | 'dashboard',
  currentTick, activeLens, agents, selectedAgentId,
  temporalBuffer, spatialIndex, networkGraph, timeSeriesStore,
  emergenceEvents, toggles
}
```

### Lens Interface
```javascript
class LensBase {
  getCanvasLayers(renderCtx)        // → [{name, zIndex, draw}]
  renderSidePanel(container, renderCtx)
  getAgentStyle(agent, renderCtx)   // → {color, shape, size, glow, label}
  onAgentSelect(agentId, renderCtx)
  getTimelineMarkers(tick, renderCtx)
  getKeyBindings()
  onActivate(renderCtx)
  onDeactivate(renderCtx)
}
```

## What Works Now

### Live Mode (WebSocket)
1. Connects to `/ws` endpoint
2. Receives tick data from Python server
3. Buffers last 500 ticks in temporal buffer
4. Updates spatial index per tick
5. Renders agents with Physical lens
6. Shows cultural transmission lines with 2-tick decay
7. Agent selection → inspector panel
8. Keys 1-5 switch lenses (2-5 show placeholder)
9. `P` key screenshots canvas
10. `g`/`h`/`t` toggle grid/heatmap/trails

### Dashboard Mode (Replay)
1. Loads embedded JSON from `window.__DASHBOARD_DATA__`
2. Full timeline scrubber (0 to max_ticks)
3. Play/Pause/Reset controls
4. All Physical lens features
5. Emergence events timeline

## Verification Checklist

- [x] All core modules created
- [x] All renderers created
- [x] Physical lens extracts existing behavior
- [x] Lens switching works (keys 1-5)
- [x] Agent selection works (click canvas)
- [x] Inspector panel shows details
- [x] Timeline scrubber works (dashboard mode)
- [x] WebSocket connection works (live mode)
- [x] Screenshot export works (`P` key)
- [x] JSON export works (button)
- [x] No console errors on load
- [x] Physical lens rendering is **identical** to existing templates

## Next Steps (Future Phases)

### Phase 1.2: Social Lens
- Trust network visualization
- Coalition boundaries
- Theory of Mind arrows
- Social relationship heatmaps

### Phase 1.3: Cognitive Lens
- SRIE cascade visualization
- Metacognition state (System 1/2)
- Calibration curves
- Intention confidence overlays

### Phase 1.4: Cultural Lens
- Cultural group territories
- Variant flow diagrams
- Learning style indicators
- Transmission bias networks

### Phase 1.5: Temporal Lens
- Agent trails
- Metric sparklines
- Emergence event timeline
- Population-wide statistics

## File Count
- **Core**: 8 files
- **Renderers**: 4 files
- **Lenses**: 2 files
- **Panels**: 3 files
- **Utils**: 1 file
- **Main**: 1 file (app.js)
- **Templates**: 2 files
- **Total**: 21 new files

## Zero Visual Regression
The Physical lens **exactly replicates** the existing monolithic templates:
- Agent colors match archetype colors
- Health bars identical (2px height, red fill)
- Action icons same glyphs and position
- Cultural rings same colors and rendering
- Transmission lines same style (dashed, 2-tick decay)
- Grid same color and thickness
- Inspector panel identical layout and content
- All CSS classes preserved for compatibility

## Design Philosophy
1. **Modularity**: Each module has one clear responsibility
2. **Fail-safe**: Canvas layers wrapped in try-catch
3. **Extensibility**: Lens interface enables new views without touching core
4. **Backward compatibility**: New templates coexist with old ones
5. **No frameworks**: Vanilla JS ES modules only
6. **No build step**: Browser-native module loading
