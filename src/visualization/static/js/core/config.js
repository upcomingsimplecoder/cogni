/**
 * core/config.js
 * Central configuration for visualization system.
 */

export const CONFIG = {
  // Canvas rendering
  cellSize: 8,
  agentRadius: 0.3,  // as fraction of cellSize
  healthBarHeight: 2,
  selectionRingWidth: 2,

  // Temporal buffer
  maxTickHistory: 500,
  trailLength: 20,

  // Spatial index
  spatialCellSize: 8,

  // Animation
  animationFPS: 60,
  autoplaySpeed: 100,  // ms per tick

  // Colors
  backgroundColor: '#0a0a0a',
  gridColor: '#1a1a1a',
  selectionColor: '#06b6d4',

  // UI
  sidebarWidth: 400,
  headerHeight: 60,
  controlsHeight: 50,
};

export const KEYBOARD_BINDINGS = {
  // Lens switching
  '1': 'switchLens:physical',
  '2': 'switchLens:social',
  '3': 'switchLens:cognitive',
  '4': 'switchLens:cultural',
  '5': 'switchLens:temporal',

  // Controls
  'Space': 'togglePause',
  'ArrowLeft': 'previousTick',
  'ArrowRight': 'nextTick',
  'p': 'screenshot',
  'e': 'exportJSON',
  'g': 'toggleGrid',
  'h': 'toggleHeatmap',
  't': 'toggleTrails',
  'Escape': 'clearSelection',
};

export const FEATURE_TOGGLES = {
  grid: true,
  healthBars: true,
  trails: false,
  heatmap: false,
  connectionLines: true,
  emergenceLabels: true,
};
