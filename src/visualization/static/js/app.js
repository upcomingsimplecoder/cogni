/**
 * app.js
 * Main orchestrator for the AUTOCOG visualization system.
 */

import { CONFIG, KEYBOARD_BINDINGS, FEATURE_TOGGLES } from './core/config.js';
import { EventBus } from './core/event-bus.js';
import { TemporalBuffer } from './core/temporal-buffer.js';
import { SpatialIndex } from './core/spatial-index.js';
import { NetworkGraph } from './core/network-graph.js';
import { TimeSeriesStore } from './core/time-series-store.js';
import { DataSource } from './core/data-source.js';
import { CanvasCompositor } from './renderers/canvas-compositor.js';
import { GridRenderer } from './renderers/grid-renderer.js';
import { AgentRenderer } from './renderers/agent-renderer.js';
import { PhysicalLens } from './lenses/physical-lens.js';
import { SocialLens } from './lenses/social-lens.js';
import { CognitiveLens } from './lenses/cognitive-lens.js';
import { CulturalLens } from './lenses/cultural-lens.js';
import { TemporalLens } from './lenses/temporal-lens.js';
import { SidePanelManager } from './panels/side-panel-manager.js';
import { TimelinePanel } from './panels/timeline-panel.js';
import { screenshot, exportJSON } from './utils/export.js';
import { HelpOverlay, GuidedTour } from './panels/help-overlay.js';

export class AutocogApp {
  constructor() {
    this.mode = null;
    this.dataSource = null;
    this.eventBus = new EventBus();
    this.temporalBuffer = new TemporalBuffer(CONFIG.maxTickHistory);
    this.spatialIndex = null;
    this.networkGraph = new NetworkGraph();
    this.timeSeriesStore = new TimeSeriesStore();

    this.canvas = null;
    this.compositor = null;
    this.sidePanelManager = null;
    this.timelinePanel = null;

    this.currentTick = 0;
    this.selectedAgentId = null;
    this.isPaused = false;
    this.activeLens = null;
    this.lenses = new Map();
    this.toggles = { ...FEATURE_TOGGLES };

    this.worldConfig = null;
    this._animationFrame = null;
    this._lensTransitionAlpha = 1; // For crossfade (1 = fully visible)
    this._helpVisible = false;
  }

  /**
   * Initialize the application.
   */
  async init() {
    console.log('Initializing AUTOCOG visualization...');

    // Auto-detect mode
    this.dataSource = DataSource.autoDetect();
    this.mode = this.dataSource.mode;
    console.log(`Mode: ${this.mode}`);

    // Initialize data source
    await this.dataSource.initialize();

    // Get world configuration
    if (this.mode === 'dashboard') {
      const metadata = this.dataSource.getMetadata();
      this.worldConfig = {
        worldWidth: metadata.world_width,
        worldHeight: metadata.world_height,
        cellSize: CONFIG.cellSize,
        offsetX: 0,
        offsetY: 0,
        backgroundColor: CONFIG.backgroundColor,
        gridColor: CONFIG.gridColor,
        selectionColor: CONFIG.selectionColor,
      };
    } else {
      // Live mode: fetch config from server
      const config = await fetch('/api/config').then(r => r.json());
      this.worldConfig = {
        worldWidth: config.world_width,
        worldHeight: config.world_height,
        cellSize: CONFIG.cellSize,
        offsetX: 0,
        offsetY: 0,
        backgroundColor: CONFIG.backgroundColor,
        gridColor: CONFIG.gridColor,
        selectionColor: CONFIG.selectionColor,
      };
    }

    // Initialize spatial index
    this.spatialIndex = new SpatialIndex(
      this.worldConfig.worldWidth,
      this.worldConfig.worldHeight,
      CONFIG.spatialCellSize
    );

    // Setup DOM elements
    this._setupDOM();

    // Initialize lenses
    this._initializeLenses();

    // Set active lens to Physical (default)
    this.switchLens('physical');

    // Setup event listeners
    this._setupEventListeners();

    // Start data flow
    if (this.mode === 'live') {
      this.dataSource.onTick(data => this._handleTickData(data));
      this.dataSource.onConnection(connected => this._handleConnection(connected));
    } else {
      // Dashboard mode: load first tick
      this._loadTick(0);
    }

    // Start render loop
    this._startRenderLoop();

    // Show guided tour for first-time visitors
    GuidedTour.maybeShow(document.body);

    console.log('Initialization complete');
  }

  /**
   * Setup DOM elements.
   */
  _setupDOM() {
    // Canvas
    this.canvas = document.getElementById('world-canvas');
    if (!this.canvas) {
      throw new Error('Canvas element not found');
    }

    // Setup canvas size
    this.canvas.width = this.worldConfig.worldWidth * this.worldConfig.cellSize;
    this.canvas.height = this.worldConfig.worldHeight * this.worldConfig.cellSize;

    this.compositor = new CanvasCompositor(this.canvas);

    // Side panel
    const sidebarContainer = document.querySelector('.sidebar');
    if (sidebarContainer) {
      this.sidePanelManager = new SidePanelManager(sidebarContainer);
    }

    // Timeline
    const timelineContainer = document.querySelector('.control-group');
    if (timelineContainer) {
      this.timelinePanel = new TimelinePanel(timelineContainer, this.dataSource);
      this.timelinePanel.onScrub(tick => this._loadTick(tick));
    }

    // Tick display
    this.tickDisplay = document.getElementById('tick-display');

    // Create lens toolbar
    this._createLensToolbar();
  }

  /**
   * Create lens switcher toolbar.
   */
  _createLensToolbar() {
    const header = document.querySelector('header');
    if (!header) return;

    const toolbar = document.createElement('div');
    toolbar.className = 'lens-toolbar';
    toolbar.style.cssText = 'display: flex; gap: 8px; align-items: center;';

    const lensNames = ['physical', 'social', 'cognitive', 'cultural', 'temporal'];
    const lensLabels = ['Physical', 'Social', 'Cognitive', 'Cultural', 'Temporal'];

    for (let i = 0; i < lensNames.length; i++) {
      const btn = document.createElement('button');
      btn.textContent = `${i + 1}. ${lensLabels[i]}`;
      btn.className = 'lens-btn';
      btn.dataset.lens = lensNames[i];
      btn.style.cssText = 'padding: 6px 12px; font-size: 12px;';

      btn.addEventListener('click', () => this.switchLens(lensNames[i]));
      toolbar.appendChild(btn);
    }

    header.appendChild(toolbar);
  }

  /**
   * Initialize all lenses.
   */
  _initializeLenses() {
    this.lenses.set('physical', new PhysicalLens());
    this.lenses.set('social', new SocialLens());
    this.lenses.set('cognitive', new CognitiveLens());
    this.lenses.set('cultural', new CulturalLens());
    this.lenses.set('temporal', new TemporalLens());
  }

  /**
   * Switch to a different lens.
   * @param {string} lensName
   */
  switchLens(lensName) {
    const newLens = this.lenses.get(lensName);
    if (!newLens) {
      console.warn(`Lens "${lensName}" not found`);
      return;
    }

    // Skip if already on this lens
    if (this.activeLens === newLens) return;

    // Deactivate current lens
    if (this.activeLens) {
      this.activeLens.onDeactivate(this._getRenderContext());
    }

    // Crossfade transition (~200ms)
    this._lensTransitionAlpha = 0;
    const fadeStart = performance.now();
    const fadeDuration = 200;
    const fadeIn = () => {
      const elapsed = performance.now() - fadeStart;
      this._lensTransitionAlpha = Math.min(elapsed / fadeDuration, 1);
      if (this._lensTransitionAlpha < 1) {
        requestAnimationFrame(fadeIn);
      }
    };
    requestAnimationFrame(fadeIn);

    // Activate new lens
    this.activeLens = newLens;
    this.activeLens.onActivate(this._getRenderContext());

    // Update compositor layers
    const layers = this.activeLens.getCanvasLayers(this._getRenderContext());
    this.compositor.setLayers(layers);

    // Update UI
    document.querySelectorAll('.lens-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lens === lensName);
    });

    // Re-render
    this._render();

    console.log(`Switched to ${lensName} lens`);
  }

  /**
   * Handle incoming tick data (live mode).
   * @param {Object} data
   */
  _handleTickData(data) {
    if (this.isPaused) return;

    this.currentTick = data.tick;
    this.temporalBuffer.addTick(data);

    // Update spatial index
    if (data.agents) {
      this.spatialIndex.update(data.agents);

      // Record metrics for time series
      for (const agent of data.agents) {
        if (agent.alive) {
          this.timeSeriesStore.record(agent.id, {
            tick: data.tick,
            health: agent.health,
            hunger: agent.hunger,
            thirst: agent.thirst,
            energy: agent.energy,
          });
        }
      }
    }

    // Update network graph (relationships, cultural, coalitions)
    this._updateNetworkGraph(data);

    // Notify lens of new transmission events
    if (this.activeLens?.updateTransmissions && data.cultural?.transmission_events) {
      this.activeLens.updateTransmissions(data.cultural.transmission_events);
    }

    // Update UI
    if (this.tickDisplay) {
      this.tickDisplay.textContent = data.tick;
    }

    if (this.timelinePanel) {
      this.timelinePanel.update(data.tick, this._getRenderContext());
    }

    // Emit tick event
    this.eventBus.emit('tick', data);
  }

  /**
   * Handle connection status change (live mode).
   * @param {boolean} connected
   */
  _handleConnection(connected) {
    const statusIndicator = document.getElementById('ws-status');
    const statusText = document.getElementById('ws-status-text');

    if (statusIndicator) {
      statusIndicator.classList.toggle('connected', connected);
    }
    if (statusText) {
      statusText.textContent = connected ? 'Connected' : 'Disconnected';
    }
  }

  /**
   * Load tick data (dashboard mode).
   * @param {number} tick
   */
  _loadTick(tick) {
    const data = this.dataSource.getTickData(tick);
    if (!data) return;

    this.currentTick = tick;
    this.temporalBuffer.addTick(data);

    // Update spatial index
    if (data.agents) {
      this.spatialIndex.update(data.agents);
    }

    // Update UI
    if (this.tickDisplay) {
      this.tickDisplay.textContent = tick;
    }

    if (this.timelinePanel) {
      this.timelinePanel.update(tick, this._getRenderContext());
    }

    this._render();
  }

  /**
   * Update network graph with relationship/cultural/coalition data.
   * @param {Object} data
   */
  _updateNetworkGraph(data) {
    if (!data.agents) return;

    // Clear and rebuild
    this.networkGraph.clear();

    for (const agent of data.agents) {
      if (!agent.alive) continue;

      // Social relationships
      if (agent.social_relationships) {
        for (const [otherId, rel] of Object.entries(agent.social_relationships)) {
          this.networkGraph.setEdge(agent.id, otherId, 'trust', rel.trust);
        }
      }

      // Cultural group (use as edge type)
      if (agent.cultural?.cultural_group >= 0) {
        this.networkGraph.setEdge(agent.id, agent.id, 'cultural_group', agent.cultural.cultural_group);
      }

      // Coalition membership
      if (agent.coalition?.coalition_id) {
        this.networkGraph.setEdge(agent.id, agent.coalition.coalition_id, 'coalition', 1);
      }
    }
  }

  /**
   * Setup event listeners.
   */
  _setupEventListeners() {
    // Canvas click for agent selection
    this.canvas.addEventListener('click', (e) => this._handleCanvasClick(e));

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => this._handleKeyDown(e));

    // Window resize
    window.addEventListener('resize', () => this._handleResize());

    // Pause button
    const pauseBtn = document.getElementById('pause-btn');
    if (pauseBtn) {
      pauseBtn.addEventListener('click', () => this.togglePause());
    }
  }

  /**
   * Handle canvas click to select agent.
   * @param {MouseEvent} e
   */
  _handleCanvasClick(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellSize = this.worldConfig.cellSize;
    const gridX = Math.floor((x - this.worldConfig.offsetX) / cellSize);
    const gridY = Math.floor((y - this.worldConfig.offsetY) / cellSize);

    // Find nearest agent within 2 cells
    const tickData = this.temporalBuffer.getCurrentTick();
    if (!tickData || !tickData.agents) return;

    let nearest = null;
    let minDist = 2;

    for (const agent of tickData.agents) {
      if (!agent.alive) continue;
      const dist = Math.sqrt(Math.pow(agent.x - gridX, 2) + Math.pow(agent.y - gridY, 2));
      if (dist < minDist) {
        minDist = dist;
        nearest = agent;
      }
    }

    if (nearest) {
      this.selectedAgentId = nearest.id;
      this.eventBus.emit('agentSelected', nearest.id);
      this._render();
    }
  }

  /**
   * Handle keyboard shortcuts.
   * @param {KeyboardEvent} e
   */
  _handleKeyDown(e) {
    // Help overlay toggle (? key — works regardless of lens)
    if (e.key === '?') {
      e.preventDefault();
      this._helpVisible = !this._helpVisible;
      HelpOverlay.toggle(document.body, this._helpVisible);
      return;
    }

    // Check lens-specific key bindings first (they override globals)
    if (this.activeLens) {
      const lensBindings = this.activeLens.getKeyBindings();
      if (lensBindings[e.key]) {
        e.preventDefault();
        lensBindings[e.key]();
        this._render();
        return;
      }
    }

    const action = KEYBOARD_BINDINGS[e.key];
    if (!action) return;

    e.preventDefault();

    if (action.startsWith('switchLens:')) {
      const lens = action.split(':')[1];
      this.switchLens(lens);
    } else if (action === 'togglePause') {
      this.togglePause();
    } else if (action === 'screenshot') {
      screenshot(this.canvas);
    } else if (action === 'exportJSON') {
      exportJSON(this._getRenderContext());
    } else if (action === 'toggleGrid') {
      this.toggles.grid = !this.toggles.grid;
      this._render();
    } else if (action === 'toggleHeatmap') {
      this.toggles.heatmap = !this.toggles.heatmap;
      this._render();
    } else if (action === 'toggleTrails') {
      this.toggles.trails = !this.toggles.trails;
      this._render();
    } else if (action === 'clearSelection') {
      this.selectedAgentId = null;
      this._render();
    }
  }

  /**
   * Handle window resize.
   */
  _handleResize() {
    // For now, canvas size is fixed - could be made responsive
  }

  /**
   * Toggle pause state.
   */
  togglePause() {
    this.isPaused = !this.isPaused;
    const pauseBtn = document.getElementById('pause-btn');
    if (pauseBtn) {
      pauseBtn.textContent = this.isPaused ? 'Resume' : 'Pause';
      pauseBtn.classList.toggle('active', this.isPaused);
    }
  }

  /**
   * Start render loop.
   */
  _startRenderLoop() {
    const animate = () => {
      this._render();
      this._animationFrame = requestAnimationFrame(animate);
    };
    animate();
  }

  /**
   * Render current state.
   */
  _render() {
    const renderCtx = this._getRenderContext();

    // Apply crossfade alpha during lens transitions
    if (this._lensTransitionAlpha < 1) {
      renderCtx.globalAlpha = this._lensTransitionAlpha;
    }

    // Render canvas
    this.compositor.render(renderCtx);

    // Render side panel
    if (this.sidePanelManager) {
      this.sidePanelManager.render(renderCtx);
    }
  }

  /**
   * Build render context object.
   * @returns {Object}
   */
  _getRenderContext() {
    const tickData = this.temporalBuffer.getCurrentTick();

    return {
      ctx: this.canvas.getContext('2d'),
      canvas: this.canvas,
      config: this.worldConfig,
      mode: this.mode,
      currentTick: this.currentTick,
      activeLens: this.activeLens,
      agents: tickData?.agents || [],
      selectedAgentId: this.selectedAgentId,
      temporalBuffer: this.temporalBuffer,
      spatialIndex: this.spatialIndex,
      networkGraph: this.networkGraph,
      timeSeriesStore: this.timeSeriesStore,
      emergenceEvents: tickData?.emergent_events || [],
      toggles: this.toggles,
    };
  }

  /**
   * Cleanup and shutdown.
   */
  destroy() {
    if (this._animationFrame) {
      cancelAnimationFrame(this._animationFrame);
    }

    if (this.dataSource) {
      this.dataSource.close();
    }

    this.eventBus.clear();
  }
}

// Auto-start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.autocogApp = new AutocogApp();
    window.autocogApp.init().catch(err => console.error('Init failed:', err));
  });
} else {
  window.autocogApp = new AutocogApp();
  window.autocogApp.init().catch(err => console.error('Init failed:', err));
}
