/**
 * core/data-source.js
 * Unified data interface - handles both WebSocket (live) and embedded JSON (dashboard).
 */

export class DataSource {
  /**
   * @param {string} mode - 'live' or 'dashboard'
   * @param {Object} [embeddedData] - For dashboard mode
   */
  constructor(mode, embeddedData = null) {
    this.mode = mode;
    this._embeddedData = embeddedData;
    this._currentTick = 0;
    this._ws = null;
    this._tickCallbacks = [];
    this._connectionCallbacks = [];
  }

  /**
   * Auto-detect mode based on presence of embedded data.
   * @returns {DataSource}
   */
  static autoDetect() {
    if (window.__DASHBOARD_DATA__) {
      return new DataSource('dashboard', window.__DASHBOARD_DATA__);
    }
    return new DataSource('live');
  }

  /**
   * Initialize data source (connect WebSocket if live mode).
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.mode === 'live') {
      await this._connectWebSocket();
    } else {
      // Dashboard mode: data is already embedded
      this._currentTick = 0;
    }
  }

  /**
   * Connect to WebSocket server (live mode only).
   * @returns {Promise<void>}
   */
  _connectWebSocket() {
    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws`;

      this._ws = new WebSocket(wsUrl);

      this._ws.onopen = () => {
        console.log('WebSocket connected');
        this._notifyConnection(true);
        resolve();
      };

      this._ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this._currentTick = data.tick;
          this._notifyTick(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this._ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this._notifyConnection(false);
        reject(error);
      };

      this._ws.onclose = () => {
        console.log('WebSocket disconnected');
        this._notifyConnection(false);
        // Auto-reconnect after 2s
        setTimeout(() => this._connectWebSocket(), 2000);
      };
    });
  }

  /**
   * Register callback for tick updates.
   * @param {Function} callback - Called with tick data
   */
  onTick(callback) {
    this._tickCallbacks.push(callback);
  }

  /**
   * Register callback for connection status changes (live mode only).
   * @param {Function} callback - Called with boolean (connected/disconnected)
   */
  onConnection(callback) {
    this._connectionCallbacks.push(callback);
  }

  /**
   * Get current tick number.
   * @returns {number}
   */
  getCurrentTick() {
    return this._currentTick;
  }

  /**
   * Get tick data at specific index (dashboard mode only).
   * @param {number} tickIndex - Tick index
   * @returns {Object|null}
   */
  getTickData(tickIndex) {
    if (this.mode !== 'dashboard' || !this._embeddedData) {
      return null;
    }

    const tick = this._embeddedData.ticks[tickIndex];
    if (!tick) return null;

    // Reconstruct full tick data from compressed format
    return {
      tick: tickIndex,
      agents: tick.map(snapshot => this._expandSnapshot(snapshot)),
      emergent_events: this._getEventsAtTick(tickIndex),
      cultural: this._getCulturalAtTick(tickIndex),
    };
  }

  /**
   * Get total tick count (dashboard mode only).
   * @returns {number}
   */
  getTotalTicks() {
    if (this.mode !== 'dashboard' || !this._embeddedData) {
      return 0;
    }
    return this._embeddedData.metadata.actual_ticks;
  }

  /**
   * Get metadata (dashboard mode only).
   * @returns {Object|null}
   */
  getMetadata() {
    if (this.mode !== 'dashboard' || !this._embeddedData) {
      return null;
    }
    return this._embeddedData.metadata;
  }

  /**
   * Expand compressed agent snapshot to full format.
   * @param {Object} snapshot - Compressed snapshot
   * @returns {Object} Full agent data
   */
  _expandSnapshot(snapshot) {
    const agent = this._embeddedData.agents.find(a => a.id === snapshot.id);

    return {
      id: snapshot.id,
      name: agent?.name || snapshot.id,
      archetype: agent?.archetype || 'unknown',
      color: agent?.color || '#888',
      x: snapshot.pos[0],
      y: snapshot.pos[1],
      alive: snapshot.alive,
      hunger: snapshot.hunger,
      thirst: snapshot.thirst,
      energy: snapshot.energy,
      health: snapshot.health,
      action_type: snapshot.action,
      action_success: snapshot.success,
      intention: snapshot.intention,
      monologue: snapshot.monologue,
      inventory: snapshot.inventory,
      cultural: snapshot.cultural,
    };
  }

  /**
   * Get emergence events at a specific tick.
   * @param {number} tickIndex
   * @returns {Array<string>}
   */
  _getEventsAtTick(tickIndex) {
    if (!this._embeddedData.emergence_events) return [];
    return this._embeddedData.emergence_events
      .filter(e => e.tick === tickIndex)
      .map(e => e.description);
  }

  /**
   * Get cultural data at a specific tick.
   * @param {number} tickIndex
   * @returns {Object|null}
   */
  _getCulturalAtTick(tickIndex) {
    if (!this._embeddedData.cultural_summary) return null;
    return this._embeddedData.cultural_summary[tickIndex] || null;
  }

  /**
   * Notify tick callbacks.
   * @param {Object} tickData
   */
  _notifyTick(tickData) {
    for (const callback of this._tickCallbacks) {
      try {
        callback(tickData);
      } catch (error) {
        console.error('Error in tick callback:', error);
      }
    }
  }

  /**
   * Notify connection callbacks.
   * @param {boolean} connected
   */
  _notifyConnection(connected) {
    for (const callback of this._connectionCallbacks) {
      try {
        callback(connected);
      } catch (error) {
        console.error('Error in connection callback:', error);
      }
    }
  }

  /**
   * Close data source (cleanup).
   */
  close() {
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
  }
}
