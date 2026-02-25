/**
 * core/temporal-buffer.js
 * Ring buffer storing last N ticks of agent data.
 */

export class TemporalBuffer {
  /**
   * @param {number} maxSize - Maximum ticks to store
   */
  constructor(maxSize = 500) {
    this.maxSize = maxSize;
    this._buffer = [];
    this._tickToIndex = new Map();
  }

  /**
   * Add a tick of agent data.
   * @param {Object} tickData - Data for this tick {tick, agents: [...]}
   */
  addTick(tickData) {
    const tick = tickData.tick;

    // Remove oldest if at capacity
    if (this._buffer.length >= this.maxSize) {
      const oldest = this._buffer.shift();
      this._tickToIndex.delete(oldest.tick);
    }

    this._buffer.push(tickData);
    this._tickToIndex.set(tick, this._buffer.length - 1);
  }

  /**
   * Get agent's position trail for last N ticks.
   * @param {string} agentId - Agent ID
   * @param {number} n - Number of ticks to retrieve
   * @returns {Array<{x: number, y: number, tick: number}>}
   */
  getAgentTrail(agentId, n = 20) {
    const trail = [];
    const startIndex = Math.max(0, this._buffer.length - n);

    for (let i = startIndex; i < this._buffer.length; i++) {
      const tickData = this._buffer[i];
      const agent = tickData.agents?.find(a => a.id === agentId);
      if (agent) {
        trail.push({
          x: agent.x,
          y: agent.y,
          tick: tickData.tick,
        });
      }
    }

    return trail;
  }

  /**
   * Get tick data at offset from current (negative = past).
   * @param {number} tickOffset - Offset from current tick (0 = current, -1 = previous, etc.)
   * @returns {Object|null}
   */
  getTickData(tickOffset = 0) {
    const index = this._buffer.length - 1 + tickOffset;
    if (index < 0 || index >= this._buffer.length) return null;
    return this._buffer[index];
  }

  /**
   * Get current tick data.
   * @returns {Object|null}
   */
  getCurrentTick() {
    return this._buffer.length > 0 ? this._buffer[this._buffer.length - 1] : null;
  }

  /**
   * Get metric series for an agent over last N ticks.
   * @param {string} agentId - Agent ID
   * @param {string} metric - Metric name (e.g., 'health', 'hunger')
   * @param {number} n - Number of ticks
   * @returns {Array<{tick: number, value: number}>}
   */
  getMetricSeries(agentId, metric, n = 50) {
    const series = [];
    const startIndex = Math.max(0, this._buffer.length - n);

    for (let i = startIndex; i < this._buffer.length; i++) {
      const tickData = this._buffer[i];
      const agent = tickData.agents?.find(a => a.id === agentId);
      if (agent && agent[metric] !== undefined) {
        series.push({
          tick: tickData.tick,
          value: agent[metric],
        });
      }
    }

    return series;
  }

  /**
   * Get all ticks in the buffer.
   * @returns {Array}
   */
  getAllTicks() {
    return [...this._buffer];
  }

  /**
   * Clear the buffer.
   */
  clear() {
    this._buffer = [];
    this._tickToIndex.clear();
  }

  /**
   * Get buffer size.
   * @returns {number}
   */
  get size() {
    return this._buffer.length;
  }
}
