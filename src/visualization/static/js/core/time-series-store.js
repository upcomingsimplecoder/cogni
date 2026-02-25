/**
 * core/time-series-store.js
 * Per-agent time series storage for sparklines and metrics.
 */

export class TimeSeriesStore {
  /**
   * @param {number} maxLength - Maximum series length per agent
   */
  constructor(maxLength = 200) {
    this.maxLength = maxLength;
    this._series = new Map(); // agentId -> {metric -> [{tick, value}]}
  }

  /**
   * Record metrics for an agent at a tick.
   * @param {string} agentId - Agent ID
   * @param {Object} metrics - Map of metric names to values
   */
  record(agentId, metrics) {
    if (!this._series.has(agentId)) {
      this._series.set(agentId, new Map());
    }

    const agentSeries = this._series.get(agentId);

    for (const [metric, value] of Object.entries(metrics)) {
      if (!agentSeries.has(metric)) {
        agentSeries.set(metric, []);
      }

      const series = agentSeries.get(metric);
      series.push({ tick: metrics.tick || 0, value });

      // Trim if exceeds max length
      if (series.length > this.maxLength) {
        series.shift();
      }
    }
  }

  /**
   * Get time series for a specific agent and metric.
   * @param {string} agentId - Agent ID
   * @param {string} metric - Metric name
   * @param {number} [n] - Optional: last N points
   * @returns {Array<{tick: number, value: number}>}
   */
  getSeries(agentId, metric, n = null) {
    const agentSeries = this._series.get(agentId);
    if (!agentSeries || !agentSeries.has(metric)) {
      return [];
    }

    const series = agentSeries.get(metric);
    if (n === null) {
      return [...series];
    }

    return series.slice(-n);
  }

  /**
   * Get population-wide series (aggregated across all agents).
   * @param {string} metric - Metric name
   * @param {number} [n] - Optional: last N ticks
   * @returns {Array<{tick: number, mean: number, min: number, max: number}>}
   */
  getPopulationSeries(metric, n = null) {
    const tickMap = new Map(); // tick -> [values]

    for (const agentSeries of this._series.values()) {
      if (!agentSeries.has(metric)) continue;

      const series = agentSeries.get(metric);
      for (const point of series) {
        if (!tickMap.has(point.tick)) {
          tickMap.set(point.tick, []);
        }
        tickMap.get(point.tick).push(point.value);
      }
    }

    const result = [];
    for (const [tick, values] of tickMap) {
      if (values.length === 0) continue;

      const sum = values.reduce((a, b) => a + b, 0);
      result.push({
        tick,
        mean: sum / values.length,
        min: Math.min(...values),
        max: Math.max(...values),
      });
    }

    // Sort by tick
    result.sort((a, b) => a.tick - b.tick);

    if (n !== null) {
      return result.slice(-n);
    }

    return result;
  }

  /**
   * Clear all series data.
   */
  clear() {
    this._series.clear();
  }

  /**
   * Clear data for a specific agent.
   * @param {string} agentId - Agent ID
   */
  clearAgent(agentId) {
    this._series.delete(agentId);
  }
}
