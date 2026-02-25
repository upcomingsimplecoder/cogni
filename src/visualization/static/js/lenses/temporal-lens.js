/**
 * lenses/temporal-lens.js
 * Temporal analysis view - visualizes time-based patterns, agent trails,
 * metric histories, trend analysis, and action sequences.
 */

import { LensBase } from './lens-base.js';
import { ARCHETYPE_COLORS, NEED_COLORS, withAlpha } from '../core/colors.js';
import { GridRenderer } from '../renderers/grid-renderer.js';
import { AgentRenderer } from '../renderers/agent-renderer.js';

export class TemporalLens extends LensBase {
  constructor() {
    super('temporal');
    this._showTrails = true;
    this._trailsMode = 'all'; // 'all' or 'selected'
  }

  getCanvasLayers(renderCtx) {
    return [
      // Layer 1: Grid
      GridRenderer.createGridLayer(),
      // Layer 2: Resources
      GridRenderer.createResourceLayer(),
      // Layer 4: Agent trails
      this._createTrailLayer(),
      // Layer 5: Metric trend arrows
      this._createTrendArrowLayer(),
      // Layer 6: Agent bodies
      AgentRenderer.createAgentLayer(),
    ];
  }

  /**
   * Create layer for agent position trails.
   */
  _createTrailLayer() {
    return {
      name: 'agent-trails',
      zIndex: 4,
      draw: (renderCtx) => {
        if (!this._showTrails) return;

        const { ctx, config, agents, selectedAgentId, temporalBuffer } = renderCtx;
        const { cellSize, offsetX, offsetY } = config;

        if (!agents || agents.length === 0 || !temporalBuffer) return;

        const agentsToTrail = this._trailsMode === 'selected' && selectedAgentId
          ? agents.filter(a => a.id === selectedAgentId)
          : agents.filter(a => a.alive);

        for (const agent of agentsToTrail) {
          const trail = temporalBuffer.getAgentTrail(agent.id, 20);
          if (trail.length < 2) continue;

          const color = agent.color || ARCHETYPE_COLORS[agent.archetype] || '#888';

          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;

          for (let i = 1; i < trail.length; i++) {
            const prev = trail[i - 1];
            const curr = trail[i];

            const alpha = 0.1 + (i / trail.length) * 0.7;
            ctx.globalAlpha = alpha;

            const x1 = offsetX + prev.x * cellSize + cellSize / 2;
            const y1 = offsetY + prev.y * cellSize + cellSize / 2;
            const x2 = offsetX + curr.x * cellSize + cellSize / 2;
            const y2 = offsetY + curr.y * cellSize + cellSize / 2;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }

          ctx.restore();
        }
      },
    };
  }

  /**
   * Create layer for metric trend arrows on selected agent.
   */
  _createTrendArrowLayer() {
    return {
      name: 'trend-arrows',
      zIndex: 5,
      draw: (renderCtx) => {
        const { ctx, config, agents, selectedAgentId, temporalBuffer } = renderCtx;
        if (!selectedAgentId || !temporalBuffer) return;

        const agent = agents?.find(a => a.id === selectedAgentId);
        if (!agent || !agent.alive) return;

        const { cellSize, offsetX, offsetY } = config;
        const x = offsetX + agent.x * cellSize + cellSize / 2;
        const y = offsetY + agent.y * cellSize + cellSize / 2;

        const metrics = ['health', 'hunger', 'energy'];
        const offsetAngles = [0, Math.PI * 2 / 3, Math.PI * 4 / 3];

        metrics.forEach((metric, idx) => {
          const trend = this._calculateTrend(agent.id, metric, temporalBuffer);
          if (trend === 0) return;

          const angle = offsetAngles[idx];
          const arrowX = x + Math.cos(angle) * cellSize * 0.6;
          const arrowY = y + Math.sin(angle) * cellSize * 0.6;

          const symbol = trend > 0 ? '▲' : '▼';
          const color = trend > 0 ? '#22c55e' : '#ef4444';

          ctx.save();
          ctx.fillStyle = color;
          ctx.font = `${cellSize * 0.4}px monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(symbol, arrowX, arrowY);
          ctx.restore();
        });
      },
    };
  }

  /**
   * Calculate metric trend (positive, negative, or stable).
   */
  _calculateTrend(agentId, metric, temporalBuffer) {
    const series = temporalBuffer.getMetricSeries(agentId, metric, 10);
    if (series.length < 2) return 0;

    const current = series[series.length - 1].value;
    const old = series[0].value;
    const delta = current - old;

    if (Math.abs(delta) < 10) return 0; // Stable
    return delta > 0 ? 1 : -1;
  }

  getAgentStyle(agent, renderCtx) {
    // Color by archetype
    const color = agent.color || ARCHETYPE_COLORS[agent.archetype] || '#888';

    // Shape is always circle for temporal lens
    const shape = 'circle';

    // Size is standard
    const size = 0.3;

    // Glow based on health trend
    let glow = null;
    if (renderCtx.temporalBuffer) {
      const healthTrend = this._calculateHealthTrend(agent.id, renderCtx.temporalBuffer);
      if (healthTrend < 0) {
        glow = { color: '#ef4444', alpha: 0.3, radius: 0.5 };
      } else if (healthTrend > 0) {
        glow = { color: '#22c55e', alpha: 0.2, radius: 0.5 };
      }
    }

    return { color, shape, size, glow };
  }

  /**
   * Calculate health trend over last 10 ticks.
   */
  _calculateHealthTrend(agentId, temporalBuffer) {
    const series = temporalBuffer.getMetricSeries(agentId, 'health', 10);
    if (series.length < 2) return 0;

    const current = series[series.length - 1].value;
    const old = series[0].value;
    const delta = current - old;

    if (Math.abs(delta) < 10) return 0;
    return delta > 0 ? 1 : -1;
  }

  renderSidePanel(container, renderCtx) {
    const { selectedAgentId, agents } = renderCtx;

    if (!selectedAgentId) {
      this._renderOverview(container, renderCtx);
      return;
    }

    const agent = agents?.find(a => a.id === selectedAgentId);
    if (!agent) {
      container.innerHTML = '<div class="inspector-empty">Agent not found</div>';
      return;
    }

    this._renderAgentInspector(container, agent, renderCtx);
  }

  _renderAgentInspector(container, agent, renderCtx) {
    const color = agent.color || ARCHETYPE_COLORS[agent.archetype] || '#888';
    const { temporalBuffer } = renderCtx;

    // Metric sparklines
    const sparklinesHtml = temporalBuffer
      ? this._renderSparklines(agent.id, temporalBuffer)
      : '<div style="color: #666; font-size: 11px;">No temporal data</div>';

    // Movement trail stats
    const trailHtml = temporalBuffer
      ? this._renderMovementTrail(agent.id, temporalBuffer)
      : '<div style="color: #666; font-size: 11px;">No movement data</div>';

    // Action history
    const actionHtml = temporalBuffer
      ? this._renderActionHistory(agent.id, temporalBuffer)
      : '<div style="color: #666; font-size: 11px;">No action data</div>';

    container.innerHTML = `
      <div class="agent-header-info">
        <div class="agent-color-dot" style="background: ${color};"></div>
        <div class="agent-name-archetype">
          <div class="name">${agent.name}</div>
          <div class="archetype">${agent.archetype}</div>
        </div>
      </div>

      <div class="sparklines-section">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Metric Sparklines</h3>
        ${sparklinesHtml}
      </div>

      <div class="movement-trail-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Movement Trail</h3>
        ${trailHtml}
      </div>

      <div class="action-history-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Action History</h3>
        ${actionHtml}
      </div>
    `;
  }

  _renderSparklines(agentId, temporalBuffer) {
    const metrics = ['health', 'hunger', 'thirst', 'energy'];

    return metrics.map(metric => {
      const series = temporalBuffer.getMetricSeries(agentId, metric, 50);
      if (series.length === 0) {
        return `<div style="margin-bottom: 8px; color: #666; font-size: 11px;">${metric}: No data</div>`;
      }

      const current = series[series.length - 1].value;
      const values = series.map(s => s.value);
      const min = Math.min(...values);
      const max = Math.max(...values);

      // Calculate trend
      const trend = series.length > 1 && values[values.length - 1] > values[0] ? '▲' :
                    series.length > 1 && values[values.length - 1] < values[0] ? '▼' : '─';
      const trendColor = trend === '▲' ? '#22c55e' : trend === '▼' ? '#ef4444' : '#888';

      // Generate SVG sparkline
      const svgWidth = 200;
      const svgHeight = 40;
      const padding = 2;
      const range = max - min || 1;

      const points = series.map((point, i) => {
        const x = padding + (i / (series.length - 1 || 1)) * (svgWidth - padding * 2);
        const y = svgHeight - padding - ((point.value - min) / range) * (svgHeight - padding * 2);
        return `${x},${y}`;
      }).join(' ');

      const metricColor = NEED_COLORS[metric] || '#888';

      return `
        <div style="margin-bottom: 12px;">
          <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
            <span style="color: #888; text-transform: capitalize;">${metric}</span>
            <span style="color: #fff;">${current.toFixed(1)}</span>
            <span style="color: ${trendColor};">${trend}</span>
          </div>
          <svg width="${svgWidth}" height="${svgHeight}" style="display: block; background: #1a1a1a; border-radius: 2px;">
            <polyline
              points="${points}"
              fill="none"
              stroke="${metricColor}"
              stroke-width="1.5"
            />
          </svg>
          <div style="display: flex; justify-content: space-between; font-size: 9px; color: #666; margin-top: 2px;">
            <span>Min: ${min.toFixed(1)}</span>
            <span>Max: ${max.toFixed(1)}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  _renderMovementTrail(agentId, temporalBuffer) {
    const trail = temporalBuffer.getAgentTrail(agentId, 50);
    if (trail.length < 2) {
      return '<div style="color: #666; font-size: 11px;">Insufficient data</div>';
    }

    // Calculate total distance
    let totalDistance = 0;
    for (let i = 1; i < trail.length; i++) {
      const dx = trail[i].x - trail[i - 1].x;
      const dy = trail[i].y - trail[i - 1].y;
      totalDistance += Math.sqrt(dx * dx + dy * dy);
    }

    // Calculate average speed (distance per tick)
    const avgSpeed = totalDistance / (trail.length - 1);

    // Current position
    const currentPos = trail[trail.length - 1];

    // Direction tendency (which quadrant they move toward most)
    let northCount = 0, southCount = 0, eastCount = 0, westCount = 0;
    for (let i = 1; i < trail.length; i++) {
      const dx = trail[i].x - trail[i - 1].x;
      const dy = trail[i].y - trail[i - 1].y;
      if (dy < 0) northCount++;
      if (dy > 0) southCount++;
      if (dx > 0) eastCount++;
      if (dx < 0) westCount++;
    }

    const directions = [];
    if (northCount > southCount && northCount > trail.length * 0.3) directions.push('North');
    if (southCount > northCount && southCount > trail.length * 0.3) directions.push('South');
    if (eastCount > westCount && eastCount > trail.length * 0.3) directions.push('East');
    if (westCount > eastCount && westCount > trail.length * 0.3) directions.push('West');

    const directionTendency = directions.length > 0 ? directions.join('-') : 'Stationary';

    return `
      <div style="display: flex; flex-direction: column; gap: 4px; font-size: 12px;">
        <div style="display: flex; justify-content: space-between;">
          <span style="color: #888;">Total Distance:</span>
          <span style="color: #fff;">${totalDistance.toFixed(2)} cells</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: #888;">Avg Speed:</span>
          <span style="color: #fff;">${avgSpeed.toFixed(2)} cells/tick</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: #888;">Current Position:</span>
          <span style="color: #fff;">(${currentPos.x}, ${currentPos.y})</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: #888;">Direction Tendency:</span>
          <span style="color: #fff;">${directionTendency}</span>
        </div>
      </div>
    `;
  }

  _renderActionHistory(agentId, temporalBuffer) {
    const allTicks = temporalBuffer.getAllTicks();
    const recentTicks = allTicks.slice(-10);

    const actions = [];
    for (const tickData of recentTicks) {
      const agent = tickData.agents?.find(a => a.id === agentId);
      if (agent && agent.action_type) {
        actions.push({
          tick: tickData.tick,
          type: agent.action_type,
          success: agent.action_success,
        });
      }
    }

    if (actions.length === 0) {
      return '<div style="color: #666; font-size: 11px;">No recent actions</div>';
    }

    // Calculate success rate
    const successCount = actions.filter(a => a.success).length;
    const successRate = (successCount / actions.length) * 100;

    const actionIcons = {
      gather: '⛏',
      move: '→',
      give: '↗',
      attack: '✕',
      rest: '…',
      talk: '○',
      eat: '🍞',
      drink: '💧',
    };

    const actionsHtml = actions.reverse().map(action => {
      const icon = actionIcons[action.type.toLowerCase()] || '?';
      const statusColor = action.success ? '#22c55e' : '#ef4444';
      const statusIcon = action.success ? '✓' : '✗';

      return `
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px; background: #1a1a1a; border-radius: 2px; margin-bottom: 4px; font-size: 11px;">
          <span style="color: #888;">Tick ${action.tick}</span>
          <span style="color: #fff;">${icon} ${action.type}</span>
          <span style="color: ${statusColor}; font-weight: bold;">${statusIcon}</span>
        </div>
      `;
    }).join('');

    return `
      <div style="margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
          <span style="color: #888;">Success Rate:</span>
          <span style="color: #fff;">${successRate.toFixed(0)}%</span>
        </div>
        <div style="background: #2a2a2a; height: 12px; border-radius: 2px; overflow: hidden; margin-bottom: 8px;">
          <div style="background: #22c55e; height: 12px; width: ${successRate}%;"></div>
        </div>
      </div>
      ${actionsHtml}
    `;
  }

  _renderOverview(container, renderCtx) {
    const { agents, temporalBuffer } = renderCtx;
    const tickData = temporalBuffer?.getCurrentTick();

    if (!tickData) {
      container.innerHTML = '<div class="inspector-empty">Click an agent to inspect temporal data</div>';
      return;
    }

    // Population stats
    const livingCount = agents?.filter(a => a.alive).length || 0;
    const deadCount = agents?.filter(a => !a.alive).length || 0;

    // Average metrics
    const livingAgents = agents?.filter(a => a.alive) || [];
    const avgHealth = livingAgents.length > 0
      ? livingAgents.reduce((sum, a) => sum + (a.health || 0), 0) / livingAgents.length
      : 0;
    const avgHunger = livingAgents.length > 0
      ? livingAgents.reduce((sum, a) => sum + (a.hunger || 0), 0) / livingAgents.length
      : 0;
    const avgThirst = livingAgents.length > 0
      ? livingAgents.reduce((sum, a) => sum + (a.thirst || 0), 0) / livingAgents.length
      : 0;
    const avgEnergy = livingAgents.length > 0
      ? livingAgents.reduce((sum, a) => sum + (a.energy || 0), 0) / livingAgents.length
      : 0;

    // Day/time
    const day = tickData.day || 0;
    const timeOfDay = tickData.time_of_day || 'unknown';

    // Recent emergence events
    const emergenceHtml = this._renderEmergenceEvents(temporalBuffer);

    container.innerHTML = `
      <div class="population-trends-section">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Population Trends</h3>
        <div style="display: flex; flex-direction: column; gap: 4px; font-size: 12px;">
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Living Agents:</span>
            <span style="color: #22c55e;">${livingCount}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Dead Agents:</span>
            <span style="color: #ef4444;">${deadCount}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Day:</span>
            <span style="color: #fff;">${day}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Time of Day:</span>
            <span style="color: #fff;">${timeOfDay}</span>
          </div>
        </div>
      </div>

      <div class="average-metrics-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Average Metrics</h3>
        <div style="display: flex; flex-direction: column; gap: 6px;">
          ${this._renderMetricBar('Health', avgHealth, NEED_COLORS.health)}
          ${this._renderMetricBar('Hunger', avgHunger, NEED_COLORS.hunger)}
          ${this._renderMetricBar('Thirst', avgThirst, NEED_COLORS.thirst)}
          ${this._renderMetricBar('Energy', avgEnergy, NEED_COLORS.energy)}
        </div>
      </div>

      <div class="emergence-events-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Recent Events</h3>
        ${emergenceHtml}
      </div>
    `;
  }

  _renderMetricBar(label, value, color) {
    return `
      <div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
          <span style="color: #888;">${label}</span>
          <span style="color: #fff;">${value.toFixed(1)}</span>
        </div>
        <div style="background: #2a2a2a; height: 12px; border-radius: 2px; overflow: hidden;">
          <div style="background: ${color}; height: 12px; width: ${value}%;"></div>
        </div>
      </div>
    `;
  }

  _renderEmergenceEvents(temporalBuffer) {
    const allTicks = temporalBuffer.getAllTicks();
    const events = [];

    for (let i = allTicks.length - 1; i >= 0 && events.length < 5; i--) {
      const tickData = allTicks[i];
      if (!tickData.emergence) continue;

      for (const event of tickData.emergence) {
        events.push({
          tick: tickData.tick,
          text: event.text || event.description || 'Unknown event',
        });
        if (events.length >= 5) break;
      }
    }

    if (events.length === 0) {
      return '<div style="color: #666; font-size: 11px;">No recent events</div>';
    }

    return events.map(event => `
      <div style="padding: 4px; background: #1a1a1a; border-radius: 2px; margin-bottom: 4px; font-size: 11px;">
        <div style="color: #888; font-size: 10px; margin-bottom: 2px;">Tick ${event.tick}</div>
        <div style="color: #fff;">${event.text}</div>
      </div>
    `).join('');
  }

  getTimelineMarkers(tick, renderCtx) {
    const markers = [];
    const allTicks = renderCtx.temporalBuffer?.getAllTicks() || [];

    for (const tickData of allTicks) {
      if (!tickData.emergence) continue;

      for (const event of tickData.emergence) {
        const text = event.text || event.description || '';
        if (text.toLowerCase().includes('died') || text.toLowerCase().includes('death')) {
          markers.push({
            tick: tickData.tick,
            label: 'Death',
            color: '#ef4444',
            shape: 'cross',
          });
        }
      }
    }

    return markers;
  }

  getKeyBindings() {
    return {
      t: () => {
        this._showTrails = !this._showTrails;
        return `Agent trails ${this._showTrails ? 'shown' : 'hidden'}`;
      },
      a: () => {
        this._trailsMode = this._trailsMode === 'all' ? 'selected' : 'all';
        return `Trail mode: ${this._trailsMode}`;
      },
    };
  }
}
