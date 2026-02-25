/**
 * lenses/cognitive-lens.js
 * Cognitive Lens - Internal reasoning view.
 * Shows metacognition, Theory of Mind, SRIE cascade, calibration.
 */

import { LensBase } from './lens-base.js';
import { ARCHETYPE_COLORS, CULTURAL_GROUP_COLORS, interpolate, withAlpha } from '../core/colors.js';
import { GridRenderer } from '../renderers/grid-renderer.js';
import { AgentRenderer } from '../renderers/agent-renderer.js';
import { OverlayRenderer } from '../renderers/overlay-renderer.js';

export class CognitiveLens extends LensBase {
  constructor() {
    super('cognitive');
    this._showDeliberationHeatmap = true;
    this._showToMLines = true;
  }

  getCanvasLayers(renderCtx) {
    return [
      // Layer 1: Grid
      GridRenderer.createGridLayer(),
      // Layer 3: Deliberation heatmap
      this._createDeliberationHeatmapLayer(),
      // Layer 5: ToM prediction lines
      this._createToMLayer(),
      // Layer 6: Agent bodies
      AgentRenderer.createAgentLayer(),
      // Layer 7: Decorations
      AgentRenderer.createDecorationLayer(),
    ];
  }

  /**
   * Create deliberation heatmap layer.
   */
  _createDeliberationHeatmapLayer() {
    return {
      name: 'deliberation-heatmap',
      zIndex: 3,
      draw: (renderCtx) => {
        if (!this._showDeliberationHeatmap) return;

        const { ctx, config, agents } = renderCtx;
        if (!agents) return;

        const heatmapData = agents
          .filter(a => a.alive && a.metacognition?.deliberation_invoked)
          .map(a => ({ x: a.x, y: a.y, value: 1 }));

        if (heatmapData.length === 0) return;

        OverlayRenderer.drawHeatmap(ctx, config, heatmapData, {
          colorLow: '#f59e0b',
          colorHigh: '#ef4444',
          maxValue: 1,
        });
      },
    };
  }

  /**
   * Create Theory of Mind prediction lines layer.
   */
  _createToMLayer() {
    return {
      name: 'tom-lines',
      zIndex: 5,
      draw: (renderCtx) => {
        if (!this._showToMLines) return;

        const { ctx, config, agents, selectedAgentId } = renderCtx;
        if (!selectedAgentId || !agents) return;

        const selectedAgent = agents.find(a => a.id === selectedAgentId);
        if (!selectedAgent || !selectedAgent.alive || !selectedAgent.tom?.models) return;

        const { cellSize, offsetX, offsetY } = config;
        const sx = offsetX + selectedAgent.x * cellSize + cellSize / 2;
        const sy = offsetY + selectedAgent.y * cellSize + cellSize / 2;

        for (const [otherId, model] of Object.entries(selectedAgent.tom.models)) {
          const other = agents.find(a => a.id === otherId);
          if (!other || !other.alive) continue;

          const ox = offsetX + other.x * cellSize + cellSize / 2;
          const oy = offsetY + other.y * cellSize + cellSize / 2;

          // Color by trust: red (low) → yellow (mid) → green (high)
          const trust = model.trust ?? 0.5;
          let color;
          if (trust < 0.3) {
            color = '#ef4444'; // red
          } else if (trust > 0.7) {
            color = '#22c55e'; // green
          } else {
            color = '#eab308'; // yellow
          }

          // Width by prediction accuracy
          const accuracy = model.prediction_accuracy ?? 0.5;
          const width = 1 + accuracy * 2;

          OverlayRenderer.drawArrow(ctx, sx, sy, ox, oy, {
            color,
            width,
            alpha: 0.5,
            headLength: 6,
          });
        }
      },
    };
  }

  getAgentStyle(agent, renderCtx) {
    // Shape by metacognitive strategy
    let shape = 'circle';
    const strategy = agent.metacognition?.active_strategy;
    if (strategy === 'reactive') {
      shape = 'circle';
    } else if (strategy === 'deliberative') {
      shape = 'diamond';
    } else if (strategy === 'adaptive') {
      shape = 'hex';
    }

    // Color by calibration score (red = poor, green = good)
    const calibration = agent.metacognition?.calibration_score;
    let color = '#888888';
    if (calibration !== undefined && calibration !== null) {
      color = interpolate('#ef4444', '#22c55e', calibration);
    }

    // Size
    const size = 0.3;

    // Glow if deliberating
    let glow = undefined;
    if (agent.metacognition?.deliberation_invoked) {
      glow = { color: '#f59e0b', alpha: 0.4, radius: 0.6 };
    }

    // Label for strategy
    let label = undefined;
    if (strategy) {
      label = strategy.charAt(0).toUpperCase();
    }

    return { color, shape, size, glow, label };
  }

  renderSidePanel(container, renderCtx) {
    const { selectedAgentId, agents, temporalBuffer } = renderCtx;

    if (!selectedAgentId) {
      this._renderPopulationStats(container, renderCtx);
      return;
    }

    const agent = agents?.find(a => a.id === selectedAgentId);
    if (!agent) {
      container.innerHTML = '<div class="inspector-empty">Agent not found</div>';
      return;
    }

    const color = agent.color || ARCHETYPE_COLORS[agent.archetype] || '#888';

    let html = `
      <div class="agent-header-info">
        <div class="agent-color-dot" style="background: ${color};"></div>
        <div class="agent-name-archetype">
          <div class="name">${agent.name}</div>
          <div class="archetype">${agent.archetype}</div>
        </div>
      </div>
    `;

    // SRIE Cascade section
    if (agent.srie) {
      html += this._renderSRIESection(agent.srie);
    }

    // Metacognition section
    if (agent.metacognition) {
      html += this._renderMetacognitionSection(agent.metacognition);
    }

    // Theory of Mind section
    if (agent.tom) {
      html += this._renderToMSection(agent.tom, agents);
    }

    // Plan section
    if (agent.plan) {
      html += this._renderPlanSection(agent.plan);
    }

    container.innerHTML = html;
  }

  _renderSRIESection(srie) {
    let html = `
      <div class="srie-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">SRIE Cascade</h3>
    `;

    // Sensation
    if (srie.sensation_summary) {
      const s = srie.sensation_summary;
      html += `
        <div style="margin-bottom: 8px;">
          <div style="color: #06b6d4; font-size: 10px; margin-bottom: 3px;">SENSATION</div>
          <div style="font-size: 11px; color: #ccc;">
            👁 ${s.visible_agent_count ?? 0} agents &nbsp;
            🌾 ${s.visible_resource_tiles ?? 0} tiles &nbsp;
            💬 ${s.message_count ?? 0} msgs
          </div>
          <div style="font-size: 10px; color: #888;">Time: ${s.time_of_day ?? 'unknown'}</div>
        </div>
      `;
    }

    // Reflection
    if (srie.reflection) {
      const r = srie.reflection;
      html += `
        <div style="margin-bottom: 8px;">
          <div style="color: #a78bfa; font-size: 10px; margin-bottom: 3px;">REFLECTION</div>
          <div style="font-size: 11px; color: #ccc;">
            Threat: ${(r.threat_level ?? 0).toFixed(2)} &nbsp;
            Opportunity: ${(r.opportunity_score ?? 0).toFixed(2)}
          </div>
          ${r.need_trends ? `<div style="font-size: 10px; color: #888;">Needs trending: ${Object.entries(r.need_trends).map(([k, v]) => `${k}${v > 0 ? '↑' : v < 0 ? '↓' : '→'}`).join(' ')}</div>` : ''}
        </div>
      `;
    }

    // Intention
    if (srie.intention) {
      const i = srie.intention;
      html += `
        <div style="margin-bottom: 8px;">
          <div style="color: #f59e0b; font-size: 10px; margin-bottom: 3px;">INTENTION</div>
          <div style="font-size: 11px; color: #ccc;">Goal: ${i.primary_goal ?? 'none'}</div>
          ${i.confidence !== undefined ? `
            <div style="margin-top: 3px;">
              <div style="background: #1a1a1a; height: 4px; border-radius: 2px; overflow: hidden;">
                <div style="background: #f59e0b; height: 100%; width: ${(i.confidence * 100).toFixed(0)}%;"></div>
              </div>
            </div>
          ` : ''}
          ${i.target_agent_id ? `<div style="font-size: 10px; color: #888;">Target: ${i.target_agent_id}</div>` : ''}
        </div>
      `;
    }

    html += '</div>';
    return html;
  }

  _renderMetacognitionSection(meta) {
    let html = `
      <div class="metacognition-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Metacognition</h3>
    `;

    // Active strategy badge
    if (meta.active_strategy) {
      const strategyColors = {
        reactive: '#3b82f6',
        deliberative: '#a855f7',
        adaptive: '#22c55e',
      };
      const bgColor = strategyColors[meta.active_strategy] || '#888';
      html += `
        <div style="display: inline-block; padding: 3px 8px; background: ${bgColor}; color: #fff; font-size: 10px; border-radius: 3px; margin-bottom: 8px; text-transform: uppercase;">
          ${meta.active_strategy}
        </div>
      `;
    }

    // Calibration score bar
    if (meta.calibration_score !== undefined) {
      const score = meta.calibration_score;
      const barColor = interpolate('#ef4444', '#22c55e', score);
      html += `
        <div style="margin-bottom: 8px;">
          <div style="font-size: 10px; color: #888; margin-bottom: 3px;">
            Calibration: ${score.toFixed(3)}
          </div>
          <div style="background: #1a1a1a; height: 6px; border-radius: 3px; overflow: hidden;">
            <div style="background: ${barColor}; height: 100%; width: ${(score * 100).toFixed(0)}%;"></div>
          </div>
        </div>
      `;
    }

    // Confidence bias
    if (meta.confidence_bias !== undefined) {
      const bias = meta.confidence_bias;
      const biasLabel = bias > 0.05 ? '↑ Overconfident' : bias < -0.05 ? '↓ Underconfident' : '→ Neutral';
      const biasColor = bias > 0.05 ? '#f59e0b' : bias < -0.05 ? '#06b6d4' : '#888';
      html += `
        <div style="font-size: 10px; color: ${biasColor}; margin-bottom: 6px;">
          Bias: ${biasLabel} (${bias > 0 ? '+' : ''}${bias.toFixed(3)})
        </div>
      `;
    }

    // Other stats
    html += '<div style="font-size: 11px; color: #ccc;">';
    if (meta.deliberation_threshold !== undefined) {
      html += `<div>Delib. Threshold: ${meta.deliberation_threshold.toFixed(2)}</div>`;
    }
    if (meta.total_switches !== undefined) {
      html += `<div>Strategy Switches: ${meta.total_switches}</div>`;
    }
    if (meta.self_awareness_score !== undefined) {
      html += `<div>Self-Awareness: ${meta.self_awareness_score.toFixed(2)}</div>`;
    }
    html += '</div>';

    // Calibration curve sparkline
    if (meta.calibration_curve && meta.calibration_curve.length > 0) {
      html += this._renderCalibrationCurve(meta.calibration_curve);
    }

    html += '</div>';
    return html;
  }

  _renderCalibrationCurve(curve) {
    const width = 200;
    const height = 60;
    const padding = 5;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Build SVG path for data points
    let pathData = '';
    curve.forEach((point, i) => {
      const x = padding + (point.bin_center * chartWidth);
      const y = padding + chartHeight - (point.accuracy * chartHeight);
      if (i === 0) {
        pathData += `M ${x},${y}`;
      } else {
        pathData += ` L ${x},${y}`;
      }
    });

    // Perfect calibration line (diagonal)
    const perfectLine = `M ${padding},${padding + chartHeight} L ${padding + chartWidth},${padding}`;

    return `
      <div style="margin-top: 8px;">
        <div style="font-size: 10px; color: #888; margin-bottom: 3px;">Calibration Curve</div>
        <svg width="${width}" height="${height}" style="background: #0a0a0a; border-radius: 3px;">
          <!-- Perfect calibration reference line -->
          <path d="${perfectLine}" stroke="#444" stroke-width="1" stroke-dasharray="2,2" fill="none"/>
          <!-- Data curve -->
          <path d="${pathData}" stroke="#22c55e" stroke-width="2" fill="none"/>
          <!-- Axes -->
          <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#333" stroke-width="1"/>
          <line x1="${padding}" y1="${padding + chartHeight}" x2="${padding + chartWidth}" y2="${padding + chartHeight}" stroke="#333" stroke-width="1"/>
        </svg>
      </div>
    `;
  }

  _renderToMSection(tom, agents) {
    let html = `
      <div class="tom-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Theory of Mind</h3>
        <div style="font-size: 10px; color: #888; margin-bottom: 6px;">Models: ${tom.model_count ?? 0}</div>
    `;

    if (tom.models && Object.keys(tom.models).length > 0) {
      html += '<div style="max-height: 200px; overflow-y: auto;">';

      for (const [otherId, model] of Object.entries(tom.models)) {
        const other = agents?.find(a => a.id === otherId);
        const otherName = other ? other.name : otherId;
        const trust = model.trust ?? 0.5;
        const threat = model.threat ?? 0;
        const trustColor = interpolate('#ef4444', '#22c55e', trust);

        html += `
          <div style="background: #1a1a1a; padding: 6px; margin-bottom: 6px; border-radius: 3px; border-left: 3px solid ${trustColor};">
            <div style="font-size: 11px; color: #fff; margin-bottom: 3px;">${otherName}</div>
            <div style="font-size: 10px; color: #888; margin-bottom: 3px;">
              Trust: ${trust.toFixed(2)} &nbsp; Threat: ${threat.toFixed(2)}
            </div>
            ${model.estimated_disposition ? `<div style="font-size: 9px; color: #888;">Disposition: ${model.estimated_disposition}</div>` : ''}
            ${model.prediction_accuracy !== undefined ? `<div style="font-size: 9px; color: #888;">Accuracy: ${(model.prediction_accuracy * 100).toFixed(0)}%</div>` : ''}
            ${model.times_helped_me !== undefined || model.times_attacked_me !== undefined ? `
              <div style="font-size: 9px; color: #888;">
                ${model.times_helped_me > 0 ? `<span style="color: #22c55e;">+${model.times_helped_me} helped</span>` : ''}
                ${model.times_attacked_me > 0 ? `<span style="color: #ef4444;">-${model.times_attacked_me} attacked</span>` : ''}
              </div>
            ` : ''}
          </div>
        `;
      }

      html += '</div>';
    } else {
      html += '<div style="color: #666; font-size: 11px;">No models built yet</div>';
    }

    html += '</div>';
    return html;
  }

  _renderPlanSection(plan) {
    let html = `
      <div class="plan-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Plan</h3>
        <div style="font-size: 11px; color: #ccc; margin-bottom: 6px;">${plan.goal ?? 'No goal'}</div>
    `;

    if (plan.steps && plan.current_step !== undefined) {
      const totalSteps = plan.steps.length;
      const currentStep = plan.current_step;
      html += `<div style="font-size: 10px; color: #888;">Step ${currentStep} of ${totalSteps}</div>`;
    }

    if (plan.status) {
      const statusColors = {
        active: '#22c55e',
        pending: '#eab308',
        completed: '#06b6d4',
        failed: '#ef4444',
      };
      const statusColor = statusColors[plan.status] || '#888';
      html += `<div style="font-size: 10px; color: ${statusColor}; margin-top: 3px;">Status: ${plan.status}</div>`;
    }

    if (plan.progress !== undefined) {
      html += `
        <div style="margin-top: 6px;">
          <div style="background: #1a1a1a; height: 4px; border-radius: 2px; overflow: hidden;">
            <div style="background: #22c55e; height: 100%; width: ${(plan.progress * 100).toFixed(0)}%;"></div>
          </div>
        </div>
      `;
    }

    html += '</div>';
    return html;
  }

  _renderPopulationStats(container, renderCtx) {
    const { temporalBuffer } = renderCtx;
    const tickData = temporalBuffer?.getCurrentTick();

    let html = `
      <div class="inspector-empty" style="text-align: left; padding: 16px;">
        <h3 style="font-size: 12px; color: #888; margin-bottom: 12px; text-transform: uppercase;">Population Metacognition</h3>
    `;

    if (tickData?.metacognition) {
      const meta = tickData.metacognition;

      html += `<div style="font-size: 11px; color: #ccc; margin-bottom: 8px;">
        Tracked Agents: ${meta.total_agents_tracked ?? 0}
      </div>`;

      if (meta.avg_calibration_score !== undefined) {
        html += `<div style="font-size: 11px; color: #ccc; margin-bottom: 8px;">
          Avg Calibration: ${meta.avg_calibration_score.toFixed(3)}
        </div>`;
      }

      if (meta.total_switches !== undefined) {
        html += `<div style="font-size: 11px; color: #ccc; margin-bottom: 8px;">
          Total Switches: ${meta.total_switches}
        </div>`;
      }

      // Strategy distribution
      if (meta.strategy_distribution) {
        html += `
          <div style="margin-top: 12px;">
            <h4 style="font-size: 10px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Strategy Distribution</h4>
        `;

        const total = Object.values(meta.strategy_distribution).reduce((sum, count) => sum + count, 0);

        for (const [strategy, count] of Object.entries(meta.strategy_distribution)) {
          const percentage = total > 0 ? (count / total * 100).toFixed(0) : 0;
          const strategyColors = {
            reactive: '#3b82f6',
            deliberative: '#a855f7',
            adaptive: '#22c55e',
          };
          const barColor = strategyColors[strategy] || '#888';

          html += `
            <div style="margin-bottom: 6px;">
              <div style="font-size: 10px; color: #ccc; margin-bottom: 2px;">
                ${strategy}: ${count} (${percentage}%)
              </div>
              <div style="background: #1a1a1a; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: ${barColor}; height: 100%; width: ${percentage}%;"></div>
              </div>
            </div>
          `;
        }

        html += '</div>';
      }
    } else {
      html += '<div style="color: #666; font-size: 11px;">No metacognition data available</div>';
    }

    html += '<div style="margin-top: 16px; font-size: 11px; color: #666;">Click an agent to inspect individual cognition</div>';
    html += '</div>';

    container.innerHTML = html;
  }

  getTimelineMarkers(tick, renderCtx) {
    const { temporalBuffer } = renderCtx;
    if (!temporalBuffer) return [];

    const tickData = temporalBuffer.getTick(tick);
    if (!tickData || !tickData.emergence_events) return [];

    return tickData.emergence_events
      .filter(event => event.type === 'pattern' && event.description?.includes('switch'))
      .map(event => ({
        tick,
        label: 'Strategy Switch',
        color: '#a855f7',
        shape: 'diamond',
      }));
  }

  getKeyBindings() {
    return {
      'd': () => {
        this._showDeliberationHeatmap = !this._showDeliberationHeatmap;
        return `Deliberation heatmap: ${this._showDeliberationHeatmap ? 'ON' : 'OFF'}`;
      },
      'm': () => {
        this._showToMLines = !this._showToMLines;
        return `ToM prediction lines: ${this._showToMLines ? 'ON' : 'OFF'}`;
      },
    };
  }
}
