/**
 * lenses/physical-lens.js
 * Physical world view - replicates existing visualization behavior.
 * Shows agents, resources, communication lines, action glyphs.
 */

import { LensBase } from './lens-base.js';
import { ARCHETYPE_COLORS, getCulturalGroupColor, BIAS_COLORS } from '../core/colors.js';
import { GridRenderer } from '../renderers/grid-renderer.js';
import { AgentRenderer } from '../renderers/agent-renderer.js';
import { OverlayRenderer } from '../renderers/overlay-renderer.js';

export class PhysicalLens extends LensBase {
  constructor() {
    super('physical');
    this._transmissionLines = []; // {observerId, actorId, biasType, remainingTicks}
  }

  getCanvasLayers(renderCtx) {
    return [
      // Layer 0: Background (handled by compositor)
      // Layer 1: Grid
      GridRenderer.createGridLayer(),
      // Layer 2: Resources
      GridRenderer.createResourceLayer(),
      // Layer 5: Transmission lines
      this._createTransmissionLayer(),
      // Layer 6: Agent bodies
      AgentRenderer.createAgentLayer(),
      // Layer 7: Agent decorations
      AgentRenderer.createDecorationLayer(),
    ];
  }

  /**
   * Create layer for cultural transmission lines.
   */
  _createTransmissionLayer() {
    return {
      name: 'transmissions',
      zIndex: 5,
      draw: (renderCtx) => {
        if (this._transmissionLines.length === 0) return;

        const { ctx, config, agents } = renderCtx;
        const { cellSize, offsetX, offsetY } = config;

        for (const tx of this._transmissionLines) {
          const observer = agents.find(a => a.id === tx.observerId);
          const actor = agents.find(a => a.id === tx.actorId);

          if (!observer || !actor || !observer.alive || !actor.alive) continue;

          const ox = offsetX + observer.x * cellSize + cellSize / 2;
          const oy = offsetY + observer.y * cellSize + cellSize / 2;
          const ax = offsetX + actor.x * cellSize + cellSize / 2;
          const ay = offsetY + actor.y * cellSize + cellSize / 2;

          const alpha = tx.remainingTicks / 2;
          const lineColor = BIAS_COLORS[tx.biasType] || '#888';

          OverlayRenderer.drawArrow(ctx, ax, ay, ox, oy, {
            color: lineColor,
            width: 2,
            alpha: alpha * 0.6,
            headLength: 6,
          });

          // Dashed line
          ctx.save();
          ctx.globalAlpha = alpha * 0.6;
          ctx.strokeStyle = lineColor;
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(ox, oy);
          ctx.stroke();
          ctx.restore();
        }

        // Decay transmission lines
        this._transmissionLines = this._transmissionLines
          .map(t => ({ ...t, remainingTicks: t.remainingTicks - 1 }))
          .filter(t => t.remainingTicks > 0);
      },
    };
  }

  getAgentStyle(agent, renderCtx) {
    // Color by archetype
    const color = agent.color || ARCHETYPE_COLORS[agent.archetype] || '#888';

    // Shape is always circle for physical lens
    const shape = 'circle';

    // Size is standard
    const size = 0.3;

    // Cultural group ring (handled in rendering, not glow)
    const style = { color, shape, size };

    return style;
  }

  renderSidePanel(container, renderCtx) {
    const { selectedAgentId, agents } = renderCtx;

    if (!selectedAgentId) {
      container.innerHTML = '<div class="inspector-empty">Click an agent on the canvas to inspect</div>';
      return;
    }

    const agent = agents?.find(a => a.id === selectedAgentId);
    if (!agent) {
      container.innerHTML = '<div class="inspector-empty">Agent not found</div>';
      return;
    }

    const color = agent.color || ARCHETYPE_COLORS[agent.archetype] || '#888';

    // Build traits display
    const traitsHtml = agent.traits
      ? Object.entries(agent.traits).map(([key, value]) => {
          const shortKey = key.replace('_tendency', '').replace('_', '').substring(0, 6);
          return `<span class="trait-item"><span class="trait-key">${shortKey}</span>:${typeof value === 'number' ? value.toFixed(2) : value}</span>`;
        }).join('')
      : '<span style="color: #666;">No traits data</span>';

    // Build inventory display
    const inventoryHtml = agent.inventory && Object.keys(agent.inventory).length > 0
      ? Object.entries(agent.inventory).map(([item, count]) =>
          `<div class="inventory-item">${item}<span class="count">×${count}</span></div>`
        ).join('')
      : '<div style="color: #666; font-size: 11px;">Empty</div>';

    container.innerHTML = `
      <div class="agent-header-info">
        <div class="agent-color-dot" style="background: ${color};"></div>
        <div class="agent-name-archetype">
          <div class="name">${agent.name}</div>
          <div class="archetype">${agent.archetype}</div>
        </div>
      </div>

      <div class="needs-bars">
        <div class="need-bar">
          <div class="need-bar-label">
            <span class="label">Hunger</span>
            <span class="value">${agent.hunger?.toFixed(1) || '?'}</span>
          </div>
          <div class="need-bar-bg">
            <div class="need-bar-fill hunger" style="width: ${agent.hunger || 0}%"></div>
          </div>
        </div>
        <div class="need-bar">
          <div class="need-bar-label">
            <span class="label">Thirst</span>
            <span class="value">${agent.thirst?.toFixed(1) || '?'}</span>
          </div>
          <div class="need-bar-bg">
            <div class="need-bar-fill thirst" style="width: ${agent.thirst || 0}%"></div>
          </div>
        </div>
        <div class="need-bar">
          <div class="need-bar-label">
            <span class="label">Energy</span>
            <span class="value">${agent.energy?.toFixed(1) || '?'}</span>
          </div>
          <div class="need-bar-bg">
            <div class="need-bar-fill energy" style="width: ${agent.energy || 0}%"></div>
          </div>
        </div>
        <div class="need-bar">
          <div class="need-bar-label">
            <span class="label">Health</span>
            <span class="value">${agent.health?.toFixed(1) || '?'}</span>
          </div>
          <div class="need-bar-bg">
            <div class="need-bar-fill health" style="width: ${agent.health || 0}%"></div>
          </div>
        </div>
      </div>

      <div class="traits">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Traits</h3>
        <div class="traits-content">${traitsHtml}</div>
      </div>

      ${agent.intention ? `
      <div class="intention-section">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Current Intention</h3>
        <div class="intention-content">
          ${agent.intention}
          ${agent.confidence !== undefined ? `<div class="confidence">Confidence: ${(agent.confidence * 100).toFixed(0)}%</div>` : ''}
        </div>
      </div>
      ` : ''}

      ${agent.monologue ? `
      <div class="monologue-section">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Internal Monologue</h3>
        <div class="monologue-content">${agent.monologue}</div>
      </div>
      ` : ''}

      <div class="inventory-section">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Inventory</h3>
        <div class="inventory-items">${inventoryHtml}</div>
      </div>

      ${agent.action_type ? `
      <div class="action-section">
        <span style="color: #888;">Last Action:</span>
        <span class="action-type">${agent.action_type}</span>
        <span class="${agent.action_success ? 'action-success' : 'action-fail'}">${agent.action_success ? '✓' : '✗'}</span>
      </div>
      ` : ''}
    `;
  }

  onActivate(renderCtx) {
    // Handle transmission events from tick data
    const tickData = renderCtx.temporalBuffer?.getCurrentTick();
    if (tickData && tickData.cultural && tickData.cultural.transmission_events) {
      for (const event of tickData.cultural.transmission_events) {
        if (!event.adopted || !event.actor) continue;
        this._transmissionLines.push({
          observerId: event.observer,
          actorId: event.actor,
          biasType: event.bias,
          remainingTicks: 2,
        });
      }
    }
  }

  /**
   * Update transmission lines when new tick arrives.
   */
  updateTransmissions(transmissionEvents) {
    if (!transmissionEvents) return;

    for (const event of transmissionEvents) {
      if (!event.adopted || !event.actor) continue;
      this._transmissionLines.push({
        observerId: event.observer,
        actorId: event.actor,
        biasType: event.bias,
        remainingTicks: 2,
      });
    }
  }
}
