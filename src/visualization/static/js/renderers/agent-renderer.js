/**
 * renderers/agent-renderer.js
 * Renders agent bodies, health bars, and selection highlights.
 */

import { NEED_COLORS } from '../core/colors.js';

export class AgentRenderer {
  /**
   * Create layer for agent bodies.
   * @returns {{name: string, zIndex: number, draw: Function}}
   */
  static createAgentLayer() {
    return {
      name: 'agents',
      zIndex: 6,
      draw: (renderCtx) => {
        const { ctx, config, agents, activeLens, selectedAgentId } = renderCtx;

        if (!agents || agents.length === 0) return;

        const { cellSize, offsetX, offsetY } = config;

        for (const agent of agents) {
          if (!agent.alive) continue;

          const x = offsetX + agent.x * cellSize + cellSize / 2;
          const y = offsetY + agent.y * cellSize + cellSize / 2;

          // Get style from active lens
          const style = activeLens.getAgentStyle(agent, renderCtx);

          // Selection highlight
          if (agent.id === selectedAgentId) {
            ctx.strokeStyle = config.selectionColor || '#06b6d4';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, cellSize / 2 + 2, 0, Math.PI * 2);
            ctx.stroke();
          }

          // Agent glow (if specified)
          if (style.glow) {
            ctx.save();
            ctx.globalAlpha = style.glow.alpha || 0.3;
            ctx.fillStyle = style.glow.color || style.color;
            ctx.beginPath();
            ctx.arc(x, y, cellSize * (style.glow.radius || 0.6), 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
          }

          // Agent body
          ctx.fillStyle = style.color;
          AgentRenderer._drawShape(ctx, x, y, cellSize * (style.size || 0.3), style.shape || 'circle');

          // Label (if specified)
          if (style.label) {
            ctx.fillStyle = '#fff';
            ctx.font = `${cellSize * 0.4}px monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(style.label, x, y - cellSize * 0.8);
          }
        }
      },
    };
  }

  /**
   * Create layer for agent decorations (health bars, belief bubbles, action icons).
   * @returns {{name: string, zIndex: number, draw: Function}}
   */
  static createDecorationLayer() {
    return {
      name: 'decorations',
      zIndex: 7,
      draw: (renderCtx) => {
        const { ctx, config, agents, toggles } = renderCtx;

        if (!agents || agents.length === 0) return;

        const { cellSize, offsetX, offsetY } = config;

        for (const agent of agents) {
          if (!agent.alive) continue;

          const x = offsetX + agent.x * cellSize + cellSize / 2;
          const y = offsetY + agent.y * cellSize + cellSize / 2;

          // Health bar
          if (toggles.healthBars && agent.health !== undefined) {
            AgentRenderer._drawHealthBar(ctx, x, y, cellSize, agent.health);
          }

          // Action icon
          if (agent.action_type) {
            AgentRenderer._drawActionIcon(ctx, x, y, cellSize, agent.action_type);
          }
        }
      },
    };
  }

  /**
   * Draw agent shape.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x - Center X
   * @param {number} y - Center Y
   * @param {number} radius - Shape radius
   * @param {string} shape - Shape type
   */
  static _drawShape(ctx, x, y, radius, shape) {
    ctx.beginPath();

    switch (shape) {
      case 'circle':
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        break;

      case 'diamond':
        ctx.moveTo(x, y - radius);
        ctx.lineTo(x + radius, y);
        ctx.lineTo(x, y + radius);
        ctx.lineTo(x - radius, y);
        ctx.closePath();
        break;

      case 'triangle':
        ctx.moveTo(x, y - radius);
        ctx.lineTo(x + radius * 0.866, y + radius * 0.5);
        ctx.lineTo(x - radius * 0.866, y + radius * 0.5);
        ctx.closePath();
        break;

      case 'square':
        const half = radius * 0.866;
        ctx.rect(x - half, y - half, half * 2, half * 2);
        break;

      case 'hex':
        for (let i = 0; i < 6; i++) {
          const angle = (Math.PI / 3) * i;
          const px = x + radius * Math.cos(angle);
          const py = y + radius * Math.sin(angle);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        break;

      default:
        ctx.arc(x, y, radius, 0, Math.PI * 2);
    }

    ctx.fill();
  }

  /**
   * Draw health bar below agent.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x - Center X
   * @param {number} y - Center Y
   * @param {number} cellSize
   * @param {number} health - Health percentage (0-100)
   */
  static _drawHealthBar(ctx, x, y, cellSize, health) {
    const barWidth = cellSize;
    const barHeight = 2;
    const barX = x - barWidth / 2;
    const barY = y + cellSize / 2 + 2;

    // Background
    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(barX, barY, barWidth, barHeight);

    // Health fill
    ctx.fillStyle = NEED_COLORS.health || '#ef4444';
    ctx.fillRect(barX, barY, barWidth * (health / 100), barHeight);
  }

  /**
   * Draw action icon above agent.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x - Center X
   * @param {number} y - Center Y
   * @param {number} cellSize
   * @param {string} actionType
   */
  static _drawActionIcon(ctx, x, y, cellSize, actionType) {
    const icons = {
      gather: '⛏',
      move: '→',
      give: '↗',
      attack: '✕',
      rest: '…',
      talk: '○',
      eat: '🍞',
      drink: '💧',
    };

    const icon = icons[actionType.toLowerCase()];
    if (!icon) return;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = `${cellSize * 0.5}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(icon, x, y - cellSize * 0.4);
  }
}
