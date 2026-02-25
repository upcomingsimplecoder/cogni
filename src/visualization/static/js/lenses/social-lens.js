/**
 * lenses/social-lens.js
 * Social Lens - Relationships and coalitions view.
 * Shows trust networks, coalition territories, social dynamics.
 */

import { LensBase } from './lens-base.js';
import { ARCHETYPE_COLORS, CULTURAL_GROUP_COLORS, getCulturalGroupColor, interpolate, withAlpha } from '../core/colors.js';
import { GridRenderer } from '../renderers/grid-renderer.js';
import { AgentRenderer } from '../renderers/agent-renderer.js';
import { OverlayRenderer } from '../renderers/overlay-renderer.js';

export class SocialLens extends LensBase {
  constructor() {
    super('social');
    this._showAllTrustLines = false;
    this._showCoalitionHulls = true;
  }

  getCanvasLayers(renderCtx) {
    return [
      // Layer 1: Grid
      GridRenderer.createGridLayer(),
      // Layer 4: Coalition territory hulls
      this._createCoalitionHullsLayer(),
      // Layer 5: Trust lines
      this._createTrustLinesLayer(),
      // Layer 6: Agent bodies
      AgentRenderer.createAgentLayer(),
      // Layer 7: Decorations
      AgentRenderer.createDecorationLayer(),
    ];
  }

  /**
   * Create coalition territory hulls layer.
   */
  _createCoalitionHullsLayer() {
    return {
      name: 'coalition-hulls',
      zIndex: 4,
      draw: (renderCtx) => {
        if (!this._showCoalitionHulls) return;

        const { ctx, config, agents, temporalBuffer } = renderCtx;
        const tickData = temporalBuffer?.getCurrentTick();

        if (!tickData?.coalitions?.coalitions || !agents) return;

        const { cellSize, offsetX, offsetY } = config;

        for (let i = 0; i < tickData.coalitions.coalitions.length; i++) {
          const coalition = tickData.coalitions.coalitions[i];
          if (!coalition.members || coalition.members.length < 3) continue;

          // Get canvas positions of all coalition members
          const memberPoints = [];
          for (const memberId of coalition.members) {
            const member = agents.find(a => a.id === memberId && a.alive);
            if (member) {
              const x = offsetX + member.x * cellSize + cellSize / 2;
              const y = offsetY + member.y * cellSize + cellSize / 2;
              memberPoints.push({ x, y });
            }
          }

          if (memberPoints.length < 3) continue;

          // Get coalition color
          const coalitionColor = CULTURAL_GROUP_COLORS[i % CULTURAL_GROUP_COLORS.length];

          // Draw filled hull (background)
          ctx.save();
          ctx.globalAlpha = 0.05;
          ctx.fillStyle = coalitionColor;
          const hull = OverlayRenderer._computeConvexHull(memberPoints);
          if (hull.length >= 3) {
            ctx.beginPath();
            ctx.moveTo(hull[0].x, hull[0].y);
            for (let j = 1; j < hull.length; j++) {
              ctx.lineTo(hull[j].x, hull[j].y);
            }
            ctx.closePath();
            ctx.fill();
          }
          ctx.restore();

          // Draw hull outline
          OverlayRenderer.drawConvexHull(ctx, memberPoints, {
            color: coalitionColor,
            alpha: 0.15,
            lineWidth: 2,
          });
        }
      },
    };
  }

  /**
   * Create trust lines layer.
   */
  _createTrustLinesLayer() {
    return {
      name: 'trust-lines',
      zIndex: 5,
      draw: (renderCtx) => {
        const { ctx, config, agents, selectedAgentId } = renderCtx;
        if (!agents) return;

        const { cellSize, offsetX, offsetY } = config;

        // Determine which agents to show trust lines for
        let sourceAgents = [];
        if (this._showAllTrustLines) {
          sourceAgents = agents.filter(a => a.alive && a.social_relationships);
        } else if (selectedAgentId) {
          const selectedAgent = agents.find(a => a.id === selectedAgentId);
          if (selectedAgent?.alive && selectedAgent.social_relationships) {
            sourceAgents = [selectedAgent];
          }
        }

        for (const agent of sourceAgents) {
          const ax = offsetX + agent.x * cellSize + cellSize / 2;
          const ay = offsetY + agent.y * cellSize + cellSize / 2;

          for (const [otherId, rel] of Object.entries(agent.social_relationships)) {
            const other = agents.find(a => a.id === otherId);
            if (!other || !other.alive) continue;

            const ox = offsetX + other.x * cellSize + cellSize / 2;
            const oy = offsetY + other.y * cellSize + cellSize / 2;

            const trust = rel.trust ?? 0.5;

            // Color by trust level
            let color;
            if (trust > 0.5) {
              color = '#22c55e'; // green
            } else if (trust < 0.3) {
              color = '#ef4444'; // red
            } else {
              color = '#eab308'; // yellow
            }

            // Width by trust magnitude
            const width = 1 + trust * 2;

            OverlayRenderer.drawLine(ctx, ax, ay, ox, oy, {
              color,
              width,
              alpha: 0.6,
            });
          }
        }
      },
    };
  }

  getAgentStyle(agent, renderCtx) {
    const { temporalBuffer } = renderCtx;
    const tickData = temporalBuffer?.getCurrentTick();

    // Find agent's coalition (if any)
    let coalitionIndex = -1;
    let coalitionRole = null;

    if (agent.coalition?.coalition_id && tickData?.coalitions?.coalitions) {
      const coalition = tickData.coalitions.coalitions.find(c => c.id === agent.coalition.coalition_id);
      if (coalition) {
        coalitionIndex = tickData.coalitions.coalitions.indexOf(coalition);
        coalitionRole = agent.coalition.role;
      }
    }

    // Shape by coalition role
    let shape = 'triangle'; // no coalition
    if (coalitionRole === 'leader') {
      shape = 'diamond';
    } else if (coalitionRole === 'member' || coalitionIndex >= 0) {
      shape = 'circle';
    }

    // Color by coalition or archetype
    let color = '#888888';
    if (coalitionIndex >= 0) {
      color = CULTURAL_GROUP_COLORS[coalitionIndex % CULTURAL_GROUP_COLORS.length];
    } else {
      color = agent.color || ARCHETYPE_COLORS[agent.archetype] || '#888';
    }

    // Size by role
    let size = 0.25; // no coalition
    if (coalitionRole === 'leader') {
      size = 0.4;
    } else if (coalitionRole === 'member' || coalitionIndex >= 0) {
      size = 0.3;
    }

    // Glow for leaders
    let glow = undefined;
    if (coalitionRole === 'leader') {
      glow = { color, alpha: 0.3, radius: 0.7 };
    }

    // Label for leaders
    let label = undefined;
    if (coalitionRole === 'leader') {
      label = '★';
    }

    return { color, shape, size, glow, label };
  }

  renderSidePanel(container, renderCtx) {
    const { selectedAgentId, agents, temporalBuffer } = renderCtx;

    if (!selectedAgentId) {
      this._renderSocialOverview(container, renderCtx);
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

    // Coalition section
    if (agent.coalition) {
      html += this._renderCoalitionSection(agent.coalition, agents, renderCtx);
    }

    // Relationships section
    if (agent.social_relationships) {
      html += this._renderRelationshipsSection(agent.social_relationships, agents);
    }

    // ToM Models section (compact)
    if (agent.tom?.models) {
      html += this._renderCompactToMSection(agent.tom.models, agents);
    }

    container.innerHTML = html;
  }

  _renderCoalitionSection(coalition, agents, renderCtx) {
    const { temporalBuffer } = renderCtx;
    const tickData = temporalBuffer?.getCurrentTick();

    let html = `
      <div class="coalition-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Coalition</h3>
    `;

    // Find full coalition data
    let fullCoalition = null;
    if (tickData?.coalitions?.coalitions) {
      fullCoalition = tickData.coalitions.coalitions.find(c => c.id === coalition.coalition_id);
    }

    const coalitionName = fullCoalition?.name || coalition.coalition_id || 'Unknown';
    const coalitionGoal = fullCoalition?.goal || coalition.goal || 'No goal';
    const cohesion = fullCoalition?.cohesion ?? coalition.cohesion;

    // Role badge
    const roleColors = {
      leader: '#f59e0b',
      member: '#888',
    };
    const roleColor = roleColors[coalition.role] || '#888';
    html += `
      <div style="display: inline-block; padding: 3px 8px; background: ${roleColor}; color: #fff; font-size: 10px; border-radius: 3px; margin-bottom: 8px; text-transform: uppercase;">
        ${coalition.role || 'member'}
      </div>
    `;

    html += `
      <div style="font-size: 11px; color: #ccc; margin-bottom: 6px;">
        ${coalitionName}
      </div>
      <div style="font-size: 10px; color: #888; margin-bottom: 6px;">
        Goal: ${coalitionGoal}
      </div>
    `;

    // Cohesion bar
    if (cohesion !== undefined) {
      html += `
        <div style="margin-bottom: 8px;">
          <div style="font-size: 10px; color: #888; margin-bottom: 3px;">
            Cohesion: ${cohesion.toFixed(2)}
          </div>
          <div style="background: #1a1a1a; height: 6px; border-radius: 3px; overflow: hidden;">
            <div style="background: #22c55e; height: 100%; width: ${(cohesion * 100).toFixed(0)}%;"></div>
          </div>
        </div>
      `;
    }

    // Members list
    const members = fullCoalition?.members || coalition.members || [];
    if (members.length > 0) {
      html += `
        <div style="margin-top: 8px;">
          <div style="font-size: 10px; color: #888; margin-bottom: 4px;">Members (${members.length}):</div>
          <div style="max-height: 120px; overflow-y: auto;">
      `;

      for (const memberId of members) {
        const member = agents?.find(a => a.id === memberId);
        const memberName = member ? member.name : memberId;

        // Show trust between selected agent and this member (if exists)
        let trustInfo = '';
        const selectedAgent = agents?.find(a => a.id === renderCtx.selectedAgentId);
        if (selectedAgent?.social_relationships?.[memberId]) {
          const trust = selectedAgent.social_relationships[memberId].trust ?? 0.5;
          const trustColor = interpolate('#ef4444', '#22c55e', trust);
          trustInfo = `
            <div style="margin-top: 2px; background: #0a0a0a; height: 3px; border-radius: 2px; overflow: hidden;">
              <div style="background: ${trustColor}; height: 100%; width: ${(trust * 100).toFixed(0)}%;"></div>
            </div>
          `;
        }

        html += `
          <div style="background: #1a1a1a; padding: 4px 6px; margin-bottom: 4px; border-radius: 2px;">
            <div style="font-size: 10px; color: #ccc;">${memberName}</div>
            ${trustInfo}
          </div>
        `;
      }

      html += '</div></div>';
    }

    html += '</div>';
    return html;
  }

  _renderRelationshipsSection(relationships, agents) {
    let html = `
      <div class="relationships-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Relationships</h3>
    `;

    // Sort by trust descending
    const sortedRels = Object.entries(relationships).sort((a, b) => {
      const trustA = a[1].trust ?? 0;
      const trustB = b[1].trust ?? 0;
      return trustB - trustA;
    });

    if (sortedRels.length === 0) {
      html += '<div style="color: #666; font-size: 11px;">No relationships yet</div>';
    } else {
      html += '<div style="max-height: 200px; overflow-y: auto;">';

      for (const [otherId, rel] of sortedRels) {
        const other = agents?.find(a => a.id === otherId);
        const otherName = other ? other.name : otherId;
        const trust = rel.trust ?? 0.5;
        const trustColor = interpolate('#ef4444', '#22c55e', trust);

        // Interaction flags
        const flags = [];
        if (rel.was_helped_by) flags.push('<span style="color: #22c55e;">+helped</span>');
        if (rel.was_attacked_by) flags.push('<span style="color: #ef4444;">-attacked</span>');

        html += `
          <div style="background: #1a1a1a; padding: 6px; margin-bottom: 6px; border-radius: 3px; border-left: 3px solid ${trustColor};">
            <div style="font-size: 11px; color: #fff; margin-bottom: 3px;">${otherName}</div>
            <div style="font-size: 10px; color: #888;">
              Trust: ${trust.toFixed(2)} &nbsp; Interactions: ${rel.interaction_count ?? 0}
            </div>
            ${rel.net_resources_given !== undefined ? `
              <div style="font-size: 9px; color: ${rel.net_resources_given > 0 ? '#22c55e' : rel.net_resources_given < 0 ? '#ef4444' : '#888'};">
                Resources: ${rel.net_resources_given > 0 ? '+' : ''}${rel.net_resources_given}
              </div>
            ` : ''}
            ${rel.last_interaction_tick !== undefined ? `
              <div style="font-size: 9px; color: #666;">Last: tick ${rel.last_interaction_tick}</div>
            ` : ''}
            ${flags.length > 0 ? `<div style="font-size: 9px; margin-top: 2px;">${flags.join(' ')}</div>` : ''}
          </div>
        `;
      }

      html += '</div>';
    }

    html += '</div>';
    return html;
  }

  _renderCompactToMSection(models, agents) {
    let html = `
      <div class="tom-compact-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Theory of Mind (Compact)</h3>
    `;

    const modelEntries = Object.entries(models);
    if (modelEntries.length === 0) {
      html += '<div style="color: #666; font-size: 11px;">No models</div>';
    } else {
      html += '<div style="max-height: 120px; overflow-y: auto;">';

      for (const [otherId, model] of modelEntries) {
        const other = agents?.find(a => a.id === otherId);
        const otherName = other ? other.name : otherId;
        const trust = model.trust ?? 0.5;
        const threat = model.threat ?? 0;

        html += `
          <div style="background: #0a0a0a; padding: 4px 6px; margin-bottom: 4px; border-radius: 2px; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 10px; color: #ccc; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${otherName}</div>
            <div style="display: flex; gap: 6px; align-items: center;">
              <div style="font-size: 9px; color: #22c55e;">T:${trust.toFixed(1)}</div>
              <div style="font-size: 9px; color: #ef4444;">⚠:${threat.toFixed(1)}</div>
            </div>
          </div>
        `;
      }

      html += '</div>';
    }

    html += '</div>';
    return html;
  }

  _renderSocialOverview(container, renderCtx) {
    const { temporalBuffer, agents } = renderCtx;
    const tickData = temporalBuffer?.getCurrentTick();

    let html = `
      <div class="inspector-empty" style="text-align: left; padding: 16px;">
        <h3 style="font-size: 12px; color: #888; margin-bottom: 12px; text-transform: uppercase;">Social Network</h3>
    `;

    // Coalition overview
    if (tickData?.coalitions) {
      const coalitions = tickData.coalitions;
      html += `
        <div style="margin-bottom: 16px;">
          <h4 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Coalitions</h4>
          <div style="font-size: 11px; color: #ccc; margin-bottom: 8px;">
            Active: ${coalitions.active_count ?? 0}
          </div>
      `;

      if (coalitions.coalitions && coalitions.coalitions.length > 0) {
        for (let i = 0; i < coalitions.coalitions.length; i++) {
          const coalition = coalitions.coalitions[i];
          const color = CULTURAL_GROUP_COLORS[i % CULTURAL_GROUP_COLORS.length];
          const memberCount = coalition.members?.length ?? 0;
          const cohesion = coalition.cohesion ?? 0;

          html += `
            <div style="background: #1a1a1a; padding: 8px; margin-bottom: 6px; border-radius: 3px; border-left: 3px solid ${color};">
              <div style="font-size: 11px; color: #fff; margin-bottom: 3px;">${coalition.name || coalition.id}</div>
              <div style="font-size: 10px; color: #888;">
                Members: ${memberCount} &nbsp; Cohesion: ${cohesion.toFixed(2)}
              </div>
              ${coalition.goal ? `<div style="font-size: 9px; color: #666; margin-top: 2px;">Goal: ${coalition.goal}</div>` : ''}
            </div>
          `;
        }
      }

      html += '</div>';
    }

    // Social network stats
    if (agents) {
      let totalRelationships = 0;
      let totalTrust = 0;
      let relationshipCount = 0;
      let mostConnectedAgent = null;
      let maxConnections = 0;

      for (const agent of agents) {
        if (!agent.alive || !agent.social_relationships) continue;

        const connections = Object.keys(agent.social_relationships).length;
        totalRelationships += connections;

        for (const rel of Object.values(agent.social_relationships)) {
          if (rel.trust !== undefined) {
            totalTrust += rel.trust;
            relationshipCount++;
          }
        }

        if (connections > maxConnections) {
          maxConnections = connections;
          mostConnectedAgent = agent;
        }
      }

      const avgTrust = relationshipCount > 0 ? (totalTrust / relationshipCount) : 0;

      html += `
        <div style="margin-bottom: 16px;">
          <h4 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Network Stats</h4>
          <div style="font-size: 11px; color: #ccc;">
            <div>Total Relationships: ${totalRelationships}</div>
            <div>Average Trust: ${avgTrust.toFixed(2)}</div>
            ${mostConnectedAgent ? `<div>Most Connected: ${mostConnectedAgent.name} (${maxConnections})</div>` : ''}
          </div>
        </div>
      `;
    }

    html += '<div style="margin-top: 16px; font-size: 11px; color: #666;">Click an agent to inspect relationships</div>';
    html += '</div>';

    container.innerHTML = html;
  }

  getKeyBindings() {
    return {
      'r': () => {
        this._showAllTrustLines = !this._showAllTrustLines;
        return `Show all trust lines: ${this._showAllTrustLines ? 'ON' : 'OFF'}`;
      },
      'c': () => {
        this._showCoalitionHulls = !this._showCoalitionHulls;
        return `Coalition hulls: ${this._showCoalitionHulls ? 'ON' : 'OFF'}`;
      },
    };
  }
}
