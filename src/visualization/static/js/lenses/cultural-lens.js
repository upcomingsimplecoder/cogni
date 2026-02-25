/**
 * lenses/cultural-lens.js
 * Cultural transmission view - visualizes learning styles, cultural groups,
 * transmission events, language evolution, and meme spreading.
 */

import { LensBase } from './lens-base.js';
import { ARCHETYPE_COLORS, getCulturalGroupColor, CULTURAL_GROUP_COLORS, BIAS_COLORS, withAlpha } from '../core/colors.js';
import { GridRenderer } from '../renderers/grid-renderer.js';
import { AgentRenderer } from '../renderers/agent-renderer.js';
import { OverlayRenderer } from '../renderers/overlay-renderer.js';

export class CulturalLens extends LensBase {
  constructor() {
    super('cultural');
    this._transmissionLines = []; // {observerId, actorId, biasType, remainingTicks}
    this._showLanguageBubbles = true;
    this._showTransmissionLines = true;
  }

  getCanvasLayers(renderCtx) {
    return [
      // Layer 1: Grid
      GridRenderer.createGridLayer(),
      // Layer 3: Cultural group territories
      this._createCulturalGroupLayer(),
      // Layer 5: Transmission lines
      this._createTransmissionLayer(),
      // Layer 6: Agent bodies
      AgentRenderer.createAgentLayer(),
      // Layer 7: Language bubbles
      this._createLanguageBubbleLayer(),
    ];
  }

  /**
   * Create layer for cultural group territories (convex hulls).
   */
  _createCulturalGroupLayer() {
    return {
      name: 'cultural-territories',
      zIndex: 3,
      draw: (renderCtx) => {
        const { ctx, config, agents } = renderCtx;
        const { cellSize, offsetX, offsetY } = config;

        if (!agents || agents.length === 0) return;

        // Group agents by cultural group
        const groupMap = new Map();
        for (const agent of agents) {
          if (!agent.alive) continue;
          const groupId = agent.cultural?.cultural_group;
          if (groupId === undefined || groupId === null || groupId < 0) continue;

          if (!groupMap.has(groupId)) {
            groupMap.set(groupId, []);
          }
          groupMap.get(groupId).push(agent);
        }

        // Draw convex hull for each group
        for (const [groupId, members] of groupMap.entries()) {
          if (members.length < 3) continue;

          const points = members.map(agent => ({
            x: offsetX + agent.x * cellSize + cellSize / 2,
            y: offsetY + agent.y * cellSize + cellSize / 2,
          }));

          const color = CULTURAL_GROUP_COLORS[groupId % CULTURAL_GROUP_COLORS.length];

          // Fill
          const hull = OverlayRenderer._computeConvexHull(points);
          if (hull.length >= 3) {
            ctx.save();
            ctx.globalAlpha = 0.12;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(hull[0].x, hull[0].y);
            for (let i = 1; i < hull.length; i++) {
              ctx.lineTo(hull[i].x, hull[i].y);
            }
            ctx.closePath();
            ctx.fill();
            ctx.restore();

            // Stroke
            ctx.save();
            ctx.globalAlpha = 0.4;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(hull[0].x, hull[0].y);
            for (let i = 1; i < hull.length; i++) {
              ctx.lineTo(hull[i].x, hull[i].y);
            }
            ctx.closePath();
            ctx.stroke();
            ctx.restore();
          }
        }
      },
    };
  }

  /**
   * Create layer for cultural transmission lines.
   */
  _createTransmissionLayer() {
    return {
      name: 'transmission-lines',
      zIndex: 5,
      draw: (renderCtx) => {
        if (!this._showTransmissionLines || this._transmissionLines.length === 0) return;

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
        }

        // Decay transmission lines
        this._transmissionLines = this._transmissionLines
          .map(t => ({ ...t, remainingTicks: t.remainingTicks - 1 }))
          .filter(t => t.remainingTicks > 0);
      },
    };
  }

  /**
   * Create layer for language bubbles.
   */
  _createLanguageBubbleLayer() {
    return {
      name: 'language-bubbles',
      zIndex: 7,
      draw: (renderCtx) => {
        if (!this._showLanguageBubbles) return;

        const { ctx, config, agents } = renderCtx;
        const { cellSize, offsetX, offsetY } = config;

        if (!agents || agents.length === 0) return;

        for (const agent of agents) {
          if (!agent.alive) continue;
          if (!agent.language?.symbols || agent.language.symbols.length === 0) continue;

          const x = offsetX + agent.x * cellSize + cellSize / 2;
          const y = offsetY + agent.y * cellSize + cellSize / 2;

          const vocabSize = agent.language.vocabulary_size || agent.language.symbols.length;

          // Draw speech bubble
          ctx.save();
          ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
          ctx.strokeStyle = '#888';
          ctx.lineWidth = 1;

          const bubbleRadius = cellSize * 0.25;
          const bubbleY = y - cellSize * 0.7;

          ctx.beginPath();
          ctx.arc(x, bubbleY, bubbleRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();

          // Draw vocabulary size
          ctx.fillStyle = '#000';
          ctx.font = `${cellSize * 0.25}px monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(vocabSize.toString(), x, bubbleY);

          ctx.restore();
        }
      },
    };
  }

  getAgentStyle(agent, renderCtx) {
    // Shape by learning style
    const learningStyle = agent.cultural?.learning_style;
    let shape = 'circle';
    if (learningStyle === 'prestige') shape = 'diamond';
    else if (learningStyle === 'conformist') shape = 'circle';
    else if (learningStyle === 'content') shape = 'hex';
    else if (learningStyle === 'anti_conformist') shape = 'triangle';

    // Color by cultural group
    const groupColor = getCulturalGroupColor(agent.cultural?.cultural_group);
    const color = groupColor || agent.color || ARCHETYPE_COLORS[agent.archetype] || '#888';

    // Size by repertoire size
    const repertoireSize = agent.cultural?.repertoire_size || 0;
    const size = Math.min(0.25 + repertoireSize * 0.01, 0.45);

    // Glow if adopted something this tick
    let glow = null;
    const tickData = renderCtx.temporalBuffer?.getCurrentTick();
    if (tickData?.cultural?.transmission_events) {
      const adopted = tickData.cultural.transmission_events.find(
        e => e.observer === agent.id && e.adopted === true
      );
      if (adopted) {
        glow = { color: '#facc15', alpha: 0.5, radius: 0.6 };
      }
    }

    return { color, shape, size, glow };
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
    const groupColor = getCulturalGroupColor(agent.cultural?.cultural_group);
    const color = groupColor || agent.color || ARCHETYPE_COLORS[agent.archetype] || '#888';

    // Cultural profile
    const learningStyle = agent.cultural?.learning_style || 'unknown';
    const biasColor = BIAS_COLORS[learningStyle] || '#888';
    const repertoireSize = agent.cultural?.repertoire_size || 0;
    const adoptedCount = agent.cultural?.adopted_count || 0;
    const culturalGroup = agent.cultural?.cultural_group ?? -1;

    // Language data
    const languageHtml = agent.language
      ? this._renderLanguageSection(agent.language)
      : '<div style="color: #666; font-size: 11px;">No language data</div>';

    // Transmission history
    const transmissionHtml = this._renderTransmissionHistory(agent.id, renderCtx);

    container.innerHTML = `
      <div class="agent-header-info">
        <div class="agent-color-dot" style="background: ${color};"></div>
        <div class="agent-name-archetype">
          <div class="name">${agent.name}</div>
          <div class="archetype">${agent.archetype}</div>
        </div>
      </div>

      <div class="cultural-profile-section">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Cultural Profile</h3>
        <div style="display: flex; flex-direction: column; gap: 4px; font-size: 12px;">
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Learning Style:</span>
            <span style="color: ${biasColor}; font-weight: bold;">${learningStyle}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Repertoire Size:</span>
            <span style="color: #fff;">${repertoireSize}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Adopted Count:</span>
            <span style="color: #fff;">${adoptedCount}</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #888;">Cultural Group:</span>
            <div style="display: flex; align-items: center; gap: 4px;">
              ${culturalGroup >= 0 ? `<div style="width: 12px; height: 12px; background: ${getCulturalGroupColor(culturalGroup)}; border-radius: 2px;"></div>` : ''}
              <span style="color: #fff;">${culturalGroup >= 0 ? culturalGroup : 'None'}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="language-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Language</h3>
        ${languageHtml}
      </div>

      <div class="transmission-history-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Transmission History</h3>
        ${transmissionHtml}
      </div>
    `;
  }

  _renderLanguageSection(language) {
    const vocabSize = language.vocabulary_size || 0;
    const conventionCount = language.convention_count || 0;
    const successRate = language.comm_success_rate || 0;
    const symbols = language.symbols || [];

    // Sort symbols by strength descending, take top 10
    const topSymbols = [...symbols]
      .sort((a, b) => (b.strength || 0) - (a.strength || 0))
      .slice(0, 10);

    const symbolsHtml = topSymbols.length > 0
      ? topSymbols.map(sym => {
          const strength = sym.strength || 0;
          const barWidth = Math.round(strength * 100);
          return `
            <div style="margin-bottom: 6px; padding: 4px; background: #1a1a1a; border-radius: 2px;">
              <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
                <span style="color: #fff; font-family: monospace;">${sym.form || '?'}</span>
                <span style="color: #888;">${sym.meaning || 'unknown'}</span>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 10px; color: #666; margin-bottom: 2px;">
                <span>Used: ${sym.times_used || 0}</span>
                <span>Success: ${((sym.success_rate || 0) * 100).toFixed(0)}%</span>
              </div>
              <div style="background: #2a2a2a; height: 3px; border-radius: 1px;">
                <div style="background: #22c55e; height: 3px; width: ${barWidth}%; border-radius: 1px;"></div>
              </div>
            </div>
          `;
        }).join('')
      : '<div style="color: #666; font-size: 11px;">No symbols learned</div>';

    return `
      <div style="display: flex; flex-direction: column; gap: 4px; font-size: 12px; margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between;">
          <span style="color: #888;">Vocabulary:</span>
          <span style="color: #fff;">${vocabSize}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: #888;">Conventions:</span>
          <span style="color: #fff;">${conventionCount}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: #888;">Success Rate:</span>
          <div style="flex: 1; margin-left: 8px; background: #2a2a2a; height: 12px; border-radius: 2px; overflow: hidden;">
            <div style="background: #22c55e; height: 12px; width: ${successRate * 100}%;"></div>
          </div>
        </div>
      </div>
      <div style="margin-top: 8px;">
        <div style="font-size: 10px; color: #888; margin-bottom: 4px; text-transform: uppercase;">Symbols</div>
        ${symbolsHtml}
      </div>
    `;
  }

  _renderTransmissionHistory(agentId, renderCtx) {
    const allTicks = renderCtx.temporalBuffer?.getAllTicks() || [];
    const events = [];

    // Collect last 5 transmission events involving this agent
    for (let i = allTicks.length - 1; i >= 0 && events.length < 5; i--) {
      const tickData = allTicks[i];
      if (!tickData.cultural?.transmission_events) continue;

      for (const event of tickData.cultural.transmission_events) {
        if (event.observer === agentId || event.actor === agentId) {
          events.push({ ...event, tick: tickData.tick });
          if (events.length >= 5) break;
        }
      }
    }

    if (events.length === 0) {
      return '<div style="color: #666; font-size: 11px;">No recent transmissions</div>';
    }

    return events.map(event => {
      const role = event.observer === agentId ? 'observer' : 'actor';
      const biasColor = BIAS_COLORS[event.bias] || '#888';
      const adopted = event.adopted ? '✓ Adopted' : '✗ Rejected';
      const adoptedColor = event.adopted ? '#22c55e' : '#ef4444';

      return `
        <div style="margin-bottom: 6px; padding: 4px; background: #1a1a1a; border-radius: 2px; font-size: 11px;">
          <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
            <span style="color: #888;">Tick ${event.tick}</span>
            <span style="color: ${adoptedColor};">${adopted}</span>
          </div>
          <div style="display: flex; justify-content: space-between; color: #ccc;">
            <span>Role: <span style="color: #fff;">${role}</span></span>
            <span>Bias: <span style="color: ${biasColor};">${event.bias}</span></span>
          </div>
        </div>
      `;
    }).join('');
  }

  _renderOverview(container, renderCtx) {
    const tickData = renderCtx.temporalBuffer?.getCurrentTick();
    const cultural = tickData?.cultural;
    const language = tickData?.language;

    if (!cultural && !language) {
      container.innerHTML = '<div class="inspector-empty">Click an agent to inspect cultural profile</div>';
      return;
    }

    // Cultural overview
    const culturalHtml = cultural ? `
      <div class="cultural-overview-section">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Cultural Overview</h3>
        <div style="display: flex; flex-direction: column; gap: 4px; font-size: 12px;">
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Cultural Groups:</span>
            <span style="color: #fff;">${cultural.cultural_groups?.length || 0}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Total Adopted:</span>
            <span style="color: #fff;">${cultural.total_adopted || 0}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Diversity Score:</span>
            <span style="color: #fff;">${(cultural.diversity || 0).toFixed(3)}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Transmissions This Tick:</span>
            <span style="color: #fff;">${cultural.transmission_events?.length || 0}</span>
          </div>
        </div>
        ${this._renderVariantFrequencies(cultural.variant_frequencies)}
      </div>
    ` : '';

    // Language overview
    const languageHtml = language ? `
      <div class="language-overview-section" style="margin-top: 12px;">
        <h3 style="font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase;">Language Overview</h3>
        <div style="display: flex; flex-direction: column; gap: 4px; font-size: 12px;">
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Total Vocabulary:</span>
            <span style="color: #fff;">${language.total_vocabulary || 0}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Unique Symbols:</span>
            <span style="color: #fff;">${language.unique_symbols || 0}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Conventions:</span>
            <span style="color: #fff;">${language.established_conventions || 0}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Messages This Tick:</span>
            <span style="color: #fff;">${language.messages_this_tick || 0}</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Innovations This Tick:</span>
            <span style="color: #fff;">${language.innovations_this_tick || 0}</span>
          </div>
        </div>
      </div>
    ` : '';

    container.innerHTML = culturalHtml + languageHtml;
  }

  _renderVariantFrequencies(frequencies) {
    if (!frequencies || Object.keys(frequencies).length === 0) {
      return '<div style="color: #666; font-size: 11px; margin-top: 8px;">No variant data</div>';
    }

    // Sort by frequency descending, take top 8
    const sorted = Object.entries(frequencies)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 8);

    const maxFreq = sorted[0]?.[1] || 1;

    const barsHtml = sorted.map(([variant, freq]) => {
      const percentage = (freq / maxFreq) * 100;
      return `
        <div style="display: flex; align-items: center; gap: 4px; margin-bottom: 4px; font-size: 11px;">
          <span style="color: #888; width: 80px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${variant}</span>
          <div style="flex: 1; background: #2a2a2a; height: 12px; border-radius: 2px; overflow: hidden;">
            <div style="background: #3b82f6; height: 12px; width: ${percentage}%;"></div>
          </div>
          <span style="color: #fff; width: 30px; text-align: right;">${freq}</span>
        </div>
      `;
    }).join('');

    return `
      <div style="margin-top: 8px;">
        <div style="font-size: 10px; color: #888; margin-bottom: 4px; text-transform: uppercase;">Variant Frequencies (Top 8)</div>
        ${barsHtml}
      </div>
    `;
  }

  getTimelineMarkers(tick, renderCtx) {
    const markers = [];
    const allTicks = renderCtx.temporalBuffer?.getAllTicks() || [];

    for (const tickData of allTicks) {
      if (!tickData.emergence) continue;

      for (const event of tickData.emergence) {
        const text = event.text || event.description || '';
        if (text.toLowerCase().includes('innovat') || text.toLowerCase().includes('convention')) {
          markers.push({
            tick: tickData.tick,
            label: 'Innovation',
            color: '#facc15',
            shape: 'diamond',
          });
        }
      }
    }

    return markers;
  }

  getKeyBindings() {
    return {
      l: () => {
        this._showLanguageBubbles = !this._showLanguageBubbles;
        return `Language bubbles ${this._showLanguageBubbles ? 'shown' : 'hidden'}`;
      },
      t: () => {
        this._showTransmissionLines = !this._showTransmissionLines;
        return `Transmission lines ${this._showTransmissionLines ? 'shown' : 'hidden'}`;
      },
    };
  }

  onActivate(renderCtx) {
    // Handle transmission events from tick data
    const tickData = renderCtx.temporalBuffer?.getCurrentTick();
    if (tickData?.cultural?.transmission_events) {
      this.updateTransmissions(tickData.cultural.transmission_events);
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
