/**
 * panels/decision-funnel.js
 * Full-canvas SRIE cascade visualization.
 * The "microscope mode" for understanding a single agent's reasoning.
 */

export class DecisionFunnel {
  /**
   * Render the SRIE decision funnel as HTML.
   * @param {Object} data
   * @param {Object} data.srie - SRIE cascade data from agent
   * @param {Object} data.metacognition - Metacognition data
   * @param {Object} data.plan - Plan data
   * @param {Object} data.agent - Agent basic info (name, archetype)
   * @param {Object} options
   * @returns {string} HTML string for the full decision funnel view
   */
  static render(data, options = {}) {
    if (!data || !data.srie) {
      return this._renderEmpty('SRIE data not available');
    }

    const { srie, metacognition, plan, agent } = data;
    const system2Active = metacognition?.deliberation_active || false;

    // Build the four-stage pipeline
    const stages = [
      this._renderSensation(srie.sensation),
      this._renderReflection(srie.reflection, system2Active),
      this._renderIntention(srie.intention, plan),
      this._renderExpression(srie.expression)
    ];

    const agentHeader = agent
      ? `<div class="funnel-header">
          <h2>${this._escapeHtml(agent.name || 'Agent')}</h2>
          <span class="archetype-badge">${this._escapeHtml(agent.archetype || 'Unknown')}</span>
        </div>`
      : '';

    return `
      <div class="decision-funnel">
        ${agentHeader}
        <div class="srie-pipeline">
          ${stages.join('\n')}
        </div>
      </div>
      <style>
        .decision-funnel {
          width: 100%;
          height: 100%;
          background: #0a0a0a;
          color: #fff;
          font-family: Inter, system-ui, sans-serif;
          padding: 20px;
          box-sizing: border-box;
          overflow: auto;
        }
        .funnel-header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 24px;
        }
        .funnel-header h2 {
          margin: 0;
          font-size: 20px;
          font-weight: 600;
        }
        .archetype-badge {
          background: #2a2a2a;
          padding: 4px 10px;
          border-radius: 4px;
          font-size: 11px;
          color: #ccc;
        }
        .srie-pipeline {
          display: flex;
          gap: 20px;
          align-items: stretch;
          position: relative;
        }
        .srie-card {
          flex: 1;
          background: #1a1a1a;
          border-radius: 8px;
          padding: 16px;
          min-height: 200px;
          position: relative;
          border: 2px solid #2a2a2a;
        }
        .srie-card-header {
          font-size: 12px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 16px;
          padding-bottom: 8px;
          border-bottom: 2px solid;
        }
        .srie-card-sensation .srie-card-header { color: #22d3ee; border-color: #22d3ee; }
        .srie-card-reflection .srie-card-header { color: #a78bfa; border-color: #a78bfa; }
        .srie-card-intention .srie-card-header { color: #fbbf24; border-color: #fbbf24; }
        .srie-card-expression .srie-card-header { color: #34d399; border-color: #34d399; }
        .srie-field {
          margin-bottom: 12px;
        }
        .srie-field-label {
          font-size: 10px;
          color: #888;
          text-transform: uppercase;
          margin-bottom: 4px;
        }
        .srie-field-value {
          font-size: 12px;
          color: #fff;
          font-weight: 500;
        }
        .srie-bar {
          height: 6px;
          background: #2a2a2a;
          border-radius: 3px;
          overflow: hidden;
          margin-top: 4px;
        }
        .srie-bar-fill {
          height: 100%;
          border-radius: 3px;
          transition: width 0.3s ease;
        }
        .srie-arrow {
          position: absolute;
          right: -16px;
          top: 50%;
          transform: translateY(-50%);
          width: 0;
          height: 0;
          border-left: 12px solid #444;
          border-top: 8px solid transparent;
          border-bottom: 8px solid transparent;
          z-index: 10;
        }
        .srie-list {
          font-size: 11px;
          color: #ccc;
          line-height: 1.6;
        }
        .srie-list li {
          margin-bottom: 4px;
        }
        .system2-badge {
          background: #a78bfa;
          color: #000;
          padding: 2px 8px;
          border-radius: 3px;
          font-size: 9px;
          font-weight: 700;
          text-transform: uppercase;
          display: inline-block;
          margin-left: 8px;
        }
        .empty-state {
          color: #666;
          font-size: 11px;
          font-style: italic;
        }
      </style>
    `;
  }

  /**
   * Render SENSATION stage
   */
  static _renderSensation(sensation) {
    if (!sensation) {
      return this._renderCard('sensation', 'SENSATION', '<p class="empty-state">No sensation data</p>');
    }

    const agentsCount = sensation.agents_visible?.length || 0;
    const resourcesCount = sensation.resources_nearby?.length || 0;
    const hungerLevel = sensation.hunger ?? 0;
    const messagesCount = sensation.messages_received?.length || 0;

    const content = `
      <div class="srie-field">
        <div class="srie-field-label">Agents Visible</div>
        <div class="srie-field-value">${agentsCount} agents</div>
      </div>
      <div class="srie-field">
        <div class="srie-field-label">Resources Nearby</div>
        <div class="srie-field-value">${resourcesCount} resources</div>
      </div>
      <div class="srie-field">
        <div class="srie-field-label">Hunger</div>
        <div class="srie-field-value">${hungerLevel}</div>
        ${this._renderBar(hungerLevel / 100, '#ff6b6b')}
      </div>
      <div class="srie-field">
        <div class="srie-field-label">Messages</div>
        <div class="srie-field-value">${messagesCount} received</div>
      </div>
    `;

    return this._renderCard('sensation', 'SENSATION', content);
  }

  /**
   * Render REFLECTION stage
   */
  static _renderReflection(reflection, system2Active) {
    if (!reflection) {
      return this._renderCard('reflection', 'REFLECTION', '<p class="empty-state">No reflection data</p>');
    }

    const threat = reflection.threat_level ?? 0;
    const opportunity = reflection.opportunity_level ?? 0;
    const trend = reflection.trend || 'STABLE';
    const lastResult = reflection.last_action_result || 'NONE';

    const trendIcon = this._getTrendIcon(trend);
    const system2Badge = system2Active ? '<span class="system2-badge">System 2</span>' : '';

    const content = `
      <div class="srie-field">
        <div class="srie-field-label">Threat Level${system2Badge}</div>
        <div class="srie-field-value">${threat.toFixed(2)}</div>
        ${this._renderBar(threat, '#ff6b6b')}
      </div>
      <div class="srie-field">
        <div class="srie-field-label">Opportunity</div>
        <div class="srie-field-value">${opportunity.toFixed(2)}</div>
        ${this._renderBar(opportunity, '#34d399')}
      </div>
      <div class="srie-field">
        <div class="srie-field-label">Trend</div>
        <div class="srie-field-value">${trendIcon} ${trend}</div>
      </div>
      <div class="srie-field">
        <div class="srie-field-label">Last Action</div>
        <div class="srie-field-value">${this._escapeHtml(lastResult)}</div>
      </div>
    `;

    return this._renderCard('reflection', 'REFLECTION', content);
  }

  /**
   * Render INTENTION stage
   */
  static _renderIntention(intention, plan) {
    if (!intention) {
      return this._renderCard('intention', 'INTENTION', '<p class="empty-state">No intention data</p>');
    }

    const goal = intention.goal || 'NONE';
    const confidence = intention.confidence ?? 0;
    const target = intention.target ? `(${intention.target.x}, ${intention.target.y})` : 'None';
    const planStep = plan ? `${plan.current_step || 0}/${plan.total_steps || 0}` : 'N/A';

    const content = `
      <div class="srie-field">
        <div class="srie-field-label">Goal</div>
        <div class="srie-field-value">${this._escapeHtml(goal)}</div>
      </div>
      <div class="srie-field">
        <div class="srie-field-label">Confidence</div>
        <div class="srie-field-value">${confidence.toFixed(2)}</div>
        ${this._renderBar(confidence, '#fbbf24')}
      </div>
      <div class="srie-field">
        <div class="srie-field-label">Target</div>
        <div class="srie-field-value">${target}</div>
      </div>
      <div class="srie-field">
        <div class="srie-field-label">Plan Progress</div>
        <div class="srie-field-value">${planStep}</div>
      </div>
    `;

    return this._renderCard('intention', 'INTENTION', content);
  }

  /**
   * Render EXPRESSION stage
   */
  static _renderExpression(expression) {
    if (!expression) {
      return this._renderCard('expression', 'EXPRESSION', '<p class="empty-state">No expression data</p>', true);
    }

    const action = expression.action || 'NONE';
    const message = expression.message || '';
    const targetAgent = expression.target_agent || '';

    const content = `
      <div class="srie-field">
        <div class="srie-field-label">Action</div>
        <div class="srie-field-value">${this._escapeHtml(action)}</div>
      </div>
      ${message ? `
        <div class="srie-field">
          <div class="srie-field-label">Message</div>
          <div class="srie-field-value">"${this._escapeHtml(message)}"</div>
        </div>
      ` : ''}
      ${targetAgent ? `
        <div class="srie-field">
          <div class="srie-field-label">Target</div>
          <div class="srie-field-value">${this._escapeHtml(targetAgent)}</div>
        </div>
      ` : ''}
    `;

    return this._renderCard('expression', 'EXPRESSION', content, true);
  }

  /**
   * Render a stage card
   */
  static _renderCard(stage, title, content, isLast = false) {
    const arrow = isLast ? '' : '<div class="srie-arrow"></div>';

    return `
      <div class="srie-card srie-card-${stage}">
        <div class="srie-card-header">${title}</div>
        ${content}
        ${arrow}
      </div>
    `;
  }

  /**
   * Render a progress bar
   */
  static _renderBar(value, color) {
    const percentage = Math.max(0, Math.min(100, value * 100));
    return `
      <div class="srie-bar">
        <div class="srie-bar-fill" style="width: ${percentage}%; background: ${color};"></div>
      </div>
    `;
  }

  /**
   * Get trend icon
   */
  static _getTrendIcon(trend) {
    const icons = {
      'IMPROVING': '↑↑↑',
      'STABLE': '→',
      'DECLINING': '↓↓↓',
      'VOLATILE': '↕'
    };
    return icons[trend] || '→';
  }

  /**
   * Render compact inline SRIE pipeline (for side panel use).
   * @param {Object} data - Same as render()
   * @returns {string} Compact HTML string
   */
  static renderCompact(data) {
    if (!data || !data.srie) {
      return '<div class="srie-compact-empty">No SRIE data</div>';
    }

    const { srie } = data;

    const sensation = srie.sensation
      ? `A:${srie.sensation.agents_visible?.length || 0} R:${srie.sensation.resources_nearby?.length || 0}`
      : '—';

    const reflection = srie.reflection
      ? `T:${(srie.reflection.threat_level ?? 0).toFixed(1)} O:${(srie.reflection.opportunity_level ?? 0).toFixed(1)}`
      : '—';

    const intention = srie.intention
      ? `${srie.intention.goal || 'NONE'}`
      : '—';

    const expression = srie.expression
      ? `${srie.expression.action || 'NONE'}`
      : '—';

    return `
      <div class="srie-compact">
        <div class="srie-compact-stage">
          <span class="srie-compact-label" style="color: #22d3ee;">S</span>
          <span class="srie-compact-value">${sensation}</span>
        </div>
        <span class="srie-compact-arrow">→</span>
        <div class="srie-compact-stage">
          <span class="srie-compact-label" style="color: #a78bfa;">R</span>
          <span class="srie-compact-value">${reflection}</span>
        </div>
        <span class="srie-compact-arrow">→</span>
        <div class="srie-compact-stage">
          <span class="srie-compact-label" style="color: #fbbf24;">I</span>
          <span class="srie-compact-value">${this._escapeHtml(intention)}</span>
        </div>
        <span class="srie-compact-arrow">→</span>
        <div class="srie-compact-stage">
          <span class="srie-compact-label" style="color: #34d399;">E</span>
          <span class="srie-compact-value">${this._escapeHtml(expression)}</span>
        </div>
      </div>
      <style>
        .srie-compact {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px;
          background: #1a1a1a;
          border-radius: 6px;
          font-family: Inter, system-ui, sans-serif;
        }
        .srie-compact-stage {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .srie-compact-label {
          font-size: 9px;
          font-weight: 700;
          text-transform: uppercase;
        }
        .srie-compact-value {
          font-size: 10px;
          color: #ccc;
        }
        .srie-compact-arrow {
          color: #444;
          font-size: 12px;
        }
        .srie-compact-empty {
          color: #666;
          font-size: 11px;
          font-style: italic;
          padding: 8px;
        }
      </style>
    `;
  }

  /**
   * Render empty state
   */
  static _renderEmpty(message) {
    return `
      <div class="decision-funnel">
        <div class="empty-state" style="text-align: center; padding: 40px;">
          ${this._escapeHtml(message)}
        </div>
      </div>
      <style>
        .decision-funnel {
          width: 100%;
          height: 100%;
          background: #0a0a0a;
          color: #fff;
          font-family: Inter, system-ui, sans-serif;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .empty-state {
          color: #666;
          font-size: 14px;
        }
      </style>
    `;
  }

  /**
   * Escape HTML entities
   */
  static _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}
