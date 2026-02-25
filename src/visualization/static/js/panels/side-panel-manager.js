/**
 * panels/side-panel-manager.js
 * Manages side panel content routing to active lens.
 */

export class SidePanelManager {
  /**
   * @param {HTMLElement} container - Side panel container
   */
  constructor(container) {
    this.container = container;
  }

  /**
   * Render side panel for current state.
   * @param {Object} renderCtx - Render context
   */
  render(renderCtx) {
    const { activeLens, selectedAgentId } = renderCtx;

    if (!activeLens) {
      this.container.innerHTML = '<div style="color: #666; padding: 20px;">No active lens</div>';
      return;
    }

    // Clear previous content
    this.container.innerHTML = '';

    // Create inspector section
    const inspectorSection = document.createElement('div');
    inspectorSection.className = 'sidebar-section inspector';
    inspectorSection.innerHTML = '<h2>Agent Inspector</h2><div id="inspector-content"></div>';
    this.container.appendChild(inspectorSection);

    const inspectorContent = inspectorSection.querySelector('#inspector-content');

    // Render lens-specific inspector content
    if (selectedAgentId) {
      activeLens.renderSidePanel(inspectorContent, renderCtx);
    } else {
      inspectorContent.innerHTML = '<div class="inspector-empty">Click an agent on the canvas to inspect</div>';
    }

    // Create events section
    const eventsSection = document.createElement('div');
    eventsSection.className = 'sidebar-section';
    eventsSection.innerHTML = `
      <h2>Emergence Events</h2>
      <div class="events-feed" id="events-feed"></div>
    `;
    this.container.appendChild(eventsSection);

    // Populate events
    this._renderEvents(renderCtx);
  }

  /**
   * Render emergence events feed.
   * @param {Object} renderCtx
   */
  _renderEvents(renderCtx) {
    const feed = this.container.querySelector('#events-feed');
    if (!feed) return;

    const tickData = renderCtx.temporalBuffer?.getCurrentTick();
    const events = tickData?.emergent_events || [];

    if (events.length === 0) {
      feed.innerHTML = '<div style="color: #666; font-size: 12px; font-style: italic;">Waiting for events...</div>';
      return;
    }

    feed.innerHTML = events.slice(-20).reverse().map((event, index) => {
      const isNew = index === 0;
      return `<div class="event-item ${isNew ? 'new' : ''}">${event}</div>`;
    }).join('');
  }
}
