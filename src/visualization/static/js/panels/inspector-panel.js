/**
 * panels/inspector-panel.js
 * Agent inspector panel that adapts to active lens.
 */

export class InspectorPanel {
  /**
   * @param {HTMLElement} container - Container element
   */
  constructor(container) {
    this.container = container;
  }

  /**
   * Render inspector content.
   * @param {Object} renderCtx - Render context with selectedAgentId, agents, activeLens
   */
  render(renderCtx) {
    const { selectedAgentId, activeLens } = renderCtx;

    if (!selectedAgentId || !activeLens) {
      this.container.innerHTML = '<div class="inspector-empty">Click an agent on the canvas to inspect</div>';
      return;
    }

    // Delegate to active lens for content
    activeLens.renderSidePanel(this.container, renderCtx);
  }

  /**
   * Clear inspector.
   */
  clear() {
    this.container.innerHTML = '<div class="inspector-empty">Click an agent on the canvas to inspect</div>';
  }
}
