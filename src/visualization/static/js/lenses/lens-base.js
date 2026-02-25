/**
 * lenses/lens-base.js
 * Abstract base class for visualization lenses.
 */

export class LensBase {
  constructor(name) {
    this.name = name;
  }

  /**
   * Get canvas layers for this lens.
   * @param {Object} renderCtx - Render context
   * @returns {Array<{name: string, zIndex: number, draw: Function}>}
   */
  getCanvasLayers(renderCtx) {
    throw new Error('getCanvasLayers() must be implemented by subclass');
  }

  /**
   * Render side panel content for this lens.
   * @param {HTMLElement} container - Side panel container
   * @param {Object} renderCtx - Render context
   */
  renderSidePanel(container, renderCtx) {
    throw new Error('renderSidePanel() must be implemented by subclass');
  }

  /**
   * Get agent visual style for this lens.
   * @param {Object} agent - Agent data
   * @param {Object} renderCtx - Render context
   * @returns {{color: string, shape: string, size: number, glow?: Object, label?: string}}
   */
  getAgentStyle(agent, renderCtx) {
    throw new Error('getAgentStyle() must be implemented by subclass');
  }

  /**
   * Handle agent selection in this lens.
   * @param {string} agentId - Selected agent ID
   * @param {Object} renderCtx - Render context
   */
  onAgentSelect(agentId, renderCtx) {
    // Default: do nothing
  }

  /**
   * Get timeline markers for this lens.
   * @param {number} tick - Current tick
   * @param {Object} renderCtx - Render context
   * @returns {Array<{tick: number, label: string, color: string, shape: string}>}
   */
  getTimelineMarkers(tick, renderCtx) {
    return [];
  }

  /**
   * Get keyboard bindings specific to this lens.
   * @returns {Object} Map of key -> handler function
   */
  getKeyBindings() {
    return {};
  }

  /**
   * Called when lens becomes active.
   * @param {Object} renderCtx - Render context
   */
  onActivate(renderCtx) {
    // Default: do nothing
  }

  /**
   * Called when lens becomes inactive.
   * @param {Object} renderCtx - Render context
   */
  onDeactivate(renderCtx) {
    // Default: do nothing
  }
}
