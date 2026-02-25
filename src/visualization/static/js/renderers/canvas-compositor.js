/**
 * renderers/canvas-compositor.js
 * Manages 10-layer canvas rendering with try-catch isolation.
 */

export class CanvasCompositor {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this._layers = [];
  }

  /**
   * Set layers for rendering. Layers are sorted by zIndex.
   * @param {Array<{name: string, zIndex: number, draw: Function}>} layers
   */
  setLayers(layers) {
    this._layers = layers.sort((a, b) => a.zIndex - b.zIndex);
  }

  /**
   * Render all layers.
   * @param {Object} renderCtx - Render context passed to each layer
   */
  render(renderCtx) {
    // Clear canvas
    this.ctx.fillStyle = renderCtx.config?.backgroundColor || '#0a0a0a';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Render each layer with isolation
    for (const layer of this._layers) {
      try {
        this.ctx.save();
        layer.draw(renderCtx);
        this.ctx.restore();
      } catch (error) {
        console.warn(`Layer "${layer.name}" failed to render:`, error);
      }
    }
  }

  /**
   * Clear the canvas.
   */
  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Resize canvas to match element size.
   */
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
  }
}
