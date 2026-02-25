/**
 * renderers/grid-renderer.js
 * Renders grid lines and resource tiles.
 */

export class GridRenderer {
  /**
   * Create layer for grid rendering.
   * @returns {{name: string, zIndex: number, draw: Function}}
   */
  static createGridLayer() {
    return {
      name: 'grid',
      zIndex: 1,
      draw: (renderCtx) => {
        const { ctx, config, toggles } = renderCtx;

        if (!toggles.grid) return;

        const { worldWidth, worldHeight, cellSize, offsetX, offsetY } = config;
        const canvasWidth = worldWidth * cellSize;
        const canvasHeight = worldHeight * cellSize;

        ctx.strokeStyle = config.gridColor || '#1a1a1a';
        ctx.lineWidth = 1;

        // Vertical lines
        for (let x = 0; x <= worldWidth; x++) {
          ctx.beginPath();
          ctx.moveTo(offsetX + x * cellSize, offsetY);
          ctx.lineTo(offsetX + x * cellSize, offsetY + canvasHeight);
          ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y <= worldHeight; y++) {
          ctx.beginPath();
          ctx.moveTo(offsetX, offsetY + y * cellSize);
          ctx.lineTo(offsetX + canvasWidth, offsetY + y * cellSize);
          ctx.stroke();
        }
      },
    };
  }

  /**
   * Create layer for resource tiles.
   * @returns {{name: string, zIndex: number, draw: Function}}
   */
  static createResourceLayer() {
    return {
      name: 'resources',
      zIndex: 2,
      draw: (renderCtx) => {
        const { ctx, config, currentTick } = renderCtx;
        const tickData = renderCtx.temporalBuffer?.getCurrentTick();

        if (!tickData || !tickData.resources) return;

        const { cellSize, offsetX, offsetY } = config;

        // Draw resource tiles (density as opacity)
        for (const tile of tickData.resources) {
          if (!tile.resources || Object.keys(tile.resources).length === 0) continue;

          const x = offsetX + tile.x * cellSize;
          const y = offsetY + tile.y * cellSize;

          // Calculate total resource density
          let totalResources = 0;
          for (const [_, qty] of Object.entries(tile.resources)) {
            totalResources += qty;
          }

          // Opacity based on density (max = 10 resources)
          const opacity = Math.min(totalResources / 10, 1);

          ctx.fillStyle = `rgba(34, 197, 94, ${opacity * 0.3})`;
          ctx.fillRect(x, y, cellSize, cellSize);

          // Draw resource icon if dense enough
          if (opacity > 0.3) {
            ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
            ctx.font = `${cellSize * 0.5}px monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('🌾', x + cellSize / 2, y + cellSize / 2);
          }
        }
      },
    };
  }
}
