/**
 * renderers/overlay-renderer.js
 * Utility functions for drawing lines, arcs, hulls, heatmaps.
 */

import { withAlpha } from '../core/colors.js';

export class OverlayRenderer {
  /**
   * Draw a line between two points.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x1 - Start X
   * @param {number} y1 - Start Y
   * @param {number} x2 - End X
   * @param {number} y2 - End Y
   * @param {Object} style - {color, width, dashed, alpha}
   */
  static drawLine(ctx, x1, y1, x2, y2, style = {}) {
    ctx.save();

    if (style.alpha) ctx.globalAlpha = style.alpha;
    ctx.strokeStyle = style.color || '#fff';
    ctx.lineWidth = style.width || 2;

    if (style.dashed) {
      ctx.setLineDash(style.dashed);
    }

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Draw an arrow from (x1, y1) to (x2, y2).
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x1
   * @param {number} y1
   * @param {number} x2
   * @param {number} y2
   * @param {Object} style - {color, width, headLength, alpha}
   */
  static drawArrow(ctx, x1, y1, x2, y2, style = {}) {
    ctx.save();

    if (style.alpha) ctx.globalAlpha = style.alpha;
    ctx.strokeStyle = style.color || '#fff';
    ctx.lineWidth = style.width || 2;

    // Line
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Arrowhead
    const headLength = style.headLength || 6;
    const angle = Math.atan2(y2 - y1, x2 - x1);

    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(
      x2 - headLength * Math.cos(angle - Math.PI / 6),
      y2 - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.moveTo(x2, y2);
    ctx.lineTo(
      x2 - headLength * Math.cos(angle + Math.PI / 6),
      y2 - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Draw a convex hull around a set of points.
   * @param {CanvasRenderingContext2D} ctx
   * @param {Array<{x: number, y: number}>} points
   * @param {Object} style - {color, alpha, lineWidth}
   */
  static drawConvexHull(ctx, points, style = {}) {
    if (points.length < 3) return;

    const hull = OverlayRenderer._computeConvexHull(points);
    if (hull.length < 3) return;

    ctx.save();

    if (style.alpha) ctx.globalAlpha = style.alpha;
    ctx.strokeStyle = style.color || '#fff';
    ctx.lineWidth = style.lineWidth || 2;

    ctx.beginPath();
    ctx.moveTo(hull[0].x, hull[0].y);
    for (let i = 1; i < hull.length; i++) {
      ctx.lineTo(hull[i].x, hull[i].y);
    }
    ctx.closePath();
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Compute convex hull using Gift Wrapping algorithm.
   * @param {Array<{x: number, y: number}>} points
   * @returns {Array<{x: number, y: number}>}
   */
  static _computeConvexHull(points) {
    if (points.length < 3) return points;

    // Find leftmost point
    let leftmost = points[0];
    for (const p of points) {
      if (p.x < leftmost.x) leftmost = p;
    }

    const hull = [];
    let current = leftmost;

    do {
      hull.push(current);
      let next = points[0];

      for (const p of points) {
        if (p === current) continue;

        const cross = OverlayRenderer._crossProduct(current, next, p);
        if (next === current || cross > 0 || (cross === 0 && OverlayRenderer._distance(current, p) > OverlayRenderer._distance(current, next))) {
          next = p;
        }
      }

      current = next;
    } while (current !== leftmost && hull.length < points.length + 1);

    return hull;
  }

  /**
   * Cross product for orientation.
   */
  static _crossProduct(o, a, b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  }

  /**
   * Distance between two points.
   */
  static _distance(a, b) {
    return Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
  }

  /**
   * Draw a heatmap overlay.
   * @param {CanvasRenderingContext2D} ctx
   * @param {Object} config - Canvas config
   * @param {Array<{x: number, y: number, value: number}>} data
   * @param {Object} style - {colorLow, colorHigh, maxValue}
   */
  static drawHeatmap(ctx, config, data, style = {}) {
    const { cellSize, offsetX, offsetY } = config;
    const colorLow = style.colorLow || '#0000ff';
    const colorHigh = style.colorHigh || '#ff0000';
    const maxValue = style.maxValue || 1;

    for (const point of data) {
      const x = offsetX + point.x * cellSize;
      const y = offsetY + point.y * cellSize;
      const normalized = Math.min(point.value / maxValue, 1);

      // Interpolate color
      const c1 = OverlayRenderer._hexToRgb(colorLow);
      const c2 = OverlayRenderer._hexToRgb(colorHigh);

      const r = Math.round(c1.r + (c2.r - c1.r) * normalized);
      const g = Math.round(c1.g + (c2.g - c1.g) * normalized);
      const b = Math.round(c1.b + (c2.b - c1.b) * normalized);

      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.3)`;
      ctx.fillRect(x, y, cellSize, cellSize);
    }
  }

  /**
   * Convert hex to RGB.
   */
  static _hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16),
    } : { r: 0, g: 0, b: 0 };
  }
}
