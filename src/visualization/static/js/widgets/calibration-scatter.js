/**
 * widgets/calibration-scatter.js
 * Scatter plot: predicted confidence (X) vs actual success rate (Y).
 * Perfect calibration = y=x diagonal.
 */

import { interpolate, withAlpha } from '../core/colors.js';

export class CalibrationScatter {
  /**
   * Render calibration scatter plot as SVG.
   * @param {Array<{bin_center: number, accuracy: number, count: number}>} curve - Calibration curve data
   * @param {Object} options
   * @param {number} options.width - SVG width (default 200)
   * @param {number} options.height - SVG height (default 150)
   * @param {string} options.dotColor - Dot color (default '#22c55e')
   * @param {string} options.lineColor - Reference diagonal color (default '#444')
   * @param {number} options.minDotSize - Minimum dot radius (default 3)
   * @param {number} options.maxDotSize - Maximum dot radius (default 8)
   * @returns {string} SVG HTML string
   */
  static render(curve, options = {}) {
    const width = options.width ?? 200;
    const height = options.height ?? 150;
    const dotColor = options.dotColor ?? '#22c55e';
    const lineColor = options.lineColor ?? '#444';
    const minDotSize = options.minDotSize ?? 3;
    const maxDotSize = options.maxDotSize ?? 8;

    // Padding for axes and labels
    const padding = { top: 10, right: 10, bottom: 30, left: 40 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Handle empty data
    if (!curve || curve.length === 0) {
      return this._renderEmpty(width, height, padding);
    }

    // Filter valid points
    const points = curve.filter(p =>
      p.bin_center != null &&
      p.accuracy != null &&
      p.count != null &&
      !isNaN(p.bin_center) &&
      !isNaN(p.accuracy)
    );

    if (points.length === 0) {
      return this._renderEmpty(width, height, padding);
    }

    // Calculate dot sizes based on count
    const counts = points.map(p => p.count);
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);
    const countRange = maxCount - minCount;

    const getDotRadius = (count) => {
      if (countRange === 0) return (minDotSize + maxDotSize) / 2;
      const normalized = (count - minCount) / countRange;
      return minDotSize + normalized * (maxDotSize - minDotSize);
    };

    // Scale functions (0-1 range for both axes)
    const scaleX = (val) => padding.left + val * plotWidth;
    const scaleY = (val) => padding.top + (1 - val) * plotHeight; // Invert Y

    // Build SVG elements
    const elements = [];

    // Grid lines (subtle)
    const gridColor = '#2a2a2a';
    for (let i = 0; i <= 4; i++) {
      const val = i / 4;
      const x = scaleX(val);
      const y = scaleY(val);

      // Vertical grid line
      elements.push(`<line x1="${x}" y1="${padding.top}" x2="${x}" y2="${height - padding.bottom}" stroke="${gridColor}" stroke-width="1"/>`);

      // Horizontal grid line
      elements.push(`<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="${gridColor}" stroke-width="1"/>`);
    }

    // Perfect calibration diagonal (y=x)
    const diagonalStart = scaleX(0);
    const diagonalEnd = scaleX(1);
    elements.push(
      `<line x1="${diagonalStart}" y1="${scaleY(0)}" x2="${diagonalEnd}" y2="${scaleY(1)}" ` +
      `stroke="${lineColor}" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.6"/>`
    );

    // Calibration curve line (connect dots)
    if (points.length > 1) {
      const sortedPoints = [...points].sort((a, b) => a.bin_center - b.bin_center);
      const pathData = sortedPoints.map((p, i) => {
        const x = scaleX(p.bin_center);
        const y = scaleY(p.accuracy);
        return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
      }).join(' ');

      elements.push(
        `<path d="${pathData}" stroke="${dotColor}" stroke-width="2" fill="none" opacity="0.5"/>`
      );
    }

    // Dots with color based on calibration error
    for (const point of points) {
      const x = scaleX(point.bin_center);
      const y = scaleY(point.accuracy);
      const radius = getDotRadius(point.count);

      // Calculate distance from diagonal (calibration error)
      const error = Math.abs(point.accuracy - point.bin_center);

      // Color gradient: green (well-calibrated) to red (poorly calibrated)
      const errorThreshold = 0.15; // Max expected error for color scaling
      const errorRatio = Math.min(error / errorThreshold, 1);
      const color = interpolate('#22c55e', '#ef4444', errorRatio);

      elements.push(
        `<circle cx="${x}" cy="${y}" r="${radius}" ` +
        `fill="${color}" stroke="#0a0a0a" stroke-width="1" opacity="0.8"/>`
      );

      // Tooltip title for hover
      const title = `Predicted: ${(point.bin_center * 100).toFixed(0)}%, Actual: ${(point.accuracy * 100).toFixed(0)}%, n=${point.count}`;
      elements.push(`<title>${title}</title>`);
    }

    // Axis labels
    const labelColor = '#888';
    const labelSize = '10px';

    // X-axis ticks and labels
    for (let i = 0; i <= 4; i++) {
      const val = i / 4;
      const x = scaleX(val);
      const y = height - padding.bottom;

      elements.push(
        `<text x="${x}" y="${y + 15}" fill="${labelColor}" font-size="9px" text-anchor="middle">` +
        `${Math.round(val * 100)}%</text>`
      );
    }

    // Y-axis ticks and labels
    for (let i = 0; i <= 4; i++) {
      const val = i / 4;
      const x = padding.left;
      const y = scaleY(val);

      elements.push(
        `<text x="${x - 5}" y="${y + 3}" fill="${labelColor}" font-size="9px" text-anchor="end">` +
        `${Math.round(val * 100)}%</text>`
      );
    }

    // Axis titles
    elements.push(
      `<text x="${width / 2}" y="${height - 5}" fill="${labelColor}" font-size="${labelSize}" text-anchor="middle">` +
      `Predicted Confidence</text>`
    );

    elements.push(
      `<text x="15" y="${height / 2}" fill="${labelColor}" font-size="${labelSize}" text-anchor="middle" ` +
      `transform="rotate(-90 15 ${height / 2})">Actual Success</text>`
    );

    return `
      <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="background: #0a0a0a;">
        ${elements.join('\n')}
      </svg>
    `;
  }

  /**
   * Render empty state.
   * @private
   */
  static _renderEmpty(width, height, padding) {
    return `
      <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="background: #0a0a0a;">
        <text x="${width / 2}" y="${height / 2}" fill="#666" font-size="11px" text-anchor="middle">
          No calibration data
        </text>
      </svg>
    `;
  }
}
