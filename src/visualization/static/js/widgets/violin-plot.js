/**
 * widgets/violin-plot.js
 * Violin/distribution plot for trait distributions across agents.
 */

import { withAlpha } from '../core/colors.js';

export class ViolinPlot {
  /**
   * Render a set of mini-violin plots as SVG.
   * @param {Object} data - {traitName: Array<number>} mapping trait names to value arrays
   * @param {Object} options
   * @param {number} options.width - Total SVG width (default 250)
   * @param {number} options.violinHeight - Height per violin (default 30)
   * @param {number} options.gap - Gap between violins (default 4)
   * @param {Object} options.colors - {traitName: color} mapping
   * @param {boolean} options.showDots - Overlay individual agent dots (beeswarm, default true)
   * @param {number} options.dotRadius - Dot radius (default 2)
   * @param {Array<number>} options.range - [min, max] value range (default [0, 1])
   * @param {number} options.bins - Number of histogram bins for KDE (default 10)
   * @returns {string} SVG HTML string
   */
  static render(data, options = {}) {
    const width = options.width ?? 250;
    const violinHeight = options.violinHeight ?? 30;
    const gap = options.gap ?? 4;
    const colors = options.colors ?? {};
    const showDots = options.showDots ?? true;
    const dotRadius = options.dotRadius ?? 2;
    const range = options.range ?? [0, 1];
    const bins = options.bins ?? 10;

    // Default color
    const defaultColor = '#06b6d4';

    // Filter valid traits
    const traits = Object.entries(data).filter(([_, values]) =>
      Array.isArray(values) && values.length > 0
    );

    if (traits.length === 0) {
      return this._renderEmpty(width, violinHeight);
    }

    // Calculate total height
    const padding = { top: 10, right: 10, bottom: 10, left: 80 };
    const totalViolinHeight = traits.length * violinHeight + (traits.length - 1) * gap;
    const height = totalViolinHeight + padding.top + padding.bottom;

    // Scale functions
    const plotWidth = width - padding.left - padding.right;
    const [minVal, maxVal] = range;
    const scaleX = (val) => padding.left + ((val - minVal) / (maxVal - minVal)) * plotWidth;

    const elements = [];

    // Background grid lines
    const gridColor = '#2a2a2a';
    for (let i = 0; i <= 4; i++) {
      const val = minVal + (i / 4) * (maxVal - minVal);
      const x = scaleX(val);
      elements.push(
        `<line x1="${x}" y1="${padding.top}" x2="${x}" y2="${height - padding.bottom}" ` +
        `stroke="${gridColor}" stroke-width="1" opacity="0.3"/>`
      );
    }

    // Render each violin
    for (let traitIndex = 0; traitIndex < traits.length; traitIndex++) {
      const [traitName, values] = traits[traitIndex];
      const color = colors[traitName] ?? defaultColor;

      // Y position for this violin
      const yCenter = padding.top + traitIndex * (violinHeight + gap) + violinHeight / 2;

      // Filter valid values
      const validValues = values.filter(v => v != null && !isNaN(v));

      if (validValues.length === 0) {
        // Draw empty violin
        elements.push(...this._renderEmptyViolin(traitName, yCenter, padding.left, plotWidth, violinHeight));
        continue;
      }

      // Calculate statistics
      const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
      const allSame = validValues.every(v => v === validValues[0]);

      // Build histogram
      const histogram = this._buildHistogram(validValues, bins, minVal, maxVal);
      const maxCount = Math.max(...histogram.map(b => b.count));

      // Violin shape (symmetric, mirrored histogram)
      if (maxCount > 0 && !allSame) {
        const violinPoints = histogram.map(bin => {
          const x = scaleX(bin.center);
          const heightFactor = bin.count / maxCount;
          const halfHeight = (violinHeight / 2 - 2) * heightFactor; // Leave 2px margin
          return { x, halfHeight };
        });

        // Create path: top half, then bottom half (reversed)
        const topPath = violinPoints.map((p, i) => {
          const y = yCenter - p.halfHeight;
          return i === 0 ? `M ${p.x} ${yCenter}` : `L ${p.x} ${y}`;
        }).join(' ');

        const bottomPath = violinPoints.slice().reverse().map(p => {
          const y = yCenter + p.halfHeight;
          return `L ${p.x} ${y}`;
        }).join(' ');

        const pathData = topPath + bottomPath + ' Z';

        elements.push(
          `<path d="${pathData}" fill="${withAlpha(color, 0.4)}" ` +
          `stroke="${color}" stroke-width="1" opacity="0.8"/>`
        );
      } else if (allSame) {
        // All same value: draw a thin vertical line
        const x = scaleX(validValues[0]);
        const barHeight = violinHeight - 4;
        elements.push(
          `<line x1="${x}" y1="${yCenter - barHeight / 2}" x2="${x}" y2="${yCenter + barHeight / 2}" ` +
          `stroke="${color}" stroke-width="3" opacity="0.8"/>`
        );
      }

      // Overlay individual dots (beeswarm)
      if (showDots && validValues.length > 0) {
        const dots = this._beeswarm(validValues, yCenter, violinHeight, scaleX, dotRadius);
        for (const dot of dots) {
          elements.push(
            `<circle cx="${dot.x}" cy="${dot.y}" r="${dotRadius}" ` +
            `fill="${color}" stroke="#0a0a0a" stroke-width="0.5" opacity="0.6"/>`
          );
        }
      }

      // Mean line
      const meanX = scaleX(mean);
      elements.push(
        `<line x1="${meanX}" y1="${yCenter - violinHeight / 2 + 2}" ` +
        `x2="${meanX}" y2="${yCenter + violinHeight / 2 - 2}" ` +
        `stroke="#fff" stroke-width="1.5" opacity="0.7"/>`
      );

      // Trait label (left)
      const labelColor = '#ccc';
      elements.push(
        `<text x="${padding.left - 10}" y="${yCenter + 3}" fill="${labelColor}" ` +
        `font-size="10px" text-anchor="end">${this._formatLabel(traitName)}</text>`
      );

      // Mean value (right)
      elements.push(
        `<text x="${width - padding.right + 5}" y="${yCenter + 3}" fill="#888" ` +
        `font-size="9px" text-anchor="start">${mean.toFixed(2)}</text>`
      );
    }

    // X-axis labels at bottom
    const labelColor = '#888';
    for (let i = 0; i <= 4; i++) {
      const val = minVal + (i / 4) * (maxVal - minVal);
      const x = scaleX(val);
      elements.push(
        `<text x="${x}" y="${height - 2}" fill="${labelColor}" ` +
        `font-size="9px" text-anchor="middle">${val.toFixed(1)}</text>`
      );
    }

    return `
      <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="background: #0a0a0a;">
        ${elements.join('\n')}
      </svg>
    `;
  }

  /**
   * Build histogram from values.
   * @private
   */
  static _buildHistogram(values, numBins, minVal, maxVal) {
    const binWidth = (maxVal - minVal) / numBins;
    const bins = Array.from({ length: numBins }, (_, i) => ({
      min: minVal + i * binWidth,
      max: minVal + (i + 1) * binWidth,
      center: minVal + (i + 0.5) * binWidth,
      count: 0,
    }));

    for (const val of values) {
      const binIndex = Math.min(
        Math.floor((val - minVal) / binWidth),
        numBins - 1
      );
      if (binIndex >= 0 && binIndex < numBins) {
        bins[binIndex].count++;
      }
    }

    return bins;
  }

  /**
   * Create beeswarm layout for dots.
   * @private
   */
  static _beeswarm(values, yCenter, violinHeight, scaleX, dotRadius) {
    // Simple beeswarm: group values into small x-bins, then stack vertically with jitter
    const dots = [];
    const valueBinWidth = 0.05; // Group values within 0.05 range

    // Sort values
    const sorted = [...values].sort((a, b) => a - b);

    // Group into bins
    const bins = [];
    for (const val of sorted) {
      const x = scaleX(val);

      // Find or create bin
      let bin = bins.find(b => Math.abs(b.x - x) < scaleX(valueBinWidth));
      if (!bin) {
        bin = { x, values: [] };
        bins.push(bin);
      }
      bin.values.push(val);
    }

    // Layout dots in each bin
    const maxDotsPerColumn = Math.floor(violinHeight / (dotRadius * 2.5));

    for (const bin of bins) {
      const count = bin.values.length;
      const columns = Math.ceil(count / maxDotsPerColumn);

      for (let i = 0; i < count; i++) {
        const col = Math.floor(i / maxDotsPerColumn);
        const row = i % maxDotsPerColumn;

        // Vertical position with slight jitter
        const yRange = Math.min(violinHeight - dotRadius * 4, maxDotsPerColumn * dotRadius * 2.5);
        const yOffset = (row - maxDotsPerColumn / 2) * (yRange / maxDotsPerColumn);
        const yJitter = (Math.random() - 0.5) * dotRadius;

        // Horizontal jitter for multiple columns
        const xJitter = (col - (columns - 1) / 2) * dotRadius * 2.5;

        dots.push({
          x: bin.x + xJitter,
          y: yCenter + yOffset + yJitter,
        });
      }
    }

    return dots;
  }

  /**
   * Format trait label (convert snake_case to Title Case).
   * @private
   */
  static _formatLabel(traitName) {
    return traitName
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  /**
   * Render empty violin.
   * @private
   */
  static _renderEmptyViolin(traitName, yCenter, xStart, width, height) {
    const elements = [];
    const labelColor = '#666';

    // Trait label
    elements.push(
      `<text x="${xStart - 10}" y="${yCenter + 3}" fill="${labelColor}" ` +
      `font-size="10px" text-anchor="end">${this._formatLabel(traitName)}</text>`
    );

    // "No data" indicator
    elements.push(
      `<text x="${xStart + width / 2}" y="${yCenter + 3}" fill="${labelColor}" ` +
      `font-size="9px" text-anchor="middle" opacity="0.5">No data</text>`
    );

    return elements;
  }

  /**
   * Render empty state.
   * @private
   */
  static _renderEmpty(width, height) {
    return `
      <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="background: #0a0a0a;">
        <text x="${width / 2}" y="${height / 2}" fill="#666" font-size="11px" text-anchor="middle">
          No trait data
        </text>
      </svg>
    `;
  }
}
