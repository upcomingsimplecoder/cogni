/**
 * widgets/radar-chart.js
 * 6-axis radar chart for agent capability self-model.
 */

import { withAlpha } from '../core/colors.js';

export class RadarChart {
  /**
   * Render a radar chart as SVG.
   * @param {Object} data - {gathering: 0.8, social: 0.4, threat_assessment: 0.6, exploration: 0.5, combat: 0.3, planning: 0.7}
   * @param {Object} options
   * @param {number} options.size - SVG size (square, default 180)
   * @param {string} options.fillColor - Area fill color (default '#06b6d4')
   * @param {number} options.fillAlpha - Fill opacity (default 0.3)
   * @param {string} options.strokeColor - Outline color (default '#06b6d4')
   * @param {boolean} options.showLabels - Show axis labels (default true)
   * @param {boolean} options.showValues - Show value at each vertex (default false)
   * @param {Array<number>} options.rings - Concentric ring values (default [0.25, 0.5, 0.75, 1.0])
   * @returns {string} SVG HTML string
   */
  static render(data, options = {}) {
    const size = options.size ?? 180;
    const fillColor = options.fillColor ?? '#06b6d4';
    const fillAlpha = options.fillAlpha ?? 0.3;
    const strokeColor = options.strokeColor ?? '#06b6d4';
    const showLabels = options.showLabels ?? true;
    const showValues = options.showValues ?? false;
    const rings = options.rings ?? [0.25, 0.5, 0.75, 1.0];

    // Axes in order
    const axes = [
      { key: 'gathering', label: 'Gathering' },
      { key: 'social', label: 'Social' },
      { key: 'threat_assessment', label: 'Threat' },
      { key: 'exploration', label: 'Exploration' },
      { key: 'combat', label: 'Combat' },
      { key: 'planning', label: 'Planning' },
    ];

    // Center and radius
    const center = size / 2;
    const padding = showLabels ? 35 : 15;
    const radius = center - padding;

    // Handle missing or invalid data
    const values = axes.map(axis => {
      const val = data[axis.key];
      if (val == null || isNaN(val)) return 0;
      return Math.max(0, Math.min(1, val)); // Clamp to [0, 1]
    });

    // Check if all values are zero
    const allZero = values.every(v => v === 0);

    // Calculate angle for each axis (start from top, clockwise)
    const angleStep = (2 * Math.PI) / axes.length;
    const startAngle = -Math.PI / 2; // Start at top

    /**
     * Get point coordinates for a given axis and value.
     * @param {number} axisIndex - Axis index
     * @param {number} value - Value (0-1)
     * @returns {{x: number, y: number}}
     */
    const getPoint = (axisIndex, value) => {
      const angle = startAngle + axisIndex * angleStep;
      const distance = radius * value;
      return {
        x: center + distance * Math.cos(angle),
        y: center + distance * Math.sin(angle),
      };
    };

    const elements = [];

    // Draw concentric rings
    const ringColor = '#2a2a2a';
    for (const ringValue of rings) {
      const points = axes.map((_, i) => getPoint(i, ringValue));
      const pathData = points.map((p, i) =>
        i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`
      ).join(' ') + ' Z';

      elements.push(
        `<path d="${pathData}" stroke="${ringColor}" stroke-width="1" fill="none" opacity="0.5"/>`
      );
    }

    // Draw axis lines
    const axisColor = '#444';
    for (let i = 0; i < axes.length; i++) {
      const endpoint = getPoint(i, 1);
      elements.push(
        `<line x1="${center}" y1="${center}" x2="${endpoint.x}" y2="${endpoint.y}" ` +
        `stroke="${axisColor}" stroke-width="1" opacity="0.6"/>`
      );
    }

    // Draw data polygon (only if not all zeros)
    if (!allZero) {
      const dataPoints = values.map((val, i) => getPoint(i, val));
      const pathData = dataPoints.map((p, i) =>
        i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`
      ).join(' ') + ' Z';

      // Fill
      elements.push(
        `<path d="${pathData}" fill="${withAlpha(fillColor, fillAlpha)}" stroke="none"/>`
      );

      // Outline
      elements.push(
        `<path d="${pathData}" fill="none" stroke="${strokeColor}" stroke-width="2" opacity="0.9"/>`
      );

      // Vertex dots
      for (const point of dataPoints) {
        elements.push(
          `<circle cx="${point.x}" cy="${point.y}" r="3" fill="${strokeColor}" stroke="#0a0a0a" stroke-width="1"/>`
        );
      }
    }

    // Draw labels
    if (showLabels) {
      const labelColor = '#ccc';
      const labelSize = '10px';
      const labelDistance = radius + 20; // Beyond the edge

      for (let i = 0; i < axes.length; i++) {
        const angle = startAngle + i * angleStep;
        const labelPoint = {
          x: center + labelDistance * Math.cos(angle),
          y: center + labelDistance * Math.sin(angle),
        };

        // Adjust text anchor based on position
        let anchor = 'middle';
        if (Math.abs(Math.cos(angle)) > 0.5) {
          anchor = Math.cos(angle) > 0 ? 'start' : 'end';
        }

        elements.push(
          `<text x="${labelPoint.x}" y="${labelPoint.y + 4}" fill="${labelColor}" ` +
          `font-size="${labelSize}" text-anchor="${anchor}">${axes[i].label}</text>`
        );

        // Optionally show values
        if (showValues && !allZero) {
          const valueText = values[i].toFixed(2);
          const valuePoint = {
            x: center + (labelDistance - 10) * Math.cos(angle),
            y: center + (labelDistance - 10) * Math.sin(angle),
          };

          elements.push(
            `<text x="${valuePoint.x}" y="${valuePoint.y + 4}" fill="#888" ` +
            `font-size="9px" text-anchor="${anchor}">${valueText}</text>`
          );
        }
      }
    }

    // Center dot
    elements.push(
      `<circle cx="${center}" cy="${center}" r="2" fill="${axisColor}" opacity="0.8"/>`
    );

    return `
      <svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg" style="background: #0a0a0a;">
        ${elements.join('\n')}
      </svg>
    `;
  }
}
