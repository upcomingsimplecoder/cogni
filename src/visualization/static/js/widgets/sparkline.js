/**
 * widgets/sparkline.js
 * Reusable SVG sparkline for inline metric visualization.
 */

/**
 * Sparkline widget for rendering time series data as compact SVG visualizations.
 */
export class Sparkline {
  /**
   * Render an SVG sparkline.
   * @param {Array<number>} values - Data points
   * @param {Object} options
   * @param {number} options.width - SVG width (default 200)
   * @param {number} options.height - SVG height (default 40)
   * @param {string} options.color - Stroke color (default '#22c55e')
   * @param {number} options.strokeWidth - Line width (default 1.5)
   * @param {boolean} options.showArea - Fill area under curve (default false)
   * @param {boolean} options.showDots - Show dots at data points (default false)
   * @param {boolean} options.showMinMax - Label min/max values (default false)
   * @param {string} options.label - Optional label text
   * @returns {string} SVG HTML string
   */
  static render(values, options = {}) {
    const {
      width = 200,
      height = 40,
      color = '#22c55e',
      strokeWidth = 1.5,
      showArea = false,
      showDots = false,
      showMinMax = false,
      label = null
    } = options;

    // Edge cases
    if (!values || values.length === 0) {
      return `<svg width="${width}" height="${height}" style="display:block;">
        <text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#666" font-size="10px">No data</text>
      </svg>`;
    }

    if (values.length === 1) {
      const y = height / 2;
      return `<svg width="${width}" height="${height}" style="display:block;">
        <circle cx="${width/2}" cy="${y}" r="3" fill="${color}"/>
        ${showMinMax ? `<text x="${width/2}" y="${y-8}" text-anchor="middle" fill="#888" font-size="9px">${values[0].toFixed(2)}</text>` : ''}
      </svg>`;
    }

    // Calculate min/max for scaling
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;

    // Handle flat line (all same values)
    const padding = { top: 8, right: 4, bottom: 8, left: 4 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const scale = (v) => {
      if (range === 0) return plotHeight / 2;
      return plotHeight - ((v - min) / range) * plotHeight;
    };

    // Generate points
    const step = plotWidth / (values.length - 1);
    const points = values.map((v, i) => ({
      x: padding.left + i * step,
      y: padding.top + scale(v),
      value: v
    }));

    // Build polyline path
    const pathData = points.map((p, i) =>
      `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)},${p.y.toFixed(2)}`
    ).join(' ');

    // Build area path if needed
    let areaPath = '';
    if (showArea) {
      const bottom = height - padding.bottom;
      areaPath = `M ${points[0].x},${bottom} L ${points[0].x},${points[0].y} ${pathData.substring(1)} L ${points[points.length-1].x},${bottom} Z`;
    }

    // Find min/max point indices
    const minIdx = values.indexOf(min);
    const maxIdx = values.indexOf(max);

    // Build SVG
    let svg = `<svg width="${width}" height="${height}" style="display:block;">`;

    // Area fill
    if (showArea) {
      svg += `<path d="${areaPath}" fill="${color}" fill-opacity="0.15"/>`;
    }

    // Line
    svg += `<path d="${pathData}" stroke="${color}" stroke-width="${strokeWidth}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;

    // Dots
    if (showDots) {
      points.forEach(p => {
        svg += `<circle cx="${p.x}" cy="${p.y}" r="2" fill="${color}"/>`;
      });
    }

    // Min/max labels
    if (showMinMax && range > 0) {
      const minPoint = points[minIdx];
      const maxPoint = points[maxIdx];

      // Max label (above point)
      svg += `<text x="${maxPoint.x}" y="${maxPoint.y - 4}" text-anchor="middle" fill="#888" font-size="9px" font-family="monospace">${max.toFixed(2)}</text>`;

      // Min label (below point)
      svg += `<text x="${minPoint.x}" y="${minPoint.y + 12}" text-anchor="middle" fill="#888" font-size="9px" font-family="monospace">${min.toFixed(2)}</text>`;
    }

    // Label
    if (label) {
      svg += `<text x="4" y="12" fill="#888" font-size="10px">${label}</text>`;
    }

    svg += `</svg>`;
    return svg;
  }

  /**
   * Render a metric sparkline with label and current value.
   * @param {string} label - Metric name
   * @param {Array<{tick: number, value: number}>} series - Time series data
   * @param {Object} options - Sparkline options plus { currentValue, unit, showChange }
   * @returns {string} Full HTML block with label + sparkline + stats
   */
  static renderMetric(label, series, options = {}) {
    const {
      currentValue = null,
      unit = '',
      showChange = true,
      ...sparklineOptions
    } = options;

    // Extract values
    const values = series.map(s => s.value);

    if (values.length === 0) {
      return `<div style="margin-bottom: 16px;">
        <div style="color: #ccc; font-size: 11px; margin-bottom: 4px;">${label}</div>
        <div style="color: #666; font-size: 10px;">No data</div>
      </div>`;
    }

    const current = currentValue !== null ? currentValue : values[values.length - 1];
    const previous = values.length > 1 ? values[values.length - 2] : current;
    const change = current - previous;
    const changePercent = previous !== 0 ? (change / Math.abs(previous)) * 100 : 0;
    const isPositive = change > 0;
    const isNeutral = change === 0;

    // Trend arrow
    let trendArrow = '';
    if (showChange && !isNeutral) {
      const arrowColor = isPositive ? '#22c55e' : '#ef4444';
      const arrowSymbol = isPositive ? '↑' : '↓';
      trendArrow = `<span style="color: ${arrowColor}; font-size: 12px; margin-left: 6px;">${arrowSymbol} ${Math.abs(changePercent).toFixed(1)}%</span>`;
    }

    // Stats
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((a, b) => a + b, 0) / values.length;

    const sparklineSvg = this.render(values, {
      width: 180,
      height: 32,
      showArea: true,
      ...sparklineOptions
    });

    return `<div style="margin-bottom: 16px; font-family: monospace;">
      <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 6px;">
        <span style="color: #ccc; font-size: 11px;">${label}</span>
        <span style="color: #fff; font-size: 13px; font-weight: 600;">${current.toFixed(2)}${unit}${trendArrow}</span>
      </div>
      ${sparklineSvg}
      <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
        <span>min: ${min.toFixed(2)}</span>
        <span>avg: ${avg.toFixed(2)}</span>
        <span>max: ${max.toFixed(2)}</span>
      </div>
    </div>`;
  }
}
