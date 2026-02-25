/**
 * widgets/genealogy-tree.js
 * Symbol genealogy tree showing cultural innovation and adoption.
 */

import { BIAS_COLORS, withAlpha } from '../core/colors.js';

export class GenealogyTree {
  /**
   * Render a horizontal genealogy tree as SVG.
   * @param {Object} data
   * @param {Array<{id: string, form: string, meaning: string, creator: string, tick_created: number, alive: boolean}>} data.symbols
   * @param {Array<{symbol_id: string, adopter: string, tick: number, bias_type: string}>} data.adoptions
   * @param {Object} options
   * @param {number} options.width - SVG width (default 350)
   * @param {number} options.height - SVG height (default 200)
   * @param {number} options.maxTick - Maximum tick for X-axis scale
   * @returns {string} SVG HTML string
   */
  static render(data, options = {}) {
    const width = options.width || 350;
    const height = options.height || 200;
    const padding = { top: 20, right: 40, bottom: 30, left: 40 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Handle edge cases
    if (!data || !data.symbols || data.symbols.length === 0) {
      return this._renderEmpty(width, height, 'No symbols to display');
    }

    const symbols = data.symbols;
    const adoptions = data.adoptions || [];

    // Determine time scale
    const maxTickData = Math.max(...symbols.map(s => s.tick_created));
    const maxAdoptionTick = adoptions.length > 0
      ? Math.max(...adoptions.map(a => a.tick))
      : maxTickData;
    const maxTick = options.maxTick || Math.max(maxTickData, maxAdoptionTick);
    const timeScale = (tick) => padding.left + (tick / maxTick) * plotWidth;

    // Build adoption tree structure
    const symbolMap = new Map(symbols.map(s => [s.id, s]));
    const adoptionsBySymbol = new Map();

    adoptions.forEach(adoption => {
      if (!adoptionsBySymbol.has(adoption.symbol_id)) {
        adoptionsBySymbol.set(adoption.symbol_id, []);
      }
      adoptionsBySymbol.get(adoption.symbol_id).push(adoption);
    });

    // Layout: space symbols vertically
    const symbolSpacing = plotHeight / Math.max(symbols.length, 1);
    const symbolPositions = new Map();

    symbols.forEach((symbol, idx) => {
      symbolPositions.set(symbol.id, {
        x: timeScale(symbol.tick_created),
        y: padding.top + symbolSpacing * (idx + 0.5),
        symbol
      });
    });

    // Build SVG elements
    let svgContent = '';

    // Draw adoption branches
    adoptions.forEach(adoption => {
      const symbolPos = symbolPositions.get(adoption.symbol_id);
      if (!symbolPos) return;

      const x1 = symbolPos.x;
      const y1 = symbolPos.y;
      const x2 = timeScale(adoption.tick);
      const y2 = y1 + (Math.random() - 0.5) * symbolSpacing * 0.3; // Slight vertical jitter

      const biasColor = this._getBiasColor(adoption.bias_type);
      const opacity = symbolPos.symbol.alive ? 0.7 : 0.3;

      // Curved path
      const midX = (x1 + x2) / 2;
      svgContent += `<path d="M ${x1} ${y1} Q ${midX} ${y1}, ${midX} ${y2} T ${x2} ${y2}" ` +
        `stroke="${biasColor}" stroke-width="2" fill="none" opacity="${opacity}" />`;

      // Adoption endpoint
      svgContent += `<circle cx="${x2}" cy="${y2}" r="3" fill="${biasColor}" opacity="${opacity}" />`;
    });

    // Draw symbol nodes (on top of branches)
    symbolPositions.forEach(({ x, y, symbol }) => {
      const opacity = symbol.alive ? 1 : 0.4;
      const nodeColor = symbol.alive ? '#4a9eff' : '#666';

      svgContent += `<circle cx="${x}" cy="${y}" r="6" fill="${nodeColor}" opacity="${opacity}" ` +
        `stroke="#fff" stroke-width="1.5" />`;

      // Label with form
      const labelX = x;
      const labelY = y - 10;
      const fontSize = symbol.alive ? 11 : 9;
      const textColor = symbol.alive ? '#fff' : '#888';

      svgContent += `<text x="${labelX}" y="${labelY}" ` +
        `font-size="${fontSize}" fill="${textColor}" text-anchor="middle" ` +
        `font-family="Inter, system-ui, sans-serif">${this._escapeHtml(symbol.form)}</text>`;
    });

    // Draw time axis
    const axisY = height - padding.bottom + 15;
    svgContent += `<line x1="${padding.left}" y1="${axisY}" x2="${width - padding.right}" y2="${axisY}" ` +
      `stroke="#444" stroke-width="1" />`;

    // Time labels
    const tickCount = 5;
    for (let i = 0; i <= tickCount; i++) {
      const tick = (maxTick / tickCount) * i;
      const x = timeScale(tick);
      svgContent += `<text x="${x}" y="${axisY + 12}" font-size="9" fill="#888" ` +
        `text-anchor="middle" font-family="Inter, system-ui, sans-serif">${Math.round(tick)}</text>`;
    }

    // Legend for bias types (if adoptions exist)
    if (adoptions.length > 0) {
      const legend = this._renderLegend();
      svgContent += `<g transform="translate(${width - padding.right - 10}, ${padding.top})">${legend}</g>`;
    }

    return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="${width}" height="${height}" fill="#0a0a0a" />
      ${svgContent}
    </svg>`;
  }

  /**
   * Get color for bias type
   */
  static _getBiasColor(biasType) {
    const biasKey = biasType ? biasType.toLowerCase() : 'unbiased';

    if (biasKey.includes('prestige')) return BIAS_COLORS.prestige;
    if (biasKey.includes('conformist')) return BIAS_COLORS.conformist;
    if (biasKey.includes('content')) return BIAS_COLORS.content;
    if (biasKey.includes('anti')) return BIAS_COLORS.antiConformist;

    return BIAS_COLORS.unbiased;
  }

  /**
   * Render legend for bias types
   */
  static _renderLegend() {
    const biasTypes = [
      { label: 'Prestige', color: BIAS_COLORS.prestige },
      { label: 'Conform', color: BIAS_COLORS.conformist },
      { label: 'Content', color: BIAS_COLORS.content },
      { label: 'Anti', color: BIAS_COLORS.antiConformist }
    ];

    let legendContent = '';
    biasTypes.forEach((bias, idx) => {
      const y = idx * 14;
      legendContent += `<circle cx="0" cy="${y}" r="3" fill="${bias.color}" />`;
      legendContent += `<text x="8" y="${y + 4}" font-size="9" fill="#ccc" ` +
        `font-family="Inter, system-ui, sans-serif">${bias.label}</text>`;
    });

    return legendContent;
  }

  /**
   * Render empty state
   */
  static _renderEmpty(width, height, message) {
    return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="${width}" height="${height}" fill="#0a0a0a" />
      <text x="${width / 2}" y="${height / 2}"
        font-size="11" fill="#666" text-anchor="middle"
        font-family="Inter, system-ui, sans-serif">${message}</text>
    </svg>`;
  }

  /**
   * Escape HTML entities
   */
  static _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}
