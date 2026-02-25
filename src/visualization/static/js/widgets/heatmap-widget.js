/**
 * widgets/heatmap-widget.js
 * NxN matrix heatmap (trust, lexicon similarity).
 */

import { interpolate } from '../core/colors.js';

/**
 * HeatmapWidget for rendering NxN matrix data as color-coded tables.
 */
export class HeatmapWidget {
  /**
   * Render an NxN heatmap as an HTML table.
   * @param {Object} data
   * @param {Array<string>} data.labels - Row/column labels
   * @param {Array<Array<number>>} data.matrix - NxN values (0-1)
   * @param {Object} options
   * @param {string} options.colorLow - Color for value 0 (default '#1a1a1a')
   * @param {string} options.colorHigh - Color for value 1 (default '#22c55e')
   * @param {number} options.cellSize - Cell size in px (default 24)
   * @param {boolean} options.showValues - Show value text in cells (default true for N≤8)
   * @param {string} options.title - Optional title
   * @param {Function} options.onCellClick - Optional (row, col) callback
   * @param {number} options.maxLabelLength - Max label characters (default 12)
   * @returns {string} HTML string
   */
  static render(data, options = {}) {
    const {
      colorLow = '#1a1a1a',
      colorHigh = '#22c55e',
      cellSize = 24,
      showValues = null,
      title = null,
      onCellClick = null,
      maxLabelLength = 12
    } = options;

    const { labels, matrix } = data;

    // Edge cases
    if (!labels || labels.length === 0 || !matrix || matrix.length === 0) {
      return `<div style="color: #666; font-size: 10px; padding: 8px;">No data</div>`;
    }

    const N = labels.length;

    // Auto-determine showValues based on size
    const shouldShowValues = showValues !== null ? showValues : N <= 8;

    // Truncate labels
    const truncate = (str, max) => {
      if (str.length <= max) return str;
      return str.substring(0, max - 1) + '…';
    };

    const truncatedLabels = labels.map(l => truncate(l, maxLabelLength));

    // Determine if scrollable container needed
    const isLarge = N > 10;
    const tableWidth = cellSize * (N + 1); // +1 for row labels
    const maxHeight = 400;

    // Build HTML
    let html = '<div style="font-family: monospace; color: #ccc;">';

    // Title
    if (title) {
      html += `<div style="font-size: 12px; font-weight: 600; margin-bottom: 8px; color: #fff;">${title}</div>`;
    }

    // Legend
    html += `<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; font-size: 9px;">
      <span style="color: #888;">Low</span>
      <div style="width: 100px; height: 12px; background: linear-gradient(to right, ${colorLow}, ${colorHigh}); border-radius: 2px;"></div>
      <span style="color: #888;">High</span>
    </div>`;

    // Container (scrollable if needed)
    const containerStyle = isLarge
      ? `max-width: 100%; max-height: ${maxHeight}px; overflow: auto; border: 1px solid #333; border-radius: 4px;`
      : '';

    html += `<div style="${containerStyle}">`;

    // Table
    html += `<table style="border-collapse: collapse; background: #0a0a0a;">`;

    // Header row (column labels)
    html += '<tr><td style="width: ' + cellSize + 'px; height: ' + cellSize + 'px;"></td>';
    for (let col = 0; col < N; col++) {
      html += `<td style="width: ${cellSize}px; height: ${cellSize}px; padding: 2px; font-size: 9px; text-align: center; vertical-align: middle; color: #888; transform: rotate(-45deg); transform-origin: center; white-space: nowrap;">
        <div style="width: ${cellSize}px; overflow: hidden; text-overflow: ellipsis;">${truncatedLabels[col]}</div>
      </td>`;
    }
    html += '</tr>';

    // Data rows
    for (let row = 0; row < N; row++) {
      html += '<tr>';

      // Row label
      html += `<td style="width: ${cellSize}px; height: ${cellSize}px; padding: 2px; font-size: 9px; text-align: right; vertical-align: middle; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
        ${truncatedLabels[row]}
      </td>`;

      // Cells
      for (let col = 0; col < N; col++) {
        const value = matrix[row][col];
        const isDiagonal = row === col;

        // Interpolate color
        let bgColor;
        if (isDiagonal) {
          // Diagonal cells slightly different (self-reference)
          bgColor = '#2a2a2a';
        } else if (value === null || value === undefined || isNaN(value)) {
          // Missing data
          bgColor = '#0f0f0f';
        } else {
          // Clamp value to 0-1
          const clampedValue = Math.max(0, Math.min(1, value));
          bgColor = interpolate(colorLow, colorHigh, clampedValue);
        }

        // Text color (light on dark, dark on light)
        const textColor = value > 0.6 ? '#000' : '#fff';

        // Cell content
        let cellContent = '';
        if (shouldShowValues && value !== null && value !== undefined && !isNaN(value)) {
          cellContent = value.toFixed(2);
        } else if (isDiagonal) {
          cellContent = '—';
        }

        // Click handler (if provided, we'd need to register event listener after render)
        const cellId = onCellClick ? `heatmap-cell-${row}-${col}` : '';
        const dataAttrs = onCellClick ? `data-row="${row}" data-col="${col}"` : '';

        html += `<td id="${cellId}" ${dataAttrs} style="width: ${cellSize}px; height: ${cellSize}px; background: ${bgColor}; color: ${textColor}; font-size: 8px; text-align: center; vertical-align: middle; border: 1px solid #0a0a0a; cursor: ${onCellClick ? 'pointer' : 'default'};">
          ${cellContent}
        </td>`;
      }

      html += '</tr>';
    }

    html += '</table>';
    html += '</div>'; // Close container

    // Stats
    const allValues = matrix.flat().filter(v => v !== null && v !== undefined && !isNaN(v));
    if (allValues.length > 0) {
      const min = Math.min(...allValues);
      const max = Math.max(...allValues);
      const avg = allValues.reduce((a, b) => a + b, 0) / allValues.length;

      html += `<div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 9px; color: #666;">
        <span>min: ${min.toFixed(3)}</span>
        <span>avg: ${avg.toFixed(3)}</span>
        <span>max: ${max.toFixed(3)}</span>
      </div>`;
    }

    html += '</div>'; // Close main container

    return html;
  }

  /**
   * Attach click handlers after rendering (call after inserting HTML into DOM).
   * @param {HTMLElement} container - Container element with rendered heatmap
   * @param {Function} onCellClick - Callback (row, col) => void
   */
  static attachClickHandlers(container, onCellClick) {
    if (!onCellClick) return;

    const cells = container.querySelectorAll('td[data-row][data-col]');
    cells.forEach(cell => {
      cell.addEventListener('click', () => {
        const row = parseInt(cell.getAttribute('data-row'), 10);
        const col = parseInt(cell.getAttribute('data-col'), 10);
        onCellClick(row, col);
      });
    });
  }

  /**
   * Render a symmetric heatmap (e.g., similarity matrix).
   * Only renders upper triangle to reduce visual clutter.
   * @param {Object} data
   * @param {Array<string>} data.labels
   * @param {Array<Array<number>>} data.matrix - Symmetric NxN matrix
   * @param {Object} options - Same as render()
   * @returns {string} HTML string
   */
  static renderSymmetric(data, options = {}) {
    // Clone matrix and zero out lower triangle
    const { labels, matrix } = data;
    const N = labels.length;
    const upperMatrix = matrix.map((row, i) =>
      row.map((val, j) => (j >= i ? val : null))
    );

    return this.render({ labels, matrix: upperMatrix }, {
      ...options,
      showValues: options.showValues !== null ? options.showValues : N <= 10
    });
  }

  /**
   * Render an asymmetric heatmap with row→column semantics (e.g., trust: A trusts B).
   * Highlights asymmetry with color coding.
   * @param {Object} data
   * @param {Array<string>} data.labels
   * @param {Array<Array<number>>} data.matrix
   * @param {Object} options
   * @returns {string} HTML string
   */
  static renderAsymmetric(data, options = {}) {
    // Add visual indicator for asymmetric relationships
    const { labels, matrix } = data;
    const N = labels.length;

    // Calculate asymmetry score for each pair
    const asymmetryMatrix = matrix.map((row, i) =>
      row.map((val, j) => {
        if (i === j) return 0; // Diagonal
        const diff = Math.abs(val - matrix[j][i]);
        return diff;
      })
    );

    // Render with asymmetry overlay (future enhancement)
    return this.render(data, {
      ...options,
      title: options.title || 'Asymmetric Relationships (row → column)'
    });
  }
}
