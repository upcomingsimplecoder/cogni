/**
 * widgets/sankey-widget.js
 * Simple Sankey-style diagram for strategy transitions.
 */

export class SankeyWidget {
  /**
   * Render a simplified Sankey/flow diagram as SVG.
   * @param {Object} data
   * @param {Array<{from: string, to: string, count: number}>} data.flows - Transition flows
   * @param {Object} options
   * @param {number} options.width - SVG width (default 250)
   * @param {number} options.height - SVG height (default 150)
   * @param {Object} options.colors - {nodeName: color} mapping
   * @returns {string} SVG HTML string
   */
  static render(data, options = {}) {
    const width = options.width || 250;
    const height = options.height || 150;
    const padding = { top: 20, right: 10, bottom: 20, left: 10 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Handle edge cases
    if (!data || !data.flows || data.flows.length === 0) {
      return this._renderEmpty(width, height, 'No transitions');
    }

    const flows = data.flows;
    const colors = options.colors || {};

    // Build node sets and calculate totals
    const sourceNodes = new Map(); // node -> total outgoing
    const targetNodes = new Map(); // node -> total incoming
    let totalFlow = 0;

    flows.forEach(flow => {
      sourceNodes.set(flow.from, (sourceNodes.get(flow.from) || 0) + flow.count);
      targetNodes.set(flow.to, (targetNodes.get(flow.to) || 0) + flow.count);
      totalFlow += flow.count;
    });

    // Handle self-loops (nodes that are both source and target)
    const allNodes = new Set([...sourceNodes.keys(), ...targetNodes.keys()]);
    const leftNodes = [];
    const rightNodes = [];

    allNodes.forEach(node => {
      if (sourceNodes.has(node)) leftNodes.push(node);
      if (targetNodes.has(node)) rightNodes.push(node);
    });

    // Remove duplicates (self-loops appear on both sides)
    const uniqueLeftNodes = [...new Set(leftNodes)];
    const uniqueRightNodes = [...new Set(rightNodes)];

    // Calculate node positions and heights
    const leftColumn = padding.left + 40;
    const rightColumn = width - padding.right - 40;

    const calculateNodeLayout = (nodes, totals, x) => {
      const layout = [];
      let currentY = padding.top;

      nodes.forEach(node => {
        const total = totals.get(node) || 1;
        const nodeHeight = Math.max(10, (total / totalFlow) * plotHeight * 0.8);

        layout.push({
          name: node,
          x,
          y: currentY,
          height: nodeHeight,
          total
        });

        currentY += nodeHeight + 5; // 5px spacing
      });

      return layout;
    };

    const leftLayout = calculateNodeLayout(uniqueLeftNodes, sourceNodes, leftColumn);
    const rightLayout = calculateNodeLayout(uniqueRightNodes, targetNodes, rightColumn);

    // Build lookup maps
    const leftMap = new Map(leftLayout.map(n => [n.name, n]));
    const rightMap = new Map(rightLayout.map(n => [n.name, n]));

    // Build SVG
    let svgContent = '';

    // Draw flows (curves connecting nodes)
    flows.forEach(flow => {
      const source = leftMap.get(flow.from);
      const target = rightMap.get(flow.to);

      if (!source || !target) return;

      const x1 = source.x + 35; // Right edge of source box
      const y1 = source.y + source.height / 2;
      const x2 = target.x - 5; // Left edge of target box
      const y2 = target.y + target.height / 2;

      const flowWidth = Math.max(2, (flow.count / totalFlow) * 20);
      const sourceColor = this._getNodeColor(flow.from, colors);

      // Bezier curve
      const midX = (x1 + x2) / 2;
      const path = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;

      svgContent += `<path d="${path}" stroke="${sourceColor}" stroke-width="${flowWidth}" ` +
        `fill="none" opacity="0.5" />`;
    });

    // Draw left nodes
    leftLayout.forEach(node => {
      const nodeColor = this._getNodeColor(node.name, colors);
      svgContent += this._renderNode(node, nodeColor, 'end');
    });

    // Draw right nodes
    rightLayout.forEach(node => {
      const nodeColor = this._getNodeColor(node.name, colors);
      svgContent += this._renderNode(node, nodeColor, 'start');
    });

    return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="${width}" height="${height}" fill="#0a0a0a" />
      ${svgContent}
    </svg>`;
  }

  /**
   * Render a single node box with label
   */
  static _renderNode(node, color, labelAnchor) {
    const boxWidth = 35;
    const labelOffset = labelAnchor === 'end' ? -5 : boxWidth + 5;
    const labelX = node.x + labelOffset;
    const labelY = node.y + node.height / 2 + 4;

    let html = '';

    // Box
    html += `<rect x="${node.x}" y="${node.y}" width="${boxWidth}" height="${node.height}" ` +
      `fill="${color}" rx="2" opacity="0.8" />`;

    // Label
    html += `<text x="${labelX}" y="${labelY}" font-size="10" fill="#ccc" ` +
      `text-anchor="${labelAnchor}" font-family="Inter, system-ui, sans-serif">${this._escapeHtml(node.name)}</text>`;

    // Count label (on box)
    const countX = node.x + boxWidth / 2;
    const countY = node.y + node.height / 2 + 3;

    if (node.height > 15) {
      html += `<text x="${countX}" y="${countY}" font-size="9" fill="#fff" ` +
        `text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-weight="600">${node.total}</text>`;
    }

    return html;
  }

  /**
   * Get color for node (with fallback)
   */
  static _getNodeColor(nodeName, colorMap) {
    if (colorMap[nodeName]) return colorMap[nodeName];

    // Default colors based on strategy name
    const defaults = {
      'EXPLORE': '#4a9eff',
      'EXPLOIT': '#ffa94d',
      'COOPERATE': '#51cf66',
      'DEFECT': '#ff6b6b',
      'SHARE': '#a78bfa',
      'HOARD': '#f59e0b'
    };

    const upperName = nodeName.toUpperCase();
    for (const [key, color] of Object.entries(defaults)) {
      if (upperName.includes(key)) return color;
    }

    return '#888'; // Fallback gray
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
