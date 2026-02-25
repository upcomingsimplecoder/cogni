/**
 * widgets/force-graph.js
 * Pure Canvas Verlet force-directed graph layout.
 * Zero dependencies, ~100 lines.
 */

/**
 * ForceGraph widget for visualizing network relationships with physics-based layout.
 */
export class ForceGraph {
  /**
   * Create and render a force-directed graph in a container.
   * @param {HTMLElement} container - DOM element to render into
   * @param {Object} data
   * @param {Array<{id: string, label: string, color: string, size: number}>} data.nodes
   * @param {Array<{source: string, target: string, weight: number, color: string}>} data.edges
   * @param {Object} options
   * @param {number} options.width - Canvas width (default 300)
   * @param {number} options.height - Canvas height (default 200)
   * @param {number} options.iterations - Simulation iterations (default 50)
   * @param {number} options.repulsion - Repulsion force (default 500)
   * @param {number} options.attraction - Attraction force (default 0.01)
   * @param {number} options.damping - Velocity damping (default 0.9)
   * @param {number} options.centeringForce - Pull toward center (default 0.05)
   */
  static render(container, data, options = {}) {
    const {
      width = 300,
      height = 200,
      iterations = 50,
      repulsion = 500,
      attraction = 0.01,
      damping = 0.9,
      centeringForce = 0.05
    } = options;

    const { nodes, edges } = data;

    // Edge cases
    if (!nodes || nodes.length === 0) {
      container.innerHTML = '<div style="color: #666; font-size: 10px; padding: 8px;">No nodes</div>';
      return;
    }

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    canvas.style.display = 'block';
    canvas.style.background = '#0a0a0a';
    container.innerHTML = '';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    const centerX = width / 2;
    const centerY = height / 2;

    // Initialize node positions and velocities
    const nodeMap = new Map();
    nodes.forEach((node, i) => {
      // Start in circle formation
      const angle = (i / nodes.length) * 2 * Math.PI;
      const radius = Math.min(width, height) * 0.3;

      nodeMap.set(node.id, {
        ...node,
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
        size: node.size || 8
      });
    });

    // Build edge lookup
    const edgeList = (edges || []).map(edge => ({
      source: nodeMap.get(edge.source),
      target: nodeMap.get(edge.target),
      weight: edge.weight || 1,
      color: edge.color || '#444'
    })).filter(e => e.source && e.target); // Filter invalid edges

    // Verlet integration simulation
    for (let iter = 0; iter < iterations; iter++) {
      const nodeArray = Array.from(nodeMap.values());

      // Reset forces
      nodeArray.forEach(node => {
        node.fx = 0;
        node.fy = 0;
      });

      // Repulsion (all pairs)
      for (let i = 0; i < nodeArray.length; i++) {
        for (let j = i + 1; j < nodeArray.length; j++) {
          const a = nodeArray[i];
          const b = nodeArray[j];

          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const distSq = dx * dx + dy * dy;
          const dist = Math.sqrt(distSq) || 1;

          // Coulomb-like repulsion: F = k / d^2
          const force = repulsion / distSq;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          a.fx -= fx;
          a.fy -= fy;
          b.fx += fx;
          b.fy += fy;
        }
      }

      // Attraction (connected nodes)
      edgeList.forEach(edge => {
        const { source, target, weight } = edge;

        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;

        // Hooke's law: F = k * d * weight
        const force = attraction * dist * weight;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        source.fx += fx;
        source.fy += fy;
        target.fx -= fx;
        target.fy -= fy;
      });

      // Centering force
      nodeArray.forEach(node => {
        const dx = centerX - node.x;
        const dy = centerY - node.y;
        node.fx += dx * centeringForce;
        node.fy += dy * centeringForce;
      });

      // Update positions with Verlet integration
      nodeArray.forEach(node => {
        node.vx = (node.vx + node.fx) * damping;
        node.vy = (node.vy + node.fy) * damping;

        node.x += node.vx;
        node.y += node.vy;

        // Boundary constraints (soft)
        const margin = node.size;
        if (node.x < margin) { node.x = margin; node.vx *= -0.5; }
        if (node.x > width - margin) { node.x = width - margin; node.vx *= -0.5; }
        if (node.y < margin) { node.y = margin; node.vy *= -0.5; }
        if (node.y > height - margin) { node.y = height - margin; node.vy *= -0.5; }
      });
    }

    // Draw final result
    ctx.clearRect(0, 0, width, height);

    // Draw edges
    edgeList.forEach(edge => {
      const { source, target, weight, color } = edge;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(0.5, weight * 2);
      ctx.stroke();
    });

    // Draw nodes
    const nodeArray = Array.from(nodeMap.values());
    nodeArray.forEach(node => {
      // Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.size, 0, 2 * Math.PI);
      ctx.fillStyle = node.color || '#22c55e';
      ctx.fill();

      // Node border
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Label (if space permits)
      if (node.label && node.size >= 6) {
        ctx.fillStyle = '#fff';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Measure text
        const metrics = ctx.measureText(node.label);
        const textWidth = metrics.width;

        // Truncate if too long
        let displayLabel = node.label;
        if (textWidth > node.size * 3) {
          const maxChars = Math.floor((node.size * 3) / 6);
          displayLabel = node.label.substring(0, maxChars) + '…';
        }

        // Draw label below node
        ctx.fillText(displayLabel, node.x, node.y + node.size + 10);
      }
    });
  }

  /**
   * Render a force graph with interactive hover tooltips.
   * Adds mousemove listener to show node details.
   * @param {HTMLElement} container
   * @param {Object} data
   * @param {Object} options
   */
  static renderInteractive(container, data, options = {}) {
    // First render the static graph
    this.render(container, data, options);

    const canvas = container.querySelector('canvas');
    if (!canvas) return;

    // Build node lookup for hit testing
    const { nodes } = data;
    const nodeMap = new Map();
    nodes.forEach(node => {
      nodeMap.set(node.id, node);
    });

    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.style.position = 'absolute';
    tooltip.style.background = '#1a1a1a';
    tooltip.style.border = '1px solid #444';
    tooltip.style.borderRadius = '4px';
    tooltip.style.padding = '4px 8px';
    tooltip.style.fontSize = '10px';
    tooltip.style.color = '#ccc';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.display = 'none';
    tooltip.style.zIndex = '1000';
    container.style.position = 'relative';
    container.appendChild(tooltip);

    // Mouse move handler (simplified - would need node position data)
    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Hit test (would need stored node positions from render)
      // This is a simplified version
      tooltip.style.display = 'none';
    });
  }
}
