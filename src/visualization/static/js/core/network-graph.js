/**
 * core/network-graph.js
 * Weighted directed graph for relationship tracking.
 */

export class NetworkGraph {
  constructor() {
    this._edges = new Map(); // nodeId -> [{to, type, weight}]
  }

  /**
   * Set or update an edge.
   * @param {string} from - Source node ID
   * @param {string} to - Target node ID
   * @param {string} type - Edge type (e.g., 'trust', 'cultural', 'coalition')
   * @param {number} weight - Edge weight
   */
  setEdge(from, to, type, weight) {
    if (!this._edges.has(from)) {
      this._edges.set(from, []);
    }

    const edges = this._edges.get(from);
    const existing = edges.find(e => e.to === to && e.type === type);

    if (existing) {
      existing.weight = weight;
    } else {
      edges.push({ to, type, weight });
    }
  }

  /**
   * Get all edges from a node.
   * @param {string} nodeId - Node ID
   * @param {string} [typeFilter] - Optional edge type filter
   * @returns {Array<{to: string, type: string, weight: number}>}
   */
  getEdges(nodeId, typeFilter = null) {
    const edges = this._edges.get(nodeId) || [];
    if (typeFilter) {
      return edges.filter(e => e.type === typeFilter);
    }
    return edges;
  }

  /**
   * Get all edges of a specific type.
   * @param {string} type - Edge type
   * @returns {Array<{from: string, to: string, weight: number}>}
   */
  getAllEdges(type) {
    const result = [];
    for (const [from, edges] of this._edges) {
      for (const edge of edges) {
        if (edge.type === type) {
          result.push({ from, to: edge.to, weight: edge.weight });
        }
      }
    }
    return result;
  }

  /**
   * Get connected components (undirected).
   * @param {string} [edgeType] - Optional edge type filter
   * @returns {Array<Set<string>>} Array of component sets
   */
  getConnectedComponents(edgeType = null) {
    const visited = new Set();
    const components = [];

    // Build adjacency for undirected traversal
    const adjacency = new Map();
    for (const [from, edges] of this._edges) {
      for (const edge of edges) {
        if (edgeType && edge.type !== edgeType) continue;

        if (!adjacency.has(from)) adjacency.set(from, []);
        if (!adjacency.has(edge.to)) adjacency.set(edge.to, []);

        adjacency.get(from).push(edge.to);
        adjacency.get(edge.to).push(from);
      }
    }

    // DFS for each component
    for (const nodeId of adjacency.keys()) {
      if (visited.has(nodeId)) continue;

      const component = new Set();
      const stack = [nodeId];

      while (stack.length > 0) {
        const current = stack.pop();
        if (visited.has(current)) continue;

        visited.add(current);
        component.add(current);

        const neighbors = adjacency.get(current) || [];
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor)) {
            stack.push(neighbor);
          }
        }
      }

      components.push(component);
    }

    return components;
  }

  /**
   * Clear all edges.
   */
  clear() {
    this._edges.clear();
  }

  /**
   * Get all node IDs.
   * @returns {Set<string>}
   */
  getAllNodes() {
    const nodes = new Set();
    for (const [from, edges] of this._edges) {
      nodes.add(from);
      for (const edge of edges) {
        nodes.add(edge.to);
      }
    }
    return nodes;
  }
}
