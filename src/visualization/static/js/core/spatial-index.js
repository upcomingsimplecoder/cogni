/**
 * core/spatial-index.js
 * Grid-cell based spatial index for fast neighbor queries.
 */

export class SpatialIndex {
  /**
   * @param {number} worldWidth - World width in cells
   * @param {number} worldHeight - World height in cells
   * @param {number} cellSize - Spatial cell size (default 8)
   */
  constructor(worldWidth, worldHeight, cellSize = 8) {
    this.worldWidth = worldWidth;
    this.worldHeight = worldHeight;
    this.cellSize = cellSize;
    this.gridWidth = Math.ceil(worldWidth / cellSize);
    this.gridHeight = Math.ceil(worldHeight / cellSize);
    this._grid = new Map();
  }

  /**
   * Update index with current agent positions.
   * @param {Array} agents - Array of agents with {id, x, y}
   */
  update(agents) {
    this._grid.clear();

    for (const agent of agents) {
      if (!agent.alive) continue;

      const cellX = Math.floor(agent.x / this.cellSize);
      const cellY = Math.floor(agent.y / this.cellSize);
      const key = this._cellKey(cellX, cellY);

      if (!this._grid.has(key)) {
        this._grid.set(key, []);
      }
      this._grid.get(key).push(agent);
    }
  }

  /**
   * Get agents within radius of a position.
   * @param {number} x - World X coordinate
   * @param {number} y - World Y coordinate
   * @param {number} radius - Search radius
   * @returns {Array} Agents within radius
   */
  getNeighbors(x, y, radius) {
    const neighbors = [];
    const cellX = Math.floor(x / this.cellSize);
    const cellY = Math.floor(y / this.cellSize);
    const cellRadius = Math.ceil(radius / this.cellSize);

    // Check all cells within radius
    for (let dx = -cellRadius; dx <= cellRadius; dx++) {
      for (let dy = -cellRadius; dy <= cellRadius; dy++) {
        const cx = cellX + dx;
        const cy = cellY + dy;
        if (cx < 0 || cx >= this.gridWidth || cy < 0 || cy >= this.gridHeight) {
          continue;
        }

        const agents = this.getAgentsInCell(cx, cy);
        for (const agent of agents) {
          const dist = Math.sqrt(
            Math.pow(agent.x - x, 2) + Math.pow(agent.y - y, 2)
          );
          if (dist <= radius) {
            neighbors.push(agent);
          }
        }
      }
    }

    return neighbors;
  }

  /**
   * Get all agents in a specific cell.
   * @param {number} cellX - Cell X coordinate
   * @param {number} cellY - Cell Y coordinate
   * @returns {Array} Agents in cell
   */
  getAgentsInCell(cellX, cellY) {
    const key = this._cellKey(cellX, cellY);
    return this._grid.get(key) || [];
  }

  /**
   * Get cell key for hashing.
   * @param {number} cellX
   * @param {number} cellY
   * @returns {string}
   */
  _cellKey(cellX, cellY) {
    return `${cellX},${cellY}`;
  }

  /**
   * Clear the index.
   */
  clear() {
    this._grid.clear();
  }
}
