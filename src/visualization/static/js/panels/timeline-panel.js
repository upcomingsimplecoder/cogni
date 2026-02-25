/**
 * panels/timeline-panel.js
 * Timeline scrubber with emergence event markers.
 */

export class TimelinePanel {
  /**
   * @param {HTMLElement} container - Container element
   * @param {Object} dataSource - DataSource instance
   */
  constructor(container, dataSource) {
    this.container = container;
    this.dataSource = dataSource;
    this._setupUI();
  }

  /**
   * Setup UI elements.
   */
  _setupUI() {
    // For live mode: simple recent history indicator
    // For dashboard mode: full scrubber
    if (this.dataSource.mode === 'live') {
      this.container.innerHTML = `
        <input type="range" id="timeline-scrubber" min="0" max="0" value="0"
          style="width: 120px; accent-color: #06b6d4;" disabled>
        <span style="font-size: 11px; color: #888;" id="timeline-label">live</span>
      `;
    } else {
      const maxTicks = this.dataSource.getTotalTicks();
      this.container.innerHTML = `
        <input type="range" id="timeline-scrubber" min="0" max="${maxTicks}" value="0"
          style="flex: 1; accent-color: #06b6d4;">
        <span style="font-size: 14px; font-weight: 500; min-width: 100px; text-align: right;" id="timeline-label">Tick: 0</span>
      `;
    }

    this.scrubber = document.getElementById('timeline-scrubber');
    this.label = document.getElementById('timeline-label');
  }

  /**
   * Update timeline for current tick.
   * @param {number} tick - Current tick
   * @param {Object} renderCtx - Render context
   */
  update(tick, renderCtx) {
    if (this.dataSource.mode === 'live') {
      // Update scrubber max to show recent history
      const historySize = renderCtx.temporalBuffer?.size || 0;
      if (historySize > 0) {
        this.scrubber.max = historySize - 1;
        this.scrubber.value = historySize - 1;
        this.scrubber.disabled = false;
      }
      this.label.textContent = 'live';
    } else {
      // Dashboard mode: update scrubber position
      this.scrubber.value = tick;
      this.label.textContent = `Tick: ${tick}`;
    }

    // Draw markers if lens provides them
    if (renderCtx.activeLens) {
      const markers = renderCtx.activeLens.getTimelineMarkers(tick, renderCtx);
      // TODO: Draw markers on scrubber track
    }
  }

  /**
   * Add event listener for scrubber changes.
   * @param {Function} callback - Called with new tick index
   */
  onScrub(callback) {
    if (this.scrubber) {
      this.scrubber.addEventListener('input', (e) => {
        callback(parseInt(e.target.value));
      });
    }
  }
}
