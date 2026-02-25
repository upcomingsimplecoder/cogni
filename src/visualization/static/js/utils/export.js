/**
 * utils/export.js
 * Screenshot and JSON export utilities.
 */

/**
 * Capture canvas as image and copy to clipboard (if supported).
 * Falls back to download if clipboard API not available.
 * @param {HTMLCanvasElement} canvas
 */
export async function screenshot(canvas) {
  try {
    // Try clipboard API first
    if (navigator.clipboard && canvas.toBlob) {
      canvas.toBlob(async (blob) => {
        try {
          await navigator.clipboard.write([
            new ClipboardItem({ 'image/png': blob })
          ]);
          console.log('Screenshot copied to clipboard');
          showNotification('Screenshot copied to clipboard!');
        } catch (err) {
          console.warn('Clipboard write failed, downloading instead:', err);
          downloadBlob(blob, 'autocog-screenshot.png');
        }
      });
    } else {
      // Fallback: download
      const dataUrl = canvas.toDataURL('image/png');
      downloadDataUrl(dataUrl, 'autocog-screenshot.png');
    }
  } catch (error) {
    console.error('Screenshot failed:', error);
    showNotification('Screenshot failed', 'error');
  }
}

/**
 * Export current state as JSON.
 * @param {Object} renderCtx - Render context
 */
export function exportJSON(renderCtx) {
  const tickData = renderCtx.temporalBuffer?.getCurrentTick();
  if (!tickData) {
    console.warn('No data to export');
    return;
  }

  const exportData = {
    tick: tickData.tick,
    agents: tickData.agents,
    emergent_events: tickData.emergent_events,
    timestamp: new Date().toISOString(),
  };

  const jsonStr = JSON.stringify(exportData, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  downloadBlob(blob, `autocog-state-tick${tickData.tick}.json`);

  showNotification('State exported as JSON');
}

/**
 * Download a blob as a file.
 * @param {Blob} blob
 * @param {string} filename
 */
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Download a data URL as a file.
 * @param {string} dataUrl
 * @param {string} filename
 */
function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/**
 * Show a temporary notification.
 * @param {string} message
 * @param {string} type - 'info' or 'error'
 */
function showNotification(message, type = 'info') {
  const div = document.createElement('div');
  div.className = `notification notification-${type}`;
  div.textContent = message;
  div.style.cssText = `
    position: fixed;
    top: 80px;
    right: 20px;
    padding: 12px 20px;
    background: ${type === 'error' ? '#ef4444' : '#06b6d4'};
    color: white;
    border-radius: 4px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 10000;
    animation: slideIn 0.3s ease-out;
  `;

  document.body.appendChild(div);

  setTimeout(() => {
    div.style.animation = 'slideOut 0.3s ease-out';
    setTimeout(() => div.remove(), 300);
  }, 3000);
}
