/**
 * core/colors.js
 * Color palettes and utilities.
 */

// Archetype colors (from existing templates)
export const ARCHETYPE_COLORS = {
  gatherer: '#22c55e',
  explorer: '#06b6d4',
  diplomat: '#eab308',
  aggressor: '#ef4444',
  survivalist: '#f8fafc',
};

// Cultural group colors (ring colors)
export const CULTURAL_GROUP_COLORS = [
  '#f472b6', '#a78bfa', '#fb923c', '#2dd4bf',
  '#facc15', '#38bdf8', '#e879f9', '#4ade80',
];

// Cultural bias colors (transmission lines)
export const BIAS_COLORS = {
  prestige: '#facc15',
  conformist: '#3b82f6',
  content: '#22c55e',
  anti_conformist: '#a855f7',
};

// Need bar colors
export const NEED_COLORS = {
  hunger: '#f59e0b',
  thirst: '#3b82f6',
  energy: '#10b981',
  health: '#ef4444',
};

/**
 * Get cultural group color by ID.
 * @param {number} groupId - Cultural group ID
 * @returns {string|null} Hex color or null
 */
export function getCulturalGroupColor(groupId) {
  if (groupId == null || groupId < 0) return null;
  return CULTURAL_GROUP_COLORS[groupId % CULTURAL_GROUP_COLORS.length];
}

/**
 * Interpolate between two colors.
 * @param {string} color1 - Start color (hex)
 * @param {string} color2 - End color (hex)
 * @param {number} t - Interpolation factor (0-1)
 * @returns {string} Interpolated color (hex)
 */
export function interpolate(color1, color2, t) {
  const c1 = hexToRgb(color1);
  const c2 = hexToRgb(color2);

  const r = Math.round(c1.r + (c2.r - c1.r) * t);
  const g = Math.round(c1.g + (c2.g - c1.g) * t);
  const b = Math.round(c1.b + (c2.b - c1.b) * t);

  return rgbToHex(r, g, b);
}

/**
 * Add alpha channel to color.
 * @param {string} color - Hex color
 * @param {number} alpha - Alpha value (0-1)
 * @returns {string} RGBA color string
 */
export function withAlpha(color, alpha) {
  const rgb = hexToRgb(color);
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

/**
 * Convert hex color to RGB object.
 * @param {string} hex - Hex color (#RRGGBB)
 * @returns {{r: number, g: number, b: number}}
 */
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16),
  } : { r: 0, g: 0, b: 0 };
}

/**
 * Convert RGB to hex color.
 * @param {number} r - Red (0-255)
 * @param {number} g - Green (0-255)
 * @param {number} b - Blue (0-255)
 * @returns {string} Hex color
 */
function rgbToHex(r, g, b) {
  return '#' + [r, g, b].map(x => {
    const hex = x.toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  }).join('');
}
