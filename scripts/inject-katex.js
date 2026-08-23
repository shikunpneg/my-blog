/**
 * Hexo plugin: Inject KaTeX CSS into <head> of pages that contain rendered math.
 *
 * Engineering rationale:
 *   - LaTeX is rendered server-side by `markdown-it-texmath` + KaTeX (configured
 *     in _config.yml). The output HTML contains `.katex` elements that require
 *     the KaTeX stylesheet to display correctly.
 *   - KaTeX CSS is vendored locally in `source/css/katex.min.css` (with fonts
 *     in `source/css/fonts/`) to avoid CDN flakiness and offline reliability.
 *   - Injection is conditional: only pages with `.katex` get the link, so
 *     non-math pages stay lean.
 *   - Idempotent: safe to run multiple times; won't double-inject.
 *   - Operates in-memory via `after_render:html` (no filesystem scanning).
 *
 * Why this is the engineering-grade solution:
 *   - No placeholder schemes (markdown-it-texmath handles LaTeX natively at
 *     parse time, before any HTML escaping).
 *   - No client-side JS rendering (KaTeX renders at build time).
 *   - No CDN dependency (CSS is local).
 *   - No "walk public/" file scan (in-memory filter).
 */

'use strict';

const KATEX_CSS_LINK =
  '<link rel="stylesheet" href="/css/katex.min.css" crossorigin="anonymous">';

hexo.extend.filter.register('after_render:html', function (html /*, data */) {
  // Skip malformed HTML
  if (!html || !html.includes('</head>')) return html;
  // Idempotent: skip if already injected
  if (html.includes('katex.min.css')) return html;
  // Skip pages without rendered math
  if (!html.includes('class="katex"')) return html;
  // Inject just before </head>
  return html.replace('</head>', KATEX_CSS_LINK + '</head>');
});
