// Local-only Markdown/MathJax preview. No document content is uploaded.
// Usage: node experiments/preview_card.mjs PATH_TO_MARKED_PACKAGE
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const { marked } = await import(pathToFileURL(path.resolve(process.argv[2], 'lib/marked.esm.js')).href);
const source = fs.readFileSync('interpretation_card.md', 'utf8');
const math = [];
const escaped = source.replace(/\$\$[^]*?\$\$|\$[^$\n]+\$/g, expression => {
  const index = math.push(expression) - 1;
  return `MATHTOKEN${index}END`;
});
let html = marked.parse(escaped);
html = html.replace(/MATHTOKEN(\d+)END/g, (_, index) => math[+index].replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'));
fs.mkdirSync('.preview', { recursive: true });
fs.writeFileSync('.preview/interpretation_card.html', `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><title>Interpretation card — local preview</title>
<style>body{max-width:980px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;color:#24292f;background:white}h1,h2{border-bottom:1px solid #d0d7de;padding-bottom:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d0d7de;padding:8px 12px;text-align:left}th{background:#f6f8fa}pre{background:#f6f8fa;padding:16px;overflow:auto}code{font-size:90%}a{color:#0969da}.notice{background:#fff8c5;padding:12px}mjx-container[display=true]{overflow-x:auto;overflow-y:hidden;padding:6px 0}</style>
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},startup:{ready(){MathJax.startup.defaultReady();MathJax.startup.promise.then(()=>{document.getElementById('render-status').textContent='Math rendering complete';});}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script></head><body><div class="notice">Local Markdown + MathJax preview, not GitHub or Typora. <span id="render-status">Loading math renderer…</span></div>${html}</body></html>`);
console.log(`Prepared ${math.length} math expressions in .preview/interpretation_card.html`);
