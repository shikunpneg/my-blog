const fs = require('fs');
const path = require('path');

hexo.extend.filter.register('after_generate', function () {
  const publicDir = path.join(hexo.base_dir, 'public');
  if (!fs.existsSync(publicDir)) return;

  const script = '\n<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>\n<script>window.MathJax = { tex: { inlineMath: [["$", "$"], ["\\(", "\\)"]], displayMath: [["$$", "$$"], ["\\[", "\\]"]], processEscapes: true }, svg: { fontCache: "global" } };</script>\n';

  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.html')) {
        const htmlPath = fullPath;
        let content = fs.readFileSync(htmlPath, 'utf8');
        if (!content.toLowerCase().includes('mathjax')) {
          if (content.includes('</body>')) {
            content = content.replace('</body>', script + '</body>');
            fs.writeFileSync(htmlPath, content, 'utf8');
          }
        }
      }
    }
  }

  walk(publicDir);
});
