from pathlib import Path

script = '\n<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>\n<script>window.MathJax={tex:{inlineMath:[["$","$"],["\\(","\\)"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:true},svg:{fontCache:"global"}};</script>\n'

roots = [Path('.'), Path('.deploy_git')]
patched = []

for root in roots:
    if not root.exists():
        continue
    for path in root.rglob('*.html'):
        try:
            text = path.read_text('utf-8', errors='ignore')
        except Exception:
            continue
        if 'mathjax' in text.lower():
            continue
        if '$$' not in text and '$' not in text:
            continue
        if '</body>' not in text:
            continue
        text = text.replace('</body>', script + '</body>')
        path.write_text(text, encoding='utf-8')
        patched.append(str(path))

print('patched', len(patched))
for item in patched[:20]:
    print(item)
