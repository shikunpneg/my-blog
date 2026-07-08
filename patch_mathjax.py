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
        # 注入条件：含未渲染的 LaTeX 源码（\\mathbf、\\approx 等转义命令）
        if not any(marker in text for marker in ['\\mathbf', '\\approx', '\\mathcal', '\\sum', '\\max']):
            continue
        if '</body>' not in text:
            continue
        text = text.replace('</body>', script + '</body>')
        path.write_text(text, encoding='utf-8')
        patched.append(str(path))

print('patched', len(patched))
for item in patched[:20]:
    print(item)
