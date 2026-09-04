import os
import subprocess
import markdown

def build_paper_pdf():
    md_path = os.path.join('paper', 'paper.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Convert markdown with tables and fenced code
    body_html = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

    # Style definitions
    css = """
    @page {
        size: letter;
        margin: 0.75in;
    }
    body {
        font-family: 'Times New Roman', Times, serif;
        font-size: 10.5pt;
        line-height: 1.42;
        color: #111;
        margin: 0;
        padding: 0;
        text-align: justify;
    }
    h1 {
        font-size: 18pt;
        font-weight: bold;
        text-align: center;
        margin-bottom: 6px;
        line-height: 1.25;
    }
    h2 {
        font-size: 12.5pt;
        border-bottom: 1.5px solid #222;
        padding-bottom: 3px;
        margin-top: 22px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    h3 {
        font-size: 11pt;
        font-weight: bold;
        margin-top: 14px;
        margin-bottom: 6px;
    }
    p {
        margin-top: 0;
        margin-bottom: 8px;
        text-indent: 1.5em;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }
    th, td {
        padding: 5px 8px;
        text-align: left;
    }
    th {
        border-top: 1.5px solid #222;
        border-bottom: 1.2px solid #222;
        font-weight: bold;
        background: #f8f9fa;
    }
    td {
        border-bottom: 1px solid #eee;
    }
    tr:last-child td {
        border-bottom: 1.5px solid #222;
    }
    img {
        max-width: 85%;
        display: block;
        margin: 16px auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        page-break-inside: avoid;
    }
    pre {
        background: #f6f8fa;
        padding: 10px;
        border-radius: 4px;
        font-size: 8.5pt;
        line-height: 1.3;
        overflow-x: auto;
    }
    code {
        font-family: 'Courier New', monospace;
        font-size: 9pt;
    }
    blockquote {
        margin: 14px 0;
        padding: 10px 18px;
        background: #fcfcfc;
        border-left: 3.5px solid #0366d6;
        font-size: 9.5pt;
        line-height: 1.4;
    }
    ol, ul {
        margin-top: 0;
        margin-bottom: 8px;
        padding-left: 24px;
    }
    li {
        margin-bottom: 4px;
    }
    strong {
        font-weight: bold;
    }
    em {
        font-style: italic;
    }
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Kernelized Bound Synaptic Memory</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
<style>
{css}
</style>
</head>
<body>
{body_html}
<script>
document.addEventListener("DOMContentLoaded", function() {{
  if (typeof renderMathInElement !== 'undefined') {{
    renderMathInElement(document.body, {{
      delimiters: [
        {{left: "$$", right: "$$", display: true}},
        {{left: "$", right: "$", display: false}}
      ],
      throwOnError: false
    }});
  }}
}});
</script>
</body>
</html>"""

    # Embed image as Base64 so Edge renders it instantly without file URI restrictions
    import base64
    img_path = os.path.abspath(os.path.join('paper', 'loss_vs_compute_10m.png'))
    if os.path.exists(img_path):
        with open(img_path, 'rb') as img_f:
            b64_data = base64.b64encode(img_f.read()).decode('utf-8')
        html = html.replace('loss_vs_compute_10m.png', f'data:image/png;base64,{b64_data}')
        html = html.replace('results/loss_vs_compute_10m.png', f'data:image/png;base64,{b64_data}')
        print(f"Embedded base64 image ({len(b64_data)} bytes)")

    html_path = os.path.abspath(os.path.join('paper', 'paper.html'))
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated HTML at: {html_path}")

    # Compile to PDF using Microsoft Edge
    pdf_path = os.path.abspath(os.path.join('paper', 'Kernelized_Bound_Synaptic_Memory.pdf'))
    edge_paths = [
        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
    ]
    edge_exe = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_exe = p
            break

    if not edge_exe:
        print("Microsoft Edge not found.")
        return False

    cmd = [
        edge_exe,
        '--headless',
        '--disable-gpu',
        '--run-all-compositor-stages-before-draw',
        '--no-pdf-header-footer',
        f'--print-to-pdf={pdf_path}',
        html_path
    ]

    print(f"Compiling PDF via {edge_exe}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        print(f"SUCCESS: PDF created at {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
        return True
    else:
        print(f"Error compiling PDF: {res.stderr}")
        return False

if __name__ == '__main__':
    build_paper_pdf()
