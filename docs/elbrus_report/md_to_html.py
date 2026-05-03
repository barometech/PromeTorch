"""Конвертер ELBRUS_REPORT_v2.md → HTML с CSS для красивого PDF print."""
import io, re, sys, html

if len(sys.argv) < 3:
    print("Usage: md_to_html.py <input.md> <output.html>")
    sys.exit(1)

src = io.open(sys.argv[1], encoding='utf-8').read()

# Очень простая Markdown -> HTML конверсия (наш отчёт — базовый MD без экзотики)
def conv(md):
    lines = md.split('\n')
    out = []
    in_code = False
    in_table = False
    in_list = False

    for ln in lines:
        # code block
        if ln.startswith('```'):
            if in_code:
                out.append('</pre>')
                in_code = False
            else:
                out.append('<pre>')
                in_code = True
            continue
        if in_code:
            out.append(html.escape(ln))
            continue

        # Headers
        m = re.match(r'^(#{1,6})\s+(.+)$', ln)
        if m:
            lvl = len(m.group(1))
            text = m.group(2)
            out.append(f'<h{lvl}>{html.escape(text)}</h{lvl}>')
            continue

        # Tables
        if '|' in ln and ln.strip().startswith('|'):
            cells = [c.strip() for c in ln.strip().strip('|').split('|')]
            if all(set(c) <= set('-:| ') for c in cells):
                continue  # separator
            if not in_table:
                out.append('<table>')
                in_table = True
                tag = 'th'
            else:
                tag = 'td'
            cell_html = ''.join(f'<{tag}>{inline(c)}</{tag}>' for c in cells)
            out.append(f'<tr>{cell_html}</tr>')
            continue
        else:
            if in_table:
                out.append('</table>')
                in_table = False

        # Lists
        if re.match(r'^\s*[-*]\s+', ln) or re.match(r'^\s*\d+\.\s+', ln):
            if not in_list:
                out.append('<ul>')
                in_list = True
            txt = re.sub(r'^\s*[-*0-9.]+\s+', '', ln)
            out.append(f'<li>{inline(txt)}</li>')
            continue
        else:
            if in_list:
                out.append('</ul>')
                in_list = False

        # Block quote
        if ln.startswith('>'):
            out.append(f'<blockquote>{inline(ln.lstrip("> "))}</blockquote>')
            continue

        # Horizontal rule
        if ln.strip() in ('---', '***', '___'):
            out.append('<hr>')
            continue

        # Paragraph
        if ln.strip():
            out.append(f'<p>{inline(ln)}</p>')
        else:
            out.append('')

    if in_table: out.append('</table>')
    if in_list: out.append('</ul>')
    if in_code: out.append('</pre>')
    return '\n'.join(out)

def inline(text):
    text = html.escape(text)
    # bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # italic
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    # inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text

body = conv(src)

css = '''
body { font-family: 'Segoe UI', Verdana, sans-serif; line-height: 1.6;
       max-width: 920px; margin: 32px auto; color: #222; padding: 0 32px; }
h1 { color: #1a3a8a; border-bottom: 3px solid #1a3a8a; padding-bottom: 6px; }
h2 { color: #224488; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
h3 { color: #2a4080; margin-top: 22px; }
h4 { color: #335588; margin-top: 18px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th { background: #e8eef8; border: 1px solid #aac; padding: 8px 12px; text-align: left; }
td { border: 1px solid #ccd; padding: 6px 12px; }
tr:nth-child(even) td { background: #f6f8fc; }
code { background: #f0f2f8; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; font-size: 0.92em; }
pre { background: #f6f8fc; border: 1px solid #d0d4e0; border-radius: 4px; padding: 12px;
      overflow-x: auto; font-family: 'Consolas', monospace; font-size: 0.88em; line-height: 1.4; }
blockquote { background: #fff8e1; border-left: 4px solid #ffa726; margin: 12px 0;
             padding: 8px 18px; color: #555; }
hr { border: none; border-top: 2px solid #ccd; margin: 24px 0; }
strong { color: #1a3a8a; }
@page { margin: 18mm 14mm; size: A4; }
'''

html_doc = f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>PromeTorch на Эльбрус 8C2 — финальный отчёт</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>'''

io.open(sys.argv[2], 'w', encoding='utf-8').write(html_doc)
print(f"Wrote {sys.argv[2]}")
