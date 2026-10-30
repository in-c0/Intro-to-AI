"""MkDocs hook: reading-time + chapter position for the learning path.

- Chapter pages (front matter `chapter: N`) get a one-line meta row under the
  title: "Chapter N of 5 · ~M min read".
- `{{ minutes:<file.md> }}` anywhere in a page is replaced with that page's
  reading time, so the Start-here table never goes stale.
"""
import math
import re

WPM = 220
_minutes = {}
_total_chapters = 0
_FRONT = re.compile(r"\A---\n.*?\n---\n", re.S)
_TOKEN = re.compile(r"\{\{\s*minutes:([\w./-]+)\s*\}\}")


def _read_minutes(text):
    text = _FRONT.sub("", text)
    text = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]", " ", text, flags=re.S)  # display maths
    words = len(re.findall(r"[A-Za-z0-9]+", text))
    return max(1, math.ceil(words / WPM))


def on_files(files, config, **kwargs):
    global _total_chapters
    _minutes.clear()
    _total_chapters = 0
    for f in files.documentation_pages():
        src = f.content_string
        _minutes[f.src_uri] = _read_minutes(src)
        if re.search(r"^chapter:\s*\d+", _FRONT.match(src).group(0) if _FRONT.match(src) else "", re.M):
            _total_chapters += 1
    return files


def on_page_markdown(markdown, page, **kwargs):
    markdown = _TOKEN.sub(lambda m: f"{_minutes.get(m.group(1), '?')} min", markdown)
    n = page.meta.get("chapter")
    if n:
        meta = (
            f'<p class="chapter-meta">Chapter {n} of {_total_chapters}'
            f" · ~{_minutes.get(page.file.src_uri, '?')} min read</p>"
        )
        markdown = re.sub(r"^(# .+)$", r"\1\n\n" + meta, markdown, count=1, flags=re.M)
    return markdown
