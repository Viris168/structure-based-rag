"""Renders one source document as an HTML page that sources can link into."""

from html import escape

from markdown_it import MarkdownIt

from chunking.markdown_syntax import clean_heading
from config.settings import DOCS_DIR, DOCS_GLOB, SECTION_HEADING_LEVEL, SUBSECTION_HEADING_LEVEL
from ui.doc_links import heading_anchor

_TITLE_LEVEL = 1

# Raw HTML inside the docs is shown as text, never run as page code.
_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg: #ffffff; --text: #1f2328; --muted: #59636e; --code-bg: #f6f8fa; --border: #d1d9e0; --mark: #fff3bf; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #0d1117; --text: #e6edf3; --muted: #9198a1; --code-bg: #161b22; --border: #3d444d; --mark: #3b3312; }}
  }}
  body {{ background: var(--bg); color: var(--text); font: 16px/1.6 system-ui, sans-serif; margin: 0; }}
  main {{ max-width: 46rem; margin: 0 auto; padding: 2rem 1rem 6rem; }}
  .file {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 1.5rem; }}
  h1, h2, h3 {{ line-height: 1.3; scroll-margin-top: 1rem; }}
  h2 {{ border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; margin-top: 2rem; }}
  :target {{ background: var(--mark); border-radius: 4px; padding-left: 0.4rem; }}
  code {{ background: var(--code-bg); padding: 0.1rem 0.3rem; border-radius: 4px; font-size: 0.9em; }}
  pre {{ background: var(--code-bg); padding: 1rem; border-radius: 6px; overflow-x: auto; }}
  pre code {{ padding: 0; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid var(--border); padding: 0.3rem 0.6rem; }}
</style>
</head>
<body>
<main>
<div class="file">{source_path}</div>
{body}
</main>
</body>
</html>
"""


def _render_with_anchors(text: str) -> str:
    """Convert Markdown to HTML, giving each `#`, `##`, and `###` heading the ID that sources link to.

    Args:
        text: Raw Markdown of the whole document.

    Returns:
        The document body as HTML.
    """
    tokens = _markdown.parse(text)
    current_section = ""
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        heading = clean_heading(tokens[index + 1].content)
        level = int(token.tag[1:])
        if level == _TITLE_LEVEL:
            token.attrSet("id", heading_anchor(heading))
        elif level == SECTION_HEADING_LEVEL:
            current_section = heading
            token.attrSet("id", heading_anchor(heading))
        elif level == SUBSECTION_HEADING_LEVEL:
            token.attrSet("id", heading_anchor(current_section, heading))
    return _markdown.renderer.render(tokens, _markdown.options, {})


def render_doc_page(source_path: str) -> str | None:
    """Build the full-document page for one file in DOCS_DIR.

    Only files that are actually in DOCS_DIR can be shown, so a crafted link
    like "/doc/../.env" cannot read anything else on the computer.

    Args:
        source_path: Filename relative to the docs folder, as stored in metadata.

    Returns:
        The HTML page, or None if no such document exists.
    """
    documents = {path.name: path for path in DOCS_DIR.glob(DOCS_GLOB)}
    if source_path not in documents:
        return None
    text = documents[source_path].read_text(encoding="utf-8")
    return _PAGE_TEMPLATE.format(
        title=escape(source_path), source_path=escape(source_path), body=_render_with_anchors(text)
    )
