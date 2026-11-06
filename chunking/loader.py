"""Reads Markdown files from disk into SourceDocument objects."""

from pathlib import Path

from chunking.markdown_syntax import clean_heading, heading_pattern, is_fence_line
from chunking.models import SourceDocument
from config.settings import DOCS_GLOB

_TITLE_LEVEL = 1


def _find_title(text: str, fallback: str) -> str:
    """Return the first `# H1` heading outside code blocks, or the fallback.

    Args:
        text: Raw Markdown content.
        fallback: Value to use when the document has no H1.

    Returns:
        The document title.
    """
    pattern = heading_pattern(_TITLE_LEVEL)
    in_fence = False
    for line in text.splitlines():
        if is_fence_line(line):
            in_fence = not in_fence
        elif not in_fence and (match := pattern.match(line)):
            return clean_heading(match.group(1)) or fallback
    return fallback


def load_markdown_files(docs_dir: Path) -> list[SourceDocument]:
    """Load every Markdown file in a folder, sorted by filename.

    Args:
        docs_dir: Folder containing the source `.md` files.

    Returns:
        One SourceDocument per file.
    """
    documents = []
    for path in sorted(docs_dir.glob(DOCS_GLOB)):
        text = path.read_text(encoding="utf-8")
        documents.append(
            SourceDocument(
                doc_title=_find_title(text, fallback=path.stem),
                source_path=path.name,
                text=text,
            )
        )
    return documents
