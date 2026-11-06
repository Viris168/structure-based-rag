"""Shows how the docs would be chunked, without embedding or storing anything."""

from chunking.context_header import build_heading_path
from chunking.hybrid_chunker import chunk_document
from chunking.loader import load_markdown_files
from chunking.text_size import count_words
from config.settings import DOCS_DIR


def preview_chunks() -> str:
    """Chunk every doc in DOCS_DIR and describe the result.

    Returns:
        Per document: chunk count, then one line per chunk with its heading
        path, word count, and token count.
    """
    lines: list[str] = []
    for document in load_markdown_files(DOCS_DIR):
        chunks = chunk_document(document)
        lines.append(f"{document.source_path}: {len(chunks)} chunks")
        for chunk in chunks:
            lines.append(
                f"  [{chunk.chunk_index}] {build_heading_path(chunk.doc_title, chunk.section_title, chunk.subsection)}"
                f"  ({count_words(chunk.text)} words, {chunk.token_count} tokens)"
            )
    return "\n".join(lines) or f"No Markdown files found in {DOCS_DIR}"
