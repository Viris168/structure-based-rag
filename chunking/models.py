"""Plain data containers passed between the chunking, embedding, and storage steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    """One Markdown file read from the docs folder.

    Attributes:
        doc_title: The file's `# H1` title, or its filename stem if it has none.
        source_path: Filename relative to the docs folder, e.g. "DOCUMENTATION.md".
        text: The full raw Markdown content.
    """

    doc_title: str
    source_path: str
    text: str


@dataclass(frozen=True)
class Section:
    """A slice of Markdown that starts at one heading and ends before the next.

    Attributes:
        heading: The heading text without the `#` marks; "" for text before the first heading.
        text: The raw Markdown of the slice, including its heading line.
    """

    heading: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of a document, ready to be embedded and stored.

    Attributes:
        text: Raw Markdown exactly as it appears in the source (no heading prefix).
        doc_title: Title of the document the chunk came from.
        section_title: The `##` heading it sits under (doc title for the preamble).
        subsection: The `###` heading it sits under, or "" if it was not H3-split.
        chunk_index: Position of the chunk within its document, starting at 0.
        token_count: Length in embedding-model tokens.
        source_path: Filename relative to the docs folder.
    """

    text: str
    doc_title: str
    section_title: str
    subsection: str
    chunk_index: int
    token_count: int
    source_path: str
