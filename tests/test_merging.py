"""Small-chunk merging: only within a `##` section, only when the result fits the model."""

from pathlib import Path

import config.settings as settings
from chunking.chunk_merger import merge_small_chunks
from chunking.hybrid_chunker import chunk_document
from chunking.loader import load_markdown_files
from chunking.models import Chunk
from chunking.text_size import count_tokens

SMALL = "Run it."
BIG = " ".join(["word"] * 100)


def _chunk(text: str, section: str, subsection: str = "", index: int = 0) -> Chunk:
    return Chunk(
        text=text,
        doc_title="Doc",
        section_title=section,
        subsection=subsection,
        chunk_index=index,
        token_count=count_tokens(text),
        source_path="doc.md",
    )


def test_small_chunk_joins_previous_chunk_in_same_section() -> None:
    merged = merge_small_chunks([_chunk(BIG, "Usage", "Colors"), _chunk(SMALL, "Usage", "Restoring PATH", 1)])
    assert len(merged) == 1
    assert merged[0].text == f"{BIG}\n\n{SMALL}"
    assert merged[0].subsection == f"Colors{settings.SUBSECTION_JOINER}Restoring PATH"
    assert merged[0].token_count == count_tokens(merged[0].text)


def test_small_intro_joins_next_chunk_when_nothing_before_it() -> None:
    merged = merge_small_chunks([_chunk(SMALL, "Usage"), _chunk(BIG, "Usage", "Colors", 1)])
    assert len(merged) == 1
    assert merged[0].text == f"{SMALL}\n\n{BIG}"
    assert merged[0].subsection == "Colors"


def test_small_chunk_never_joins_another_section() -> None:
    chunks = [_chunk(BIG, "Usage"), _chunk(SMALL, "License", index=1), _chunk(BIG, "Problems", index=2)]
    merged = merge_small_chunks(chunks)
    assert [c.section_title for c in merged] == ["Usage", "License", "Problems"]


def test_small_chunk_is_not_merged_past_the_token_limit() -> None:
    full = " ".join(["word"] * (settings.MAX_EMBED_TOKENS - 10))
    merged = merge_small_chunks([_chunk(full, "Usage", "A"), _chunk(SMALL, "Usage", "B", 1)])
    assert len(merged) == 2


def test_merged_chunks_are_renumbered() -> None:
    chunks = [_chunk(BIG, "Usage"), _chunk(SMALL, "Usage", "B", 1), _chunk(BIG, "Problems", index=2)]
    assert [c.chunk_index for c in merge_small_chunks(chunks)] == [0, 1]


def test_merging_can_be_switched_off(docs_dir: Path) -> None:
    for document in load_markdown_files(docs_dir):
        merged = chunk_document(document)
        unmerged = chunk_document(document, merge_small=False)
        assert len(merged) <= len(unmerged)
        assert "".join(c.text for c in merged).replace("\n", "") == "".join(c.text for c in unmerged).replace("\n", "")
