"""Builds throwaway in-memory indexes with chunking features switched on or off."""

from chromadb.api.models.Collection import Collection

from chunking.context_header import build_embedding_text
from chunking.hybrid_chunker import chunk_document
from chunking.loader import load_markdown_files
from chunking.metadata import build_metadata
from config.settings import DOCS_DIR, EVAL_COLLECTION_PREFIX
from vectorstore.chroma_client import create_memory_collection
from vectorstore.doc_hasher import hash_text
from vectorstore.embedder import embed_documents
from vectorstore.indexer import replace_document_chunks


def build_memory_index(split_on_h3: bool, include_heading_path: bool, merge_small: bool) -> Collection:
    """Chunk, embed, and store every doc in a new in-memory collection.

    Uses the same steps as real ingestion, so the only differences between
    setups are the two switches.

    Args:
        split_on_h3: Whether long sections are re-split on `###` headings.
        include_heading_path: Whether the heading path is embedded with each chunk.
        merge_small: Whether chunks under MIN_CHUNK_TOKENS join a same-section neighbour.

    Returns:
        The populated collection. The real on-disk collection is not touched.
    """
    collection = create_memory_collection(
        f"{EVAL_COLLECTION_PREFIX}h3_{int(split_on_h3)}_path_{int(include_heading_path)}_merge_{int(merge_small)}"
    )
    for document in load_markdown_files(DOCS_DIR):
        chunks = chunk_document(document, split_on_h3=split_on_h3, merge_small=merge_small)
        file_hash = hash_text(document.text)
        replace_document_chunks(
            collection,
            document.source_path,
            documents=[chunk.text for chunk in chunks],
            embeddings=embed_documents([build_embedding_text(c, include_heading_path) for c in chunks]),
            metadatas=[build_metadata(chunk, file_hash) for chunk in chunks],
        )
    return collection
