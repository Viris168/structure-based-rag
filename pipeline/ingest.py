"""Phase 1 pipeline: load docs → skip unchanged → chunk → embed → store."""

from dataclasses import dataclass, field

from chunking.context_header import build_embedding_text
from chunking.hybrid_chunker import chunk_document
from chunking.loader import load_markdown_files
from chunking.metadata import build_metadata
from config.settings import DOCS_DIR
from vectorstore.chroma_client import get_collection
from vectorstore.doc_hasher import hash_text
from vectorstore.embedder import embed_documents
from vectorstore.indexer import get_stored_file_hash, remove_documents_not_in, replace_document_chunks


@dataclass
class IngestReport:
    """What one ingestion run did to each document."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks_written: int = 0

    def summary(self) -> str:
        """Format the report for the terminal.

        Returns:
            One line per category plus the number of chunks written.
        """
        return "\n".join(
            [
                f"added:   {self.added or '-'}",
                f"updated: {self.updated or '-'}",
                f"skipped: {self.skipped or '-'}",
                f"removed: {self.removed or '-'}",
                f"chunks written: {self.chunks_written}",
            ]
        )


def run_ingestion() -> IngestReport:
    """Sync the ChromaDB collection with the Markdown files in DOCS_DIR.

    Unchanged files are skipped without re-embedding; changed files have all
    their chunks replaced; files deleted from disk are removed from the store.

    Returns:
        A report of added, updated, skipped, and removed documents.
    """
    collection = get_collection()
    documents = load_markdown_files(DOCS_DIR)
    report = IngestReport()

    for document in documents:
        file_hash = hash_text(document.text)
        stored_hash = get_stored_file_hash(collection, document.source_path)
        if stored_hash == file_hash:
            report.skipped.append(document.source_path)
            continue

        chunks = chunk_document(document)
        embeddings = embed_documents([build_embedding_text(chunk) for chunk in chunks])
        replace_document_chunks(
            collection,
            document.source_path,
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[build_metadata(chunk, file_hash) for chunk in chunks],
        )
        (report.added if stored_hash is None else report.updated).append(document.source_path)
        report.chunks_written += len(chunks)

    report.removed = remove_documents_not_in(collection, {d.source_path for d in documents})
    return report
