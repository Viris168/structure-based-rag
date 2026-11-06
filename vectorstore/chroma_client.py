"""Opens the persistent ChromaDB collection that holds every chunk."""

import chromadb
from chromadb.api.models.Collection import Collection

from config.settings import CHROMA_PATH, COLLECTION_NAME, DISTANCE_METRIC


def get_collection() -> Collection:
    """Open the on-disk collection, creating it on first use.

    We pass our own embeddings, so no embedding function is attached here;
    that keeps Chroma from silently loading a different default model.

    Returns:
        The COLLECTION_NAME collection stored under CHROMA_PATH.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )


def create_memory_collection(name: str) -> Collection:
    """Create an empty in-memory collection that is never written to disk.

    The evaluation builds throwaway indexes with features switched off, so
    they must not touch the real collection. Any earlier collection with the
    same name is replaced.

    Args:
        name: Collection name, unique per evaluation setup.

    Returns:
        An empty collection using the same distance metric as the real one.
    """
    client = chromadb.EphemeralClient()
    if name in [existing.name for existing in client.list_collections()]:
        client.delete_collection(name)
    return client.create_collection(
        name=name,
        embedding_function=None,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )
