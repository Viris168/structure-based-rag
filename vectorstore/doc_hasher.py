"""Fingerprints text so ingestion can tell what changed since the last run."""

import hashlib


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of the text.

    Used for whole files (skip unchanged docs) and single chunks (stable IDs).

    Args:
        text: Any text.

    Returns:
        A 64-character hex string; identical text always gives the same value.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
