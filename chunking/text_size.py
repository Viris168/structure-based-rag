"""Measures text length in words (for the H3 rule) and in model tokens (for the embed limit)."""

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config.settings import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _tokenizer() -> PreTrainedTokenizerBase:
    """Load the embedding model's tokenizer once and reuse it.

    Returns:
        The tokenizer that matches EMBEDDING_MODEL.
    """
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def count_words(text: str) -> int:
    """Count whitespace-separated words.

    Args:
        text: Any text.

    Returns:
        Number of words.
    """
    return len(text.split())


def count_tokens(text: str) -> int:
    """Count tokens exactly as the embedding model will see them.

    Includes the model's special start/end tokens, so the result can be
    compared directly against MAX_EMBED_TOKENS.

    Args:
        text: Any text.

    Returns:
        Number of tokens.
    """
    return len(_tokenizer().encode(text, add_special_tokens=True, verbose=False))
