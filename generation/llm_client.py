"""Sends a prompt to the configured LLM and returns its reply."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from config.settings import LLM_API_KEY_ENV, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS
from generation.prompt_builder import Message


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Create the API client once, reading the key from .env or the environment.

    Returns:
        An OpenAI-compatible client pointed at LLM_BASE_URL.

    Raises:
        RuntimeError: If the API key is not set.
    """
    load_dotenv()
    api_key = os.getenv(LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{LLM_API_KEY_ENV} is not set. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=api_key, base_url=LLM_BASE_URL)


def generate_answer(messages: list[Message]) -> str:
    """Get the model's reply to a grounded prompt.

    Args:
        messages: System and user messages from build_grounded_prompt.

    Returns:
        The reply text, stripped of surrounding whitespace.
    """
    response = _client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    return (response.choices[0].message.content or "").strip()
