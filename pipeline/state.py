"""The state object passed between the steps of the question-answering graph."""

from typing import TypedDict

from generation.prompt_builder import Message
from retrieval.models import RetrievedChunk


class GraphState(TypedDict, total=False):
    """Everything the graph knows about one question.

    Each node reads what earlier nodes wrote and adds its own keys.

    Attributes:
        question: The user's question (set by the caller).
        chunks: Top-k chunks from hybrid search (set by retrieve).
        messages: The grounded prompt (set by build_prompt).
        answer: The model's reply, or the refusal (set by generate).
        citations: Citation labels the answer used (set by generate).
    """

    question: str
    chunks: list[RetrievedChunk]
    messages: list[Message]
    answer: str
    citations: list[str]
