"""Builds the grounded prompt: the question plus the retrieved chunks, nothing else."""

from chunking.context_header import build_stored_chunk_text
from config.settings import REFUSAL_MESSAGE
from retrieval.models import RetrievedChunk

Message = dict[str, str]

_SYSTEM_PROMPT = f"""You are a documentation assistant. Answer the user's question using ONLY the documentation excerpts provided.

Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge.
- After each claim, cite the excerpt it came from by its number in square brackets, e.g. [2], or [1][3] for several. Use only the numbers of the excerpts below.
- Keep code, commands, and identifiers exactly as written in the excerpts.
- If the excerpts answer only part of the question, or describe a feature, term, or tool the question asks about without answering exactly what was asked, give what they do say about it (with citations), then add one sentence naming what the documentation does not cover. For example, if asked "Is X used for Y?" and the excerpts describe X but never mention Y, say what they say about X and that Y is not covered. Never fill that gap with outside knowledge.
- If the excerpts fully answer the question, do not add a sentence about what is not covered.
- If the excerpts say nothing relevant to the question at all, reply with exactly this sentence and nothing else: {REFUSAL_MESSAGE}"""


def build_grounded_prompt(question: str, chunks: list[RetrievedChunk]) -> list[Message]:
    """Assemble the chat messages sent to the LLM.

    Each chunk is numbered [1], [2], ... and shows its heading path, so the
    model can cite a number instead of copying a long label.

    Args:
        question: The user's question.
        chunks: Retrieved chunks, best first.

    Returns:
        A system message with the rules and a user message with excerpts and question.
    """
    excerpts = "\n\n---\n\n".join(
        f"[{number}] {build_stored_chunk_text(chunk.text, chunk.metadata)}"
        for number, chunk in enumerate(chunks, start=1)
    )
    user_message = f"Documentation excerpts:\n\n{excerpts}\n\n---\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
