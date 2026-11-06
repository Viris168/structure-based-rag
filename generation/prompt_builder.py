"""Builds the grounded prompt: the question plus the retrieved chunks, nothing else."""

from config.settings import REFUSAL_MESSAGE
from generation.citation import format_citation
from retrieval.models import RetrievedChunk

Message = dict[str, str]

_SYSTEM_PROMPT = f"""You are a documentation assistant. Answer the user's question using ONLY the documentation excerpts provided.

Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge.
- After each claim, cite the excerpt it came from by copying that excerpt's label exactly, e.g. [Source: Doc · Section: Heading].
- Keep code, commands, and identifiers exactly as written in the excerpts.
- If the excerpts do not contain the answer, reply with exactly this sentence and nothing else: {REFUSAL_MESSAGE}"""


def build_grounded_prompt(question: str, chunks: list[RetrievedChunk]) -> list[Message]:
    """Assemble the chat messages sent to the LLM.

    Each chunk is labelled with its citation so the model can cite it word for word.

    Args:
        question: The user's question.
        chunks: Retrieved chunks, best first.

    Returns:
        A system message with the rules and a user message with excerpts and question.
    """
    excerpts = "\n\n---\n\n".join(f"{format_citation(chunk.metadata)}\n{chunk.text}" for chunk in chunks)
    user_message = f"Documentation excerpts:\n\n{excerpts}\n\n---\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
