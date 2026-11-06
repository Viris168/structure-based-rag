"""Phase 2 pipeline: answer a question with the graph and format the result."""

from pipeline.graph import build_graph


def answer_question(question: str) -> str:
    """Run the question through retrieve → build_prompt → generate.

    Args:
        question: A plain-English question about the docs.

    Returns:
        The answer, followed by the list of sources it cited (if any).
    """
    state = build_graph().invoke({"question": question})
    lines = [state["answer"]]
    if state["citations"]:
        lines += ["", "Sources:"] + [f"  {citation}" for citation in state["citations"]]
    return "\n".join(lines)
