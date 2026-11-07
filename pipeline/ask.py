"""Phase 2 pipeline: answer a question with the graph and format the result."""

from pipeline.graph import build_graph
from pipeline.state import GraphState


def run_question(question: str) -> GraphState:
    """Run the question through retrieve → build_prompt → generate.

    Args:
        question: A plain-English question about the docs.

    Returns:
        The final graph state, including "answer" and "citations".
    """
    return build_graph().invoke({"question": question})


def answer_question(question: str) -> str:
    """Answer a question and format the result for the terminal.

    Args:
        question: A plain-English question about the docs.

    Returns:
        The answer, followed by the list of sources it cited (if any).
    """
    state = run_question(question)
    lines = [state["answer"]]
    if state["citations"]:
        lines += ["", "Sources:"] + [f"  {citation}" for citation in state["citations"]]
    return "\n".join(lines)
