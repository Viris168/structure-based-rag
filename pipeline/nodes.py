"""The three steps of the question-answering graph: retrieve, build_prompt, generate."""

from config.settings import REFUSAL_MESSAGE
from generation.citation import cited_sources
from generation.llm_client import generate_answer
from generation.prompt_builder import build_grounded_prompt
from pipeline.state import GraphState
from retrieval.hybrid_search import hybrid_search


def retrieve_node(state: GraphState) -> GraphState:
    """Find the chunks most relevant to the question.

    Args:
        state: Must contain "question".

    Returns:
        {"chunks": top-k chunks from hybrid search}.
    """
    return {"chunks": hybrid_search(state["question"])}


def build_prompt_node(state: GraphState) -> GraphState:
    """Turn the question and retrieved chunks into a grounded prompt.

    Args:
        state: Must contain "question" and "chunks".

    Returns:
        {"messages": the system and user messages}.
    """
    return {"messages": build_grounded_prompt(state["question"], state["chunks"])}


def generate_node(state: GraphState) -> GraphState:
    """Ask the LLM, then record which sources its answer cited.

    If nothing was retrieved (e.g. the store is empty), refuse without calling
    the LLM, since there is nothing for it to ground an answer in.

    Args:
        state: Must contain "chunks" and "messages".

    Returns:
        {"answer": reply or refusal, "citations": labels found in the reply}.
    """
    if not state["chunks"]:
        return {"answer": REFUSAL_MESSAGE, "citations": []}
    answer = generate_answer(state["messages"])
    return {"answer": answer, "citations": cited_sources(answer, state["chunks"])}
