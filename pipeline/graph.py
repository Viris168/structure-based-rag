"""Connects the question-answering steps into a LangGraph graph."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pipeline.nodes import build_prompt_node, generate_node, retrieve_node
from pipeline.state import GraphState


def build_graph() -> CompiledStateGraph:
    """Build the graph: retrieve → build_prompt → generate.

    Returns:
        A compiled graph; call .invoke({"question": ...}) to run it.
    """
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "build_prompt")
    graph.add_edge("build_prompt", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
