"""Compares search setups side by side to show what each design choice adds."""

from config.settings import EVAL_QUESTIONS_PATH, TOP_K
from eval.hit_rate import SearchFunction, score_search
from eval.index_builder import build_memory_index
from eval.questions import load_questions
from retrieval.bm25_search import bm25_search
from retrieval.hybrid_search import hybrid_search_in
from retrieval.vector_search import vector_search


def run_comparison() -> str:
    """Score five setups on the same questions, from the original design to the full system.

    Setups:
        1. H2 only, no heading path, hybrid      (board design without fixes)
        2. H2 + H3, no heading path, hybrid      (adds the H3 rule)
        3. H2 + H3, heading path, vector only
        4. H2 + H3, heading path, BM25 only
        5. H2 + H3, heading path, hybrid         (the current system)

    Each index is built in memory from DOCS_DIR, so docs need not be ingested
    first and the real collection is never changed.

    Returns:
        A table of hit rate and MRR per setup.
    """
    questions = load_questions(EVAL_QUESTIONS_PATH)
    h2_only = build_memory_index(split_on_h3=False, include_heading_path=False)
    h3_no_path = build_memory_index(split_on_h3=True, include_heading_path=False)
    full = build_memory_index(split_on_h3=True, include_heading_path=True)

    setups: list[tuple[str, SearchFunction]] = [
        ("H2 only · no path · hybrid", lambda q: hybrid_search_in(h2_only, q, include_heading_path=False)),
        ("H2+H3   · no path · hybrid", lambda q: hybrid_search_in(h3_no_path, q, include_heading_path=False)),
        ("H2+H3   · path    · vector", lambda q: vector_search(full, q, TOP_K)),
        ("H2+H3   · path    · BM25", lambda q: bm25_search(full, q, TOP_K)),
        ("H2+H3   · path    · hybrid  (current)", lambda q: hybrid_search_in(full, q)),
    ]
    lines = [
        f"{len(questions)} questions, top {TOP_K}",
        "",
        f"{'setup':38} {'hit rate':>8}  {'MRR':>5}",
    ]
    for name, search in setups:
        result = score_search(search, questions)
        lines.append(f"{name:38} {result.hit_rate:>8.0%}  {result.mrr:>5.2f}")
    return "\n".join(lines)
