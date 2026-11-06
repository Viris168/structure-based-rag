"""Scores a search function on the evaluation questions: hit rate and MRR."""

from collections.abc import Callable
from dataclasses import dataclass

from config.settings import EVAL_HIT_TARGET, EVAL_QUESTIONS_PATH, TOP_K
from eval.matching import rank_of_expected
from eval.questions import EvalQuestion, load_questions
from generation.citation import format_citation
from retrieval.hybrid_search import hybrid_search
from retrieval.models import RetrievedChunk

SearchFunction = Callable[[str], list[RetrievedChunk]]


@dataclass(frozen=True)
class EvalResult:
    """How one search setup did on every question.

    Attributes:
        ranks: Rank of the expected chunk per question (None = not in top-k).
        top_results: The top result per question, or None if nothing came back.
    """

    ranks: list[int | None]
    top_results: list[RetrievedChunk | None]

    @property
    def hit_rate(self) -> float:
        """Fraction of questions whose expected chunk is in the top-k."""
        return sum(rank is not None for rank in self.ranks) / len(self.ranks)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank: 1.0 if always first, 0.5 if always second, 0 if missing."""
        return sum(1.0 / rank for rank in self.ranks if rank is not None) / len(self.ranks)


def score_search(search: SearchFunction, questions: list[EvalQuestion]) -> EvalResult:
    """Run every question through a search function and record where the answer landed.

    Args:
        search: Takes a question, returns up to TOP_K chunks, best first.
        questions: The evaluation questions.

    Returns:
        Per-question ranks and top results.
    """
    ranks: list[int | None] = []
    top_results: list[RetrievedChunk | None] = []
    for question in questions:
        results = search(question.question)
        ranks.append(rank_of_expected(question, results))
        top_results.append(results[0] if results else None)
    return EvalResult(ranks=ranks, top_results=top_results)


def run_eval() -> str:
    """Score the real system (hybrid search on the ingested docs) against EVAL_HIT_TARGET.

    Returns:
        Hit rate, MRR, pass/fail, and each missed or not-first question with
        what came back instead.
    """
    questions = load_questions(EVAL_QUESTIONS_PATH)
    result = score_search(hybrid_search, questions)
    verdict = "PASS" if result.hit_rate >= EVAL_HIT_TARGET else "FAIL"
    lines = [
        f"Questions: {len(questions)}",
        f"Hit rate (top {TOP_K}): {result.hit_rate:.0%}  (target {EVAL_HIT_TARGET:.0%})  {verdict}",
        f"MRR: {result.mrr:.2f}",
    ]
    problems = [
        (question, rank, top)
        for question, rank, top in zip(questions, result.ranks, result.top_results)
        if rank != 1
    ]
    if problems:
        lines += ["", "Not ranked first:"]
        for question, rank, top in problems:
            found = format_citation(top.metadata) if top else "nothing"
            lines.append(f"  [{'miss' if rank is None else f'rank {rank}'}] {question.question}")
            lines.append(f"      top result: {found}")
    return "\n".join(lines)
