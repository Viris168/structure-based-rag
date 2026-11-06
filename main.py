"""Command-line entry point.

Usage:
    python main.py chunks              # preview how the docs will be chunked
    python main.py ingest              # chunk, embed, and store the docs in ChromaDB
    python main.py search "<query>"    # show the top-k hybrid search results
    python main.py ask "<question>"    # answer a question with citations
    python main.py eval                # score search on eval/questions.json
    python main.py compare             # compare search setups on the same questions
"""

import argparse

from eval.compare import run_comparison
from eval.hit_rate import run_eval
from pipeline.ask import answer_question
from pipeline.chunk_preview import preview_chunks
from pipeline.ingest import run_ingestion
from pipeline.search_preview import preview_search


def main() -> None:
    """Parse the command and run it."""
    parser = argparse.ArgumentParser(description="RAG documentation assistant")
    parser.add_argument("command", choices=["chunks", "ingest", "search", "ask", "eval", "compare"])
    parser.add_argument("text", nargs="?", help="the query (search) or question (ask)")
    args = parser.parse_args()

    if args.command in ("search", "ask") and not args.text:
        parser.error(f'"{args.command}" needs a question, e.g. python main.py {args.command} "How do I install it?"')

    if args.command == "chunks":
        print(preview_chunks())
    elif args.command == "ingest":
        print(run_ingestion().summary())
    elif args.command == "search":
        print(preview_search(args.text))
    elif args.command == "ask":
        print(answer_question(args.text))
    elif args.command == "eval":
        print(run_eval())
    elif args.command == "compare":
        print(run_comparison())


if __name__ == "__main__":
    main()
