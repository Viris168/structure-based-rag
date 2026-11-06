"""Demo: show the top-k hybrid search results. Run with `python -m demo.demo_search`."""

from config.settings import DEMO_QUESTION
from pipeline.search_preview import preview_search

if __name__ == "__main__":
    print(preview_search(DEMO_QUESTION))
