"""Demo: sync the docs into ChromaDB. Run with `python -m demo.demo_ingest`."""

from pipeline.ingest import run_ingestion

if __name__ == "__main__":
    print(run_ingestion().summary())
