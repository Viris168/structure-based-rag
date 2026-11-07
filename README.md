# Structure-Based RAG Documentation Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about Markdown documentation, with citations. It is a teaching project: every concept lives in its own module, every function is called somewhere, and every constant is defined once in `config/settings.py`.

The core idea is **structure-based chunking**. Instead of cutting documents every N characters, it splits them along their own headings, so each chunk is one coherent section. Bigger chunks give worse answers, so sections that are too long are split further.

## How it works

```
Phase 1 — Ingestion                         Phase 2 — Question answering
───────────────────                         ────────────────────────────
docs/*.md                                   User question
   │                                           │
   ▼                                           ▼
Split on ## headings                        Hybrid search
   │  section > 400 words?                    vector (bge-small) + BM25 keywords
   ▼  → split on ### headings                 merged with Reciprocal Rank Fusion
Fit to model limit                             │
   │  still > 512 tokens?                      ▼
   ▼  → split at paragraphs                 Rerank the top 20 (cross-encoder)
Merge tiny chunks                              │
   │  < 40 tokens → same-section neighbour     ▼
   ▼                                        Top 5 chunks
Prepend heading path                           │
   │  "Doc > Section > Subsection"             ▼
   ▼                                        LangGraph: retrieve → build_prompt → generate
Embed (BAAI/bge-small-en-v1.5)                 │
   │                                           ▼
   ▼                                        Answer from the chunks only, with
ChromaDB (persistent, chroma_data/)         [Source: doc · Section: heading]
   only changed docs are re-indexed         or a refusal
```

### Chunking rules

1. Split each document on `##` headings. Text before the first `##` becomes its own chunk, using the document title as its section.
2. If a section is over **400 words**, split it on its `###` headings. Text between the `##` and the first `###` is kept as an "intro" chunk.
3. If a piece is still too long for the embedding model (512 tokens, counting its heading path), split it at paragraph boundaries. Code blocks and tables stay whole unless one of them is longer than the limit by itself; then it is split at line breaks.
4. A chunk under **40 tokens** is merged into the chunk before it (or after it) in the same `##` section, and the subsection labels are joined, e.g. `Usage > Setting Custom Colors / Restoring PATH`. A tiny section with no same-section neighbour (a one-line `## License`) is kept as it is, since merging it into another section would give it the wrong citation.
5. Headings inside code blocks (for example a `## comment` in a shell snippet) are ignored.

### What is stored for each chunk

| Field | Contents |
|---|---|
| Document | The raw Markdown of the chunk, unchanged |
| Embedding | 384-dimension vector of `heading path + chunk text` |
| Metadata | `doc_title`, `section_title`, `subsection`, `chunk_index`, `word_count`, `token_count`, `content_hash`, `source_path`, `file_hash` |

### Design choices

- **Hybrid search.** Vector search finds paraphrases ("how do I set it up?"); BM25 finds exact names (`get_or_create_collection`, `MCP_AUTH_TOKEN`, error codes). Reciprocal Rank Fusion combines them using ranks, since their scores are on different scales.
- **Reranker.** Vector search compares two vectors made separately, so it judges "same topic", not "answers the question"; a general "Usage" intro often outranks the subsection that actually answers. A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 90 MB) reads the question and each chunk together and re-scores the top 20 fused results. It is too slow to run on every chunk, which is why it only re-scores candidates.
- **bge-small instead of all-MiniLM-L6-v2.** MiniLM reads only 256 tokens (about 190 words), so the second half of a 400-word chunk would never be embedded. bge-small reads 512.
- **Heading path in the embedding.** A chunk like "Run `pip install`…" means little on its own; "MCP Documentation > Installation" in front of it tells the model what it is about.
- **Incremental ingestion.** Each document's hash is stored. Unchanged documents are skipped; changed documents have all their chunks replaced, so no stale chunks remain; deleted documents are removed.
- **Grounded answers.** The model is told to answer only from the retrieved excerpts, to cite them by number ([1], [2]; the code swaps each number for the exact source label, so a citation can never be misspelled), and to reply with a fixed refusal sentence when the excerpts do not contain the answer. If nothing is retrieved, it refuses without calling the model.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Then:

1. Add your DeepSeek API key to `.env` as `DEEPSEEK_API_KEY=...`. The LLM is reached through the `openai` SDK, so any OpenAI-compatible provider works by changing `LLM_BASE_URL`, `LLM_API_KEY_ENV`, and `LLM_MODEL` in `config/settings.py`.
2. Put your Markdown files in `docs/`.

The embedding model downloads automatically on first use.

## Usage

| Command | What it does |
|---|---|
| `python main.py chunks` | Show how the docs will be chunked, without storing anything |
| `python main.py ingest` | Chunk, embed, and store the docs; reports added, updated, skipped, and removed files |
| `python main.py search "<query>"` | Show the top 5 search results, without calling the LLM |
| `python main.py ask "<question>"` | Answer a question with citations |
| `python main.py eval` | Score search against `eval/questions.json` |
| `python main.py compare` | Compare seven search setups on the same questions |
| `python main.py chat` | Open a chat page at http://127.0.0.1:7860 (stop with Ctrl+C) |

Example:

```bash
python main.py ingest
python main.py ask "How do I install the SDK?"
```

Each command also has a demo script in `demo/` (for example `python -m demo.demo_ask`) that calls one function, for walking through the project step by step.

## Evaluation

The evaluation measures whether search finds the right section, not whether the LLM's answer is good.

1. Copy `eval/questions.example.json` to `eval/questions.json`.
2. Write 15–20 questions about your docs, each with the section that answers it:

   ```json
   {
     "question": "How do remote servers authenticate clients?",
     "source_path": "MCP_Documentation.md",
     "section_title": "Server Setup",
     "subsection": "Authentication"
   }
   ```

   `subsection` is optional; leave it out if the answer can be anywhere in the section. Include some questions that ask for exact names, and some about subsections.

3. Run `python main.py eval`. It reports:
   - **Hit rate**: the share of questions whose section is in the top 5. The target is 85% (`EVAL_HIT_TARGET`).
   - **MRR**: 1.0 if the right section is always first, 0.5 if always second.
   - Every question that did not rank first, and what came back instead.

4. Run `python main.py compare` to see what each design choice adds. It builds throwaway in-memory indexes (your real store is not touched) and scores:

   | Setup | What it shows |
   |---|---|
   | H2 only · no heading path · hybrid | The original design, without fixes |
   | H2 + H3 · no heading path · hybrid | What the 400-word H3 rule adds |
   | H2 + H3 · heading path · vector only | Vector search on its own |
   | H2 + H3 · heading path · BM25 only | Keyword search on its own |
   | H2 + H3 · heading path · hybrid | What the heading path adds |
   | + merge small chunks | What merging tiny chunks adds |
   | + merge + reranker | The current system |

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The tests use their own sample docs in `tests/fixtures/` and a temporary store, so they never touch `docs/` or `chroma_data/`. The LLM is replaced with a fake, so no API key is needed. They cover chunking, ingestion, retrieval, prompt and citation handling, the LangGraph pipeline, the CLI, and the evaluation code.

## Project structure

```
config/settings.py          every constant: paths, models, thresholds, top-k
chunking/
  loader.py                 read .md files and their titles
  header_splitter.py        split on ## or ### (ignoring code blocks)
  markdown_syntax.py        fence, heading, and empty-text helpers
  paragraph_splitter.py     split oversized text at paragraph boundaries
  hybrid_chunker.py         apply the H2 / H3 / token-limit rules
  chunk_merger.py           merge tiny chunks into a same-section neighbour
  text_size.py              word and model-token counts
  context_header.py         heading path and embedding text
  metadata.py               metadata stored with each chunk
  models.py                 SourceDocument, Section, Chunk
vectorstore/
  embedder.py               embed chunks and queries
  chroma_client.py          open the persistent (or in-memory) collection
  doc_hasher.py             content hashes for change detection
  indexer.py                replace a document's chunks; remove deleted docs
retrieval/
  vector_search.py          similarity search
  bm25_search.py            keyword search
  rrf.py                    Reciprocal Rank Fusion
  reranker.py               cross-encoder re-scoring of the candidates
  hybrid_search.py          vector + BM25 → fuse → rerank → top-k
  models.py                 RetrievedChunk
generation/
  prompt_builder.py         grounded prompt with labelled excerpts
  llm_client.py             call the LLM
  citation.py               citation labels; find the ones an answer used
pipeline/
  ingest.py                 Phase 1: load → chunk → embed → store
  state.py, nodes.py        LangGraph state and steps
  graph.py                  retrieve → build_prompt → generate
  ask.py                    run the graph and format the answer
  chunk_preview.py          output for `main.py chunks`
  search_preview.py         output for `main.py search`
eval/
  questions.py              load the questions file
  matching.py               is this result the expected section?
  hit_rate.py               hit rate, MRR, and the eval report
  index_builder.py          in-memory indexes with features on or off
  compare.py                the seven-setup comparison
ui/
  chat_app.py               browser chat page (Gradio)
demo/                       one thin script per command
tests/                      pytest suite and fixtures
main.py                     command-line entry point
```

## Configuration

All settings are in `config/settings.py`. The ones you are most likely to change:

| Setting | Default | Meaning |
|---|---|---|
| `H3_SPLIT_THRESHOLD` | 400 | Words before a section is split on `###` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model |
| `MAX_EMBED_TOKENS` | 512 | Model input limit; chunks are kept under it |
| `TOP_K` | 5 | Chunks given to the LLM |
| `SEARCH_CANDIDATES` | 20 | Results each search contributes before fusion |
| `RRF_K` | 60 | Fusion constant |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder that re-scores candidates |
| `RERANK_CANDIDATES` | 20 | Fused results the reranker re-scores |
| `MIN_CHUNK_TOKENS` | 40 | Chunks smaller than this merge into a same-section neighbour |
| `LLM_MODEL` | `deepseek-chat` | Model used for answers |
| `EVAL_HIT_TARGET` | 0.85 | Hit rate `eval` must reach to pass |

If you change the embedding model or the chunking settings, delete `chroma_data/` and run `ingest` again; unchanged documents are otherwise skipped and keep their old chunks.
