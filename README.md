# lite-rag

A small, fully local RAG engine built for learning: every step is plain code you can read, swap and
measure.

- **Chunking**: configurable per knowledge base: `fixed`, `recursive` (default), `sentence`,
  `paragraph`, `markdown`, `semantic` (own, or chonkie), `parent_child`, plus an optional
  LLM-written context per chunk (`contextual`). Lengths are counted with the embedding model's own
  tokenizer, and `size` is a hard limit.
- **Embedding**: local (Ollama, fastembed, sentence-transformers) or API (any OpenAI-compatible
  endpoint, Voyage, Cohere, Gemini). Each knowledge base is bound to one model; model-specific
  query and document prefixes are applied for you.
- **Storage**: PostgreSQL + pgvector (vectors, BM25 postings and the document registry written in
  one transaction), or Chroma + SQLite when you don't want a database server.
- **Retrieval**: vector search plus an own BM25 (jieba for Chinese), fused with RRF; an optional
  cross-encoder rerank. Optional graph retrieval (local GraphRAG, PostgreSQL only): an LLM lists
  each chunk's entities and relations at ingest, and a `graph` path walks them from the entities a
  query names — for questions whose answer is a few documents away from what they mention. Every
  hit carries each stage's score and rank.
- **Generation**: answers with validated citations through a switchable LLM (Ollama, any
  OpenAI-compatible endpoint, or Anthropic). No fallback chain; every answer records
  `provider/model`.
- **Evaluation**: `literag eval` compares configurations side by side (hit@1, hit@5, MRR@10,
  latency) on a bundled Chinese and English corpus, or your own. It runs on
  [lite-eval](https://github.com/ArthurGao/lite-eval): each config is a lite-eval run, so two can
  be compared with significance (`lite-eval compare`).

Use it as a Python library or as a service (`literag serve`, Docker image, HTTP API with API keys).
The default stack (Ollama `bge-m3` + `qwen3:8b`, PostgreSQL/pgvector, jieba) runs locally and costs
nothing. [llm-steward](https://github.com/ArthurGao/llm-steward) uses it as its knowledge-base
engine.

Status: v0.1, stages S0–S11 of the [design](docs/lite-rag-design.md) are done (S9: llm-steward
integration; S10: searching several kbs as one; S11: graph retrieval, local mode).

## How it works

```
ingest:  file / text → loader → chunker → chunks → embedder ─┐
                                             ├→ tokenize (jieba) ┤
                                             └→ [LLM: entities, relations] ┴→ store (one transaction: registry + vectors + postings + graph)
search:  query → embedder → vector top-n ┐
         query → tokenize → BM25 top-n   ┤
         query → named entities → walk co-mentions → reached chunks by vector ┴→ RRF → [rerank] → top-k hits
ask:     hits → prompt (prompts/*.j2) → LLM → answer (status, text, citations, provenance)
```

A few rules the engine enforces:

- **Namespaces are mandatory.** An engine is bound to one namespace (`project` or
  `project/tenant`), and every kb is stored as `<namespace>:<kb>`. No API takes a raw key.
- **An index is bound to its model.** A kb records its embedding provider, model, dimension,
  prefixes and chunker; opening it with a different config raises. To switch, build a new kb
  (`literag reindex <kb> --into <new>`).
- **Stable ids and idempotent writes.** Chunk id = `{doc_id}#{n}`; re-upserting a document replaces
  its chunks in both indexes, and an unchanged document is a no-op.
- **No silent fallbacks.** A failed model call raises; citations outside the retrieved sources are
  removed and reported.

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) on the
host, and PostgreSQL with pgvector ≥ 0.8 (or use the Chroma backend, below).

```bash
ollama pull bge-m3 && ollama pull qwen3:8b
uv sync
cp .env.example .env              # set LITERAG_DATABASE_URL
uv run literag migrate            # create the schema (lite_rag by default)

export LITERAG_NAMESPACE=demo
uv run literag ingest docs eval/corpus
uv run literag search docs "How long should coffee steep in a French press?"
uv run literag ask docs "垃圾分类有哪几类？"
```

No PostgreSQL? Use the embedded backend:

```bash
uv sync --extra chroma
export LITERAG_STORE__BACKEND=chroma LITERAG_STORE__DATA_DIR=./lite_rag_data
uv run literag migrate
```

For a faster `qwen3` on Ollama, turn off its thinking: `LITERAG_LLM__REASONING_EFFORT=none`.

## CLI

Every command takes `-c/--config <file.yaml>` and `-n/--namespace` (or `LITERAG_NAMESPACE`).

| Command | What it does |
| --- | --- |
| `literag migrate` | Create or upgrade the store (PG schema or the Chroma data directory) |
| `literag config` | Show the effective settings (secrets masked) |
| `literag ingest <kb> <paths…>` | Load, chunk, embed and index files or directories (`.md`, `.txt`, `.pdf`, `.docx`); unchanged files are skipped |
| `literag search <kb> "<q>" [-k 5] [-m vector\|bm25\|graph\|hybrid] [--rerank] [--json]` | Hits with each stage's score and rank |
| `literag ask <kb> "<q>" [--json]` | Answer with citations, model and token usage |
| `literag stats <kb>` | Size of a kb and how it was built |
| `literag delete <kb> <doc_id>` · `literag drop-kb <kb> --yes` | Remove a document or a whole kb |
| `literag reindex <kb> --into <new-kb>` | Rebuild under the current settings (another model or chunker) from the stored text |
| `literag eval -c a.yaml -c b.yaml` | Compare configurations on a corpus and golden questions |
| `literag key create --name <who> --namespace '<glob>' [--kb '<glob>'] --mode read\|write` | Create an API key for the service |
| `literag serve [--host] [--port]` | Run the HTTP API (extra `server`) |

## Library

```python
from lite_rag.config import load_settings
from lite_rag.engine import RagEngine, RagRuntime

settings = load_settings("rag.yaml")  # or load_settings() for environment only

engine = RagEngine.from_settings(settings, namespace="myproject/tenant_42")
engine.upsert_text("handbook", "leave-policy", "Leave policy", text, metadata={"team": "hr"})
hits = engine.search(
    "handbook",
    "how many days of annual leave?",
    top_k=5,
    mode="hybrid",
    rerank=False,
    where={"team": "hr"},
)
# with graph.enabled (PostgreSQL): mode="graph", or "graph" in retrieval.hybrid_paths
# several kbs of the namespace as one: embedded once, fused once; each hit names its kb
hits = engine.search_kbs(["handbook", "faq"], "annual leave", top_k=5)
answer = engine.ask("handbook", "How many days of annual leave do I get?")
print(answer.text, [c.chunk_id for c in answer.citations], answer.model_id)
engine.close()

# Several namespaces in one process (what the HTTP service does): build the runtime once,
# take a light per-namespace engine per request.
runtime = RagRuntime.from_settings(settings)
engine = runtime.engine("myproject/tenant_7")
```

`where` filters are equality on scalar metadata values (str, int, float, bool). Errors are typed
(`lite_rag.errors`): `KbNotFound`, `DocumentNotFound`, `KbMismatch`, `InvalidRequest`,
`ConfigError`, `StoreError`. The package ships `py.typed`.

Install from git: `pip install "lite-rag[server] @ git+https://github.com/ArthurGao/lite-rag"`.

## HTTP service

```bash
uv run literag key create --name me --namespace 'demo/*' --mode write   # prints the key once
export LITERAG_SERVER__KEYS='[ ...the JSON line it printed... ]'
uv run --extra server literag serve                                     # 127.0.0.1:8700
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` (`?deep=true`) | Store and embedding service reachable; no auth |
| `GET /v1/kbs` · `PUT/GET/DELETE /v1/kbs/{kb}` | List, create, stats, delete |
| `PUT /v1/kbs/{kb}/documents/{doc_id}` | Upsert a text document (`title`, `text`, `metadata`) |
| `POST /v1/kbs/{kb}/files` | Upload a file (multipart `file`, optional `doc_id`) |
| `DELETE /v1/kbs/{kb}/documents/{doc_id}` | Delete a document |
| `POST /v1/kbs/{kb}/search` | `{query, top_k, mode, rerank, where}` → hits |
| `POST /v1/search` | `{kbs, query, top_k, mode, rerank, where}` → hits across several kbs (each hit names its `kb`); the key must cover every kb |
| `POST /v1/kbs/{kb}/ask` | `{question, top_k}` → answer with citations |

```bash
curl -s http://127.0.0.1:8700/v1/kbs/docs/search \
  -H "Authorization: Bearer $KEY" -H "X-LiteRag-Namespace: demo/app" \
  -H 'content-type: application/json' -d '{"query": "french press", "top_k": 3}'
```

- **Auth**: `Authorization: Bearer lrk_…`. The server stores only each key's sha256; a key has
  namespace and kb globs and a `read` or `write` mode. No route creates keys.
- **Namespace** comes from the `X-LiteRag-Namespace` header. 401 without a valid key, 403 outside
  the key's scope.
- **Errors** are RFC 9457 Problem Details. Upstream and configuration error details are logged, not
  returned.
- **Without keys** the service is open, so it may only bind a loopback address and accepts only
  local `Host` headers. Request bodies are size-limited (`server.max_body_mb`,
  `server.max_upload_mb`).

### Docker

```bash
uv run literag key create --name me --namespace 'demo/*' --mode write
cp deploy/.env.example deploy/.env       # DB password and the key entry
docker compose -f deploy/compose.yaml --env-file deploy/.env up -d --build
```

The compose file runs lite-rag and PostgreSQL (pgvector); Ollama stays on the host
(`host.docker.internal`). The image includes the `server` and `pdf` extras and runs
`literag migrate` on start.

## Configuration

Settings come from `LITERAG_*` environment variables (nested with `__`, e.g.
`LITERAG_LLM__PROVIDER`), `.env`, or a YAML file passed with `-c`:

```yaml
store: {backend: postgres, schema: lite_rag}          # or {backend: chroma, data_dir: ./lite_rag_data}
embedding: {provider: ollama, model: bge-m3}           # ollama | openai | voyage | cohere | gemini | fastembed | sentence_transformers
chunking: {strategy: recursive, unit: tokens, size: 800, overlap: 120, min_size: 50}
retrieval: {mode: hybrid, hybrid_paths: [vector, bm25], top_k: 5, candidate_factor: 4, fusion: rrf, rrf_k: 60}
graph: {enabled: false, llm: {model: qwen3:8b}, hops: 2, hub_chunks: 5}   # add graph to hybrid_paths, or use mode graph
rerank: {enabled: false, model: BAAI/bge-reranker-v2-m3-int8, top_n: 20}
llm: {provider: ollama, model: qwen3:8b}               # ollama | openai | anthropic
server: {host: 127.0.0.1, port: 8700}
loader: {pdf: pypdf}                                   # or docling
```

Keep `LITERAG_DATABASE_URL` and API keys in the environment; a provider's key is read from
`api_key_env` or its usual variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`,
`CO_API_KEY`, `GEMINI_API_KEY`). See [.env.example](.env.example).

### Extras

| Extra | Adds |
| --- | --- |
| `server` | FastAPI service (`literag serve`) |
| `chroma` | Chroma + SQLite store |
| `fastembed` | In-process ONNX embedders and the cross-encoder reranker |
| `sentence-transformers` | Any Hugging Face embedding model via PyTorch |
| `anthropic` | Anthropic LLM |
| `pdf` | PDF text extraction with pypdf |
| `docling` | PDF / DOCX to Markdown with docling (large: pulls in torch) |
| `chonkie` | chonkie's semantic chunker, for comparison |

## Evaluation

Retrieval changes are measured, not guessed. Golden answers are marked by evidence text, not chunk
ids, so the same questions compare different chunkers. Bundled sets (Chinese and English):

| Set | Corpus | Questions | Tests |
| --- | --- | --- | --- |
| `eval/golden.jsonl` | `eval/corpus` (8 docs) | 32: keyword and paraphrased | everyday retrieval |
| `eval/golden-exact.jsonl` | `eval/corpus-exact` (2 docs) | 12: error codes, config keys, model numbers | where BM25 should help |
| `eval/golden-multihop.jsonl` | `eval/corpus-multihop` (8 docs) | 16: the answer 2–3 documents away | graph retrieval |
| `eval/golden-pdf.jsonl` | `eval/corpus-pdf` (2 PDFs) | 10 | PDF loaders |

```bash
uv run --all-extras literag eval -c eval/configs/chunk-markdown.yaml -c eval/configs/mode-vector.yaml
```

See [eval/configs/README.md](eval/configs/README.md) for the bundled configs. Results so far
(bge-m3, 120-token chunks; details in design §13):

| Config | hit@1 | hit@5 | MRR@10 | p50 ms |
| --- | --- | --- | --- | --- |
| markdown + hybrid | 0.91 | 0.94 | 0.930 | 5 |
| markdown + vector only | 0.97 | 1.00 | 0.979 | 3 |
| markdown + BM25 only | 0.75 | 0.91 | 0.823 | 2 |
| markdown + hybrid + rerank | 0.97 | 1.00 | 0.979 | 1061 |

On this small corpus vector search alone matches hybrid + rerank; 32 questions are too few to
separate close configurations, so the defaults are unchanged until the corpus grows.

Graph retrieval (S11) on 12 multi-hop questions, where the answer sits two or three documents away
from what the question names (`eval/golden-multihop.jsonl`, local qwen3 extraction):

| Config | multi-hop hit@5 | MRR@10 | p50 ms |
| --- | --- | --- | --- |
| markdown + vector only | 0.50 | 0.373 | 6 |
| markdown + hybrid + rerank | 0.33 | 0.347 | 861 |
| markdown + graph (alone, or fused with vector / BM25) | 1.00 | 0.589 | 10–15 |
| markdown + graph + rerank | 0.25 | 0.381 | 768 |

The graph puts the first link of the chain first and the answer right after it, so hit@1 stays at
0 for these questions — both chunks are what an answer needs. Rerank undoes the gain: it scores the
question against one chunk at a time and prefers look-alikes (another council) over the answer.
On the 32 ordinary questions the graph costs one question (vector 0.97 → 0.94 hit@1).

### Choosing a retrieval mode

What the measurements above and a 4 embedders × 5 chunkers × 4 modes matrix (80 configs, 44
questions) say. The numbers are small corpora: re-run `literag eval` on yours before you rely on
them.

- **The embedder decides vector vs hybrid.** With a strong embedder (bge-m3, multilingual-e5-large,
  Qwen3-Embedding) `vector` alone beats `hybrid`: BM25's wrong hits push paraphrased answers down
  through RRF, and BM25 never ranked an answer above vector — not even on error codes and model
  numbers. With a weak one (MiniLM) `hybrid` wins in every chunker.
- **Rerank flattens the embedder, not the chunker.** With the cross-encoder even MiniLM reaches the
  best score (0.95 hit@1), at ~1 s per query; recursive or markdown 120-token chunks do best, small
  40-token chunks worst.
- **Match the chunker to the embedder.** bge-m3 is 0.95 on markdown or parent_child chunks but
  0.82 on fixed or recursive ones; e5 prefers recursive.
- **Small chunks need a parent.** markdown 40 is last in every row; parent_child 40/120 recovers it.
- **No automatic routing.** A perfect per-question choice between vector and BM25 scores exactly
  vector alone with bge-m3, so `mode` stays a per-call parameter with a configured default.
- **Questions that chain facts need the graph**, and no rerank on top of it (see above).

| You want | Use |
| --- | --- |
| best quality, ~1 s per query is fine | recursive or markdown chunks + `hybrid` + rerank |
| milliseconds, strong embedder | markdown (or parent_child) chunks + `vector` |
| milliseconds, small or weak embedder | recursive or markdown chunks + `hybrid` |
| questions across documents ("the council chaired by X's lead") | `graph.enabled` + `hybrid_paths: [vector, bm25, graph]`, rerank off |

## Development

```bash
uv sync --all-extras
uv run ruff check && uv run ruff format --check && uv run mypy
uv run pytest                          # unit, contract and integration tests, fake models only
uv run pytest -m ollama                # opt-in: real Ollama (bge-m3 and qwen3:8b pulled)
uv run --all-extras pytest -m model    # opt-in: real Hugging Face models (the reranker)
```

Integration and contract tests need `LITERAG_TEST_DATABASE_URL`; they use (and drop) the schema
`lite_rag_test` only. Tests never call a real model or paid API unless you opt in with a marker.

```
src/lite_rag/
  engine.py              RagRuntime + per-namespace RagEngine
  chunking.py            chunkers (contextual.py, chunking_chonkie.py)
  embedding.py           embedders; model_profiles.py holds prefixes and tokenizers
  keyword.py fusion.py   BM25 and RRF / weighted fusion
  graph.py               graph retrieval: entity extraction, the co-mention walk
  rerank.py llm.py answer.py prompts/*.j2
  stores/                Store protocol, pg.py, chroma.py, factory.py
  migrations/            PostgreSQL schema; migrations_sqlite/ for Chroma
  server/                FastAPI app and API-key auth
  eval.py cli.py config.py loaders.py testing.py (fakes)
tests/   unit/  contract/ (same cases on both stores)  integration/
eval/    corpus, golden questions, configs
docs/lite-rag-design.md   the design (in Chinese); AGENTS.md for contributors and coding agents
```

## License

Apache-2.0.
