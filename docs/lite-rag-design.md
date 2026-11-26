# lite-rag Design Document

> Status: S0–S19 done (2026-11-08 to 11-22; S18 = lite-cache L1, S19 = the ollama / openai embeddings now go through lite-model). Work follows the stages in section 12; when a stage changes behaviour, this document is updated in the same commit.

## 1. Overview

lite-rag is a lightweight RAG engine that is **readable, changeable and deployable on its own**. It has two uses:

1. **Learning**: chunking, embedding, BM25, fusion, rerank and generation are each code written in this repo. Each step can be replaced alone, and the effect can be compared with numbers.
2. **General-purpose RAG**: it can be installed and used as a Python library (`pip install`), or run as a standalone service (`literag serve`, or the Docker image) and called over an HTTP API. A caller does not need the source of this repo. [llm-steward](https://github.com/ArthurGao/llm-steward) is the first host: it does governance, lite-rag does storage and retrieval.

### Goals

- **Zero cost by default**: Ollama provides embedding and the LLM; storage is PostgreSQL + pgvector, or Chroma for a purely embedded run.
- **One core for library and service**: `RagEngine` is the only implementation. The HTTP service is a thin wrapper around it, so both behave the same.
- **Every step is pluggable**: `Chunker`, `Embedder`, `Store`, `Reranker` and `LLM` are Protocols; swapping an implementation is a config change only.
- **Measurable**: it ships its own eval (hit@k, MRR, latency); the same questions are compared side by side under different configs.
- **Switchable LLM**: `ollama`, `openai` (and any OpenAI-compatible endpoint, such as llm-gateway) and `anthropic`, by config only.

### Non-goals (v1)

- **No governance**: review, state machines and user permissions are left to a higher-level system such as llm-steward. lite-rag only does namespace isolation (see §4) and server-side API key authentication.
- **No complex document parsing**: OCR, layout analysis and table structure are left to engines such as RAGFlow. v1 supports only txt and md; PDF gets plain-text extraction only, as an optional extra.
- **No distribution**: one PG instance is enough. The target size is on the order of a hundred thousand chunks per kb.

## 2. Data flow

```
Ingest: file/text → Loader → Document → Chunker → chunks → Embedder ─┐
                                                   ├→ tokenize(jieba) ────┤
                                                   └→ [LLM extracts entities/relations] ┴→ Store (one transaction: registry + vectors + inverted index + graph)
Search:  query → Embedder → vector top-n ┐
        query → tokenize → BM25 top-n ┤
        query → seed entities → expand by co-occurrence → rank by vector in range ┴→ fusion(RRF) → [Reranker] → top-k Hit
Answer: Hit[] → prompt template(answer_system.j2 + answer_user.j2) → LLM → Answer(status, text, citations, provenance)
```

## 3. Components and interfaces

All interfaces are `typing.Protocol`; tests use fake implementations and need no network or model.

| Interface | Methods | Default implementation | Other implementations |
| --- | --- | --- | --- |
| `Loader` | `load_file(path, pdf) -> Loaded(title, text, pages)` | txt, md; pdf uses pypdf (extra `[pdf]`, plain text page by page) | `docling` (extra `[docling]`, `loader.pdf: docling`): converts PDF and DOCX to Markdown and keeps headings; `.docx` always goes through it |
| `Chunker` | `split(doc) -> list[Chunk]`; `split_parents(doc) -> (parents, chunks)` | `recursive` | See §3.1: fixed, sentence, paragraph, markdown, semantic, parent-child, plus the contextual enhancer |
| `Embedder` | `embed_documents(texts)`, `embed_query(text)`, `dim`, `model` | `ollama` (`bge-m3`) | See §3.2: openai-compatible, voyage, cohere, gemini, fastembed, sentence-transformers, `FakeEmbedder` |
| `Store` | `get_or_create_kb`, `get_kb`, `upsert_document`, `delete_document`, `vector_query` (`among` limits the chunk range), `bm25_query`, `chunks`, `stats`; graph retrieval has a separate `GraphStore` (only PgStore implements it): `graph_names_in`, `graph_chunks_of`, `graph_entities_of` | `PgStore` (pgvector holds vectors; PG tables hold the BM25 inverted index and the document registry) | `ChromaStore` (extra `[chroma]`): vectors in Chroma, inverted index and registry in SQLite |
| `Reranker` | `score(query, texts) -> list[float]` (scores only; ordering and context assembly are the engine's job) | Not enabled (`rerank.enabled=false`) | Local cross-encoder `BAAI/bge-reranker-v2-m3-int8` (extra `[fastembed]`); Voyage `rerank-2.5` (`provider: voyage`, paid, S20) |
| `LLM` | `complete(messages) -> Completion` | `OpenAICompatLLM` (Ollama `/v1`, OpenAI, llm-gateway) | `AnthropicLLM` (extra `[anthropic]`), `FakeLLM` |

`Store` keeps vectors and the inverted index in one interface, so the PG backend can finish all writes for one document in one transaction. The BM25 formula is in `keyword.py`; the store only offers `bm25_query` and decides inside where to score: PgStore can score in SQL (`store.bm25_impl: sql`, the default), or fetch the postings and hand them to `keyword.bm25` (`python`). Both ways share one statistics SQL. The Chroma backend uses only the Python way.

The library has two entry layers (since S8): `RagRuntime` holds everything that does not depend on a namespace (config, store, embedder, token counter, chunker, and the reranker / LLM, which are built on first use) and is built once; `runtime.engine(namespace)` returns a `RagEngine` bound to one namespace. It is very light, so building one per request is fine. The HTTP service holds one runtime. A library caller that uses only one namespace can call `RagEngine.from_settings(settings, namespace)` directly; it brings its own runtime, which is closed together with it on `close()`.

```python
runtime = RagRuntime.from_settings(settings)   # server: max_connections = number of worker threads
engine = runtime.engine("tenderted/tenant_42")  # or RagEngine.from_settings(settings, namespace)
engine.ingest(kb, path)                                   # file → document id
engine.upsert_text(kb, doc_id, title, text, metadata={})  # replaces the whole document if it exists
engine.delete(kb, doc_id)
engine.search(kb, query, top_k=5, mode="hybrid", rerank=False) -> list[Hit]
engine.ask(kb, question, top_k=5) -> Answer
engine.stats(kb)
```

### 3.1 Chunking strategies (configured per kb)

The chunking method can be configured per kb and is chosen by name from a registry; to add a strategy later, implement the `Chunker` Protocol and register it. Every chunk carries the character offsets `start` and `end` (used to locate citations), `heading_path` (the heading levels it sits under), `tokens` (the real token count), `page` (the page it starts on, from 1; empty when the document has no pages, S5b) and `parent` (the index of the parent chunk, for `parent_child`). Which strategy was used is recorded at kb level (the kb identity), not per chunk: one kb cannot contain chunks cut by two strategies.

**Parse structure first, then chunk** (2026-11-08): most of the quality of chunking depends on whether structure was kept at parsing. Once a PDF is flattened to plain text, no chunker can recover the heading levels. So the loaders come in two tiers: pypdf is fast and has no model, and gives only plain text and page numbers; docling uses a layout model to convert the PDF to Markdown (a page marker is inserted between pages, so page numbers are kept too) and hands it to the `markdown` strategy to get `heading_path`. The two limits of docling 2.148 that were measured are in the §13 S5b note.

| Strategy | What it does | Fits | Stage |
| --- | --- | --- | --- |
| `fixed` | Cuts at a fixed length, with overlap | Baseline, for comparison | S2 |
| `recursive` (default) | Cuts recursively by a hierarchy of separators: heading → blank line → newline → sentence-end punctuation → space, until smaller than `size`, then adds overlap | Most text | S2 |
| `sentence` | Splits sentences at Chinese and English sentence-end punctuation, then merges by window (N sentences per chunk, M sentences overlap) | Question answering, short facts | S2 |
| `paragraph` | Splits paragraphs at blank lines, merges too-short paragraphs forward, and cuts too-long ones with recursive | Well-structured articles | S2 |
| `markdown` | Cuts by heading level, with recursive inside a chunk; `heading_path` is written as a prefix into the embedding text | Documents, manuals, wikis | S2 |
| `semantic` | Splits by sentence (headings still cut), embeds all sentences of the whole document at once (through the kb's embedder and cache); breaks where the cosine distance between two adjacent sentences is above the `semantic_threshold` percentile (default 95, linear interpolation, so a document with one topic is not cut) of the adjacent distances in the whole document, then packs each group by `size` | Long text with frequent topic changes | S5b |
| `semantic` (`semantic_impl: chonkie`) | Uses the SemanticChunker of chonkie 1.7.0 (fixed cosine-similarity threshold `chonkie_threshold`, 3-sentence window, Savitzky-Golay valley finding), with lite-rag's embedder and token counter; outside it, exact offsets, the hard `size` limit and no crossing of headings are still guaranteed | Comparison with the self-written version (extra `[chonkie]`) | S5b |
| `parent_child` | Parent chunk: `markdown` strategy, `parent_size` (default 4×`size`), no overlap; child chunk: cut with `recursive` inside each parent, so a child never crosses a parent. Only child chunks are embedded and get BM25; parent chunks are stored in the `parents` table. Retrieval ranks and reranks on child chunks, then swaps them for parents and dedups by parent: the top-k are k distinct parents, `Hit.text` is the parent, and `Hit.matched` is the range of the matching child | Needs exact hits and also needs context | S5b |
| `contextual` (enhancer, `chunking.contextual.enabled`) | Can be stacked on any strategy: the LLM writes one or two sentences for each chunk saying "what this passage is about in the whole document" (when the document is longer than `max_doc_tokens`, it is given only the stretch centred on the chunk), which is put in front of the chunk before embedding and BM25, stored in `chunks.context`, and does not appear in `Hit.text`. The kb identity records its provider/model and the prompt hash; if a call fails, the whole upsert fails, with no retry on another model | Documents with many references that cannot be understood from one passage alone; one more LLM call per chunk | S5b |

- **Length unit**: `chars` or `tokens`, default `tokens`, counted with the embedding model's own tokenizer. The lesson from the KB is that cutting by word count distorts (`rag-production-guide` §3). When no tokenizer is available, characters are used as an estimate, and the fact that this was an estimate is recorded.
- **Config example**: `chunking: {strategy: recursive, unit: tokens, size: 800, overlap: 120, min_size: 50}`. 800/120 is a starting point measured in the KB; the final choice is decided by eval.
- **`size` is a hard limit** (the embedding model has an input length limit): no step produces a chunk larger than `size`. When the last chunk is smaller than `min_size`, the last two chunks are re-split evenly so that neither exceeds `size` and the tail is not smaller than `min_size`; if no such split is found, they are left as they were. When counting by estimate, every 6 consecutive letters or digits count as one token, so an overlong string without spaces is also cut. (Codex review, 2026-11-08)
- **Changing the chunking means rebuilding the kb**: the `kbs` table records the chunker config, and a changed config requires a rebuild (`literag reindex` is postponed to S6: the original document text is not stored, so a rebuild needs the source files). Thresholds must be recalibrated too, see §5. A new chunking parameter enters the identity only in the strategy that uses it (`parent_size` only in `parent_child`, `semantic_*` only in `semantic`, `contextual` only when enabled), so the identity of an existing kb does not change.
- Out of v1 scope: late chunking and proposition chunking. The former needs a dedicated long-context embedding model and the latter needs one LLM call per chunk; they will be considered when eval shows S5b is not enough.

### 3.2 Embedding (each kb is fixed to one model)

`Embedder` offers `embed_documents` and `embed_query` separately, because many models require documents and queries to be told apart: for example the `input_type` parameter of voyage and cohere, and the instruction prefix that models such as bge, e5 and qwen3 add to a query. These prefixes are written in each model's config, not hard-coded.

| provider | How it is called | Candidate models (locked after checking against the official docs and the MTEB / C-MTEB leaderboards at S0/S5b) | Cost |
| --- | --- | --- | --- |
| `ollama` (default) | Local Ollama `/api/embed` | `bge-m3` (multilingual, 1024 dims), `qwen3-embedding`, `nomic-embed-text`, `mxbai-embed-large`, `snowflake-arctic-embed2` | Free |
| `fastembed` | In-process ONNX, no Ollama needed | `BAAI/bge-small-zh-v1.5`, `intfloat/multilingual-e5-*`, etc. | Free |
| `sentence_transformers` | In-process PyTorch (extra `[sentence-transformers]`, 6.2.0; CPU / CUDA / Apple MPS; torch is loaded on first use) | Any HF embedding model, for example `Qwen/Qwen3-Embedding-0.6B` | Free |
| `openai` | OpenAI-compatible `/v1/embeddings` | OpenAI `text-embedding-3-large` / `-small` (support truncation by `dimensions`); the same implementation also reaches DashScope (Tongyi `text-embedding-v*`), SiliconFlow, Jina, vLLM, TEI | Paid (free quotas differ by vendor) |
| `voyage` | lite-model's `voyage` type (since S20; before that, direct HTTP), `input_type` = query / document (Voyage prepends a retrieval prompt itself), `output_dimension`; at most 1000 items per call; key `VOYAGE_API_KEY` | `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-3.5`, etc. The KB records tenderted measuring voyage-3-large | Paid |
| `cohere` | lite-model's `cohere` type (since S23; before that, direct HTTP) `POST /v2/embed`, `input_type` = search_document / search_query, `embedding_types: [float]`; at most 96 items per call; key `CO_API_KEY` | `embed-v4.0`, `embed-multilingual-v3.0`, etc. | Paid |
| `gemini` | lite-model's `gemini` type (since S23; before that, direct HTTP) `POST /v1beta/models/{model}:batchEmbedContents`, one Content per text (several parts in one Content are combined into one vector); `gemini-embedding-001` uses `taskType` = RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY, `gemini-embedding-2` has no such parameter and writes the task as a prefix (see the table below); `outputDimensionality`; at most 100 items per call; key `GEMINI_API_KEY` | `gemini-embedding-2`, `gemini-embedding-001` (their vector spaces are not compatible) | Paid, with a free quota |

Each provider has its own default `base_url` (`EMBEDDING_URLS`), which `embedding.base_url` can override (for example to go through a proxy); when `api_key_env` is not set, the variable name that each vendor's SDK conventionally uses is used. All HTTP embedders go through lite-model (S19 ollama / openai, S20 voyage, S23 cohere / gemini): batching, concurrency, halving on 400 / 413 and same-model retries are its job; lite-rag only turns its own `EmbeddingConfig` into lite-model's config, and keeps the prefixes, the vector cache and the index identity. (History: at S5c they were lite-rag's own direct HTTP calls sharing `_HTTPEmbedder`; removed in S23.) The in-process fastembed / sentence-transformers stay in lite-rag (lite-model carries no ONNX / torch, see its design §3.4).

- **"Best" is measured by ourselves**: a leaderboard can only narrow the candidates. The choice of model rests on the `literag eval` results on this project's corpus (AGENTS rule 8). So the acceptance of S5c is a multi-model comparison table that lists hit@k, MRR, latency and cost (0 for local models; paid APIs run only after the user provides a key and agrees).
- **Interface**: every `Embedder` offers `model_id` (`provider/model`, for example `ollama/bge-m3:567m`), `dim`, `prefixes`, `embed_documents` and `embed_query`. Which implementation is used is decided by config; callers depend only on this interface.
- **The difference between documents and queries is encapsulated in the implementation** (the easiest trap): a wrong prefix raises no error and only lowers retrieval quality silently. So `model_profiles.py` keeps a table of the prefixes of known models, and the embedder adds the prefix by model name itself; callers need not know these details. The table below comes from each model's model card on Hugging Face, checked on 2026-11-08:

  | Model | Query | Document |
  | --- | --- | --- |
  | e5 family | `query: ` | `passage: ` (required) |
  | nomic-embed-text | `search_query: ` | `search_document: ` (required) |
  | bge-*-v1.5 | Chinese `为这个句子生成表示以用于检索相关文章：`, English `Represent this sentence for searching relevant passages:` | None |
  | bge-m3 | None | None |
  | qwen3-embedding | `Instruct: <task>\nQuery:` | None |
  | gemini-embedding-2 | `task: search result \| query: ` | `title: none \| text: ` (the official format is `title: {title} \| text: …`; lite-rag's prefix is a fixed string and a chunk carries its own heading path, so `none` is used) |

  `query_prefix` / `document_prefix` in the config can override this table; setting `""` means no prefix. fastembed and sentence-transformers both use plain encode plus the prefix from the table (not their own query/passage methods), so the prefix recorded by the kb is the prefix actually added. Voyage, Cohere and gemini-embedding-001 tell documents from queries with a request parameter; this is encapsulated in their own classes, and the prefix is empty.
- **The index is bound to the model** (the most important point; `rag-production-guide` §4 rule 2, and also the check RAGFlow makes when the embedding is changed): even when the dimensions are the same, vectors of different models must not be mixed, for example bge-m3 and voyage-3-large are both 1024 dims. The `kbs` table records `provider`, `model`, `dim`, the query prefix, the document prefix and the chunking config, and every time a kb is opened (to write or to query) they are checked; a mismatch raises an error, and a wrong result is never returned. To change the model, create a new kb and embed again, for example named `docs-voyage3`; the old index is not changed in place. When `literag reindex` is added later, it will follow the same flow: "build a new kb → write → verify → switch".
- **Vector cache**: one `embedding_cache_<dim>` table per dimension, with primary key `(model_id, sha256(role ‖ prefix ‖ text))`. The same model reuses the result directly for exactly the same input, shared across kbs and namespaces, because a vector depends only on the model and the input, not on who stores it. The key includes the role (document / query), so a query never gets a document's vector. This way repeated text is not computed twice, and it also saves money after switching to a paid model. It can be turned off with `embedding.cache: false`.
- **A single exit (optional)**: if llm-gateway later offers an OpenAI-compatible `/v1/embeddings`, point the `base_url` of the `openai` provider at it. All model calls then leave from one place, and usage statistics are centralised too. This is a matter for the gateway repo; lite-rag needs no change.
- **Batching and rate limiting**: `batch_size`, the concurrency limit (default 4) and the backoff retry on 429/5xx (default 4 times) are all configurable. What is retried is the same model, which is not a fallback chain; after the retries are used up it raises directly, and never returns an empty result.
- **Dimension truncation**: when a model supports truncation (OpenAI `dimensions`, Gemini, Matryoshka models), `dimensions` can be configured. The PG backend uses a separate table per dimension (§4).
- The key of a paid provider is read only from environment variables. Tests always use fakes, or replay recorded responses, and never call a real API.
- **What to do first in each layer**: first build the abstractions of the Embedder and Store layers, because they decide how data is stored and are hard to change later. These two layers were finished in S1 and S2. The Reranker is added when it is really needed (S4); the LLM goes through llm-gateway or Ollama (S4). The default storage is still pgvector (decided 2026-11-08), with Chroma as an optional backend (S7).

## 4. Storage and ids

### PostgreSQL backend (default)

The database (for example `lite_rag`) holds one schema (default `lite_rag`, configurable). These tables are created by `literag migrate`, and the migration scripts live inside the package:

| Table | Content |
| --- | --- |
| `kbs` | `name` (primary key), `embedding_provider`, `embedding_model`, `dim`, `query_prefix`, `chunker` (JSON), `n_chunks` and `total_length` (N and avgdl of BM25), `created_at` |
| `documents` | `kb`, `doc_id`, `title`, `source`, `content_hash`, `chunk_count`, `metadata` JSONB, `updated_at`; primary key `(kb, doc_id)` |
| `chunks` | `id` (`"{doc_id}#{n}"`), `kb`, `doc_id`, `n`, `text`, `length` (word count), `metadata` JSONB |
| `embeddings_<dim>` | `chunk_id`, `kb`, `embedding vector(<dim>)`, HNSW index (cosine). One table per dimension, created on first use |
| `postings` | `kb`, `term`, `chunk_id`, `tf`; index `(kb, term)` |
| `graph_mentions` (S11) | `kb`, `chunk_id`, `doc_id`, `entity` (the normalised name), `name`, `type`: which chunk mentions which entity; deleted by cascade with the chunk |
| `graph_relations` (S11) | `kb`, `chunk_id`, `doc_id`, `source`, `target`, `description`: the relations extracted from a chunk; deleted by cascade with the chunk |

- **One transaction**: `upsert_document` deletes the old chunks, embeddings and postings in the same transaction, writes the new ones, and then updates the statistics in `documents` and `kbs`. There is no intermediate state and no recovery logic is needed.
- **BM25 query**: first fetch the postings by `(kb, term)` in SQL, together with `n(t)`, `N` and `avgdl`, then score by the formula in section 5. Scoring can be done in Python (`keyword.py`, easy to read) or entirely in SQL (fast). The results of the two algorithms are checked against each other with the same set of tests.
- **Vector query**: `ORDER BY embedding <=> :q LIMIT n`, filtered by `kb`.

### Chroma backend (optional, S7)

- Config `store: {backend: chroma, data_dir: ./lite_rag_data}` (environment variables `LITERAG_STORE__BACKEND`, `LITERAG_STORE__DATA_DIR`), extra `[chroma]` (`chromadb==1.5.9`, Apache-2.0, a heavy dependency, so it is not in the core). Run `literag migrate` first to create the data directory. The data directory has two parts:
  - `chroma/`: a `PersistentClient` with telemetry off (`anonymized_telemetry=False`); one collection per kb (cosine), named by the hash of the kb key (Chroma allows only `[a-zA-Z0-9._-]`);
  - `lite_rag.sqlite`: the same `kbs`, `documents`, `chunks`, `parents`, `postings` and vector cache as the PG backend, with migration files in `migrations_sqlite/`; JSON is stored in TEXT columns. N and avgdl are computed from `chunks` on every query; no counters are kept on `kbs`.
- **BM25** is implemented in Python only (`keyword.bm25`); the default is `bm25_impl: python`, and setting `sql` raises an error when the config is loaded. As in PG, the statistics are taken over the whole kb, before the metadata filter.
- **Metadata filter**: scalar values (str, int, float, bool) in the document metadata are copied onto each of its Chroma records, and filtering compares for equality (Chroma `$eq` / `$and`); bool is not equal to a number, in both retrieval paths. A filter value must be a scalar. This rule is the same for both backends and is checked by the engine before retrieval starts (see section 5).
- **Writes are not in one transaction**: vectors are in Chroma, the rest in SQLite. A document carries a state:
  - upsert: in one SQLite transaction, delete the old version and write the new rows, state `pending` → delete the old vectors in Chroma and write the new vectors in batches (each batch no larger than `get_max_batch_size()`) → mark `ready`;
  - delete: mark `deleting` → delete the Chroma vectors → delete the SQLite rows;
  - reads see only `ready` documents;
  - when the store is opened, documents in `pending` and `deleting` are deleted from both sides, then collections without a matching kb are deleted; the caller just ingests again (the content hash keeps this idempotent). When a kb is deleted, the rows are deleted first and then the collection; before a kb is created, a leftover collection with the same name is deleted first, so that old vectors do not get mixed into the new kb.
  - Difference from PG: when a write fails (not a crash) it is cleaned up at once, so a failed upsert deletes the document instead of keeping the old version; PG rolls the whole document back and keeps the old version.
- **Concurrency**: single process only; one lock serialises all operations of the store, and SQLite runs in WAL mode. Several processes sharing one data directory is not supported.
- Suited to single-machine learning where you do not want to install PG. The contract tests (`tests/contract/`) run the same cases against both backends.

### Namespace (mandatory)

- **A namespace must be given**: every project, and every tenant inside a project, has its own namespace, for example `tenderted/tenant_42`. It must be passed when the engine is created: `RagEngine.from_settings(settings, namespace="tenderted/tenant_42")`; the CLI takes it with `-n` or `LITERAG_NAMESPACE`; in the future HTTP API the API key binds the namespace. If it is not given, an error is raised at once.
- **Physical key**: the key of a kb in the database is `<namespace>:<kb>`, and the `kbs.namespace` column is used to list the kbs under a namespace. Every read and write of the engine builds this key itself, so a caller cannot get around it, and a kb name cannot contain `:` or `/`. Isolation is enforced at the module level and does not depend on callers behaving, so the TenderTED problem of several projects sharing one index cannot happen again.
- **Format**: a namespace consists of lowercase letters, digits, `_`, `-` and `.`, with at most 4 levels, separated by `/`.

### General rules

- **Id rule**: a document id is given by the caller (`upsert_text`), or generated from the hash of the file path (`ingest`). A chunk id is `"{doc_id}#{n}"`.
- **Idempotent**: when neither `content_hash` nor the chunker config has changed, `upsert_text` returns at once; otherwise the whole document is replaced.
- **Changing the embedding model**: the `kbs` table records the model and dimension of every kb. When the config and the record differ, opening is refused, with a hint to run `literag reindex <kb>`.

### Lineage and query trace (S25)

The detailed design is in `docs/superpowers/specs/2026-11-24-lineage-design.md`. Only the tables and rules are listed here:

| Table | Content |
| --- | --- |
| `lineage_artifacts` / `lineage_edges` / `lineage_schema_migrations` | A generic ledger (`lite_rag.lineage`, which knows nothing about RAG): artifact versions and derivation edges; its migrations are run together by `literag migrate` / `lite_rag.db.migrate` |
| `chunks.lineage_key` (0007) | The chunk's ledger key `{doc_id}/{first 16 chars of the source hash}/{k}`, computed at write time and read at retrieval time |
| `rag_query_traces` / `rag_query_sources` (0008) | One retrieval or question, and the chunks used at each stage (by the `lineage_key` read at retrieval time); `in_context` is not stored, and at read time it is taken as the first `top_k` of the last ranking stage |

- Every upsert / delete / delete_kb is recorded in the ledger in the same transaction (`lineage.enabled`, on by default). A chunk gets its identity from its source text: a chunk whose text did not change reuses the old artifact and gets one more edge, with its position and context recorded on the edge; only a chunk that the new version of the document no longer contains is ended.
- A deleted artifact has its text and metadata (title / source) cleared at once; a superseded one keeps its text for `lineage.keep_text_days` (cleaned by `literag lineage gc`), and the row is kept for ever. The contextual context does not enter the ledger.
- `literag lineage backfill` also records documents deleted while the ledger was off as deleted; a whole kb that was dropped while the ledger was off cannot be recovered.
- The time of a trace (`trace.enabled`, off by default) is taken after the chunks have been read; `trace show` uses `started_by` to find the version that was read at that time: if it has ended, it is "stale", and the source text is shown.
- A caller of the ledger must serialise writes of the same namespace; lite-rag's namespace is the kb physical key, and a PG write holds the kb row lock.

## 5. Retrieval

- **Vector**: first embed the query, then take the top-n (n = `top_k × candidate_factor`, 4 times by default).
- **BM25**: Chinese is tokenised with jieba and English is lowercased; the stop word list is configurable. The standard formula is used:

  `score(q, d) = Σ IDF(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 − b + b·|d|/avgdl))`,
  `IDF(t) = ln(1 + (N − n(t) + 0.5)/(n(t) + 0.5))`, with default `k1=1.5` and `b=0.75`.
- **Fusion**: RRF by default, `score = Σ 1/(k + rank)`, `k=60`. Weighted fusion can also be chosen: the scores of the two paths are first min-max normalised separately, then added by weight.
- **Rerank** (optional): take the first n after fusion, rescore each `(query, chunk)` pair with the cross-encoder, then take the first k.
- **Modes**: `vector`, `bm25`, `hybrid` (and `graph` since S11); and `route` since S21: the `route` role (local qwen3:1.7b by default, no thinking) picks one mode from `retrieval.router.modes` for each question, and every `Hit.route` records what was picked, which model picked it, and the reason; if the call fails it raises an error, and there is no default mode to fall back to. Every `Hit` carries the scores of each stage `scores={vector, bm25, fused, rerank}` and the ranks, for comparison and debugging.
- **Querying several kbs together** (S10, 2026-11-09, for llm-steward's joint retrieval): `RagEngine.search_kbs(kbs, query, …)` queries several kbs of the same namespace as one. The query is embedded only once; each path (vector, BM25) takes the first n from each kb, merges them into one list by score, and then fuses only once and reranks only once. `Hit.kb` of a result says where it came from (a chunk id is unique only inside a kb, so internally "kb index/chunk id" is used as the key). Vector scores are comparable across kbs (one engine has only one model); the IDF of BM25 is counted by each kb itself, so the merged BM25 list is approximate: in a small kb the IDF of the same word is higher and the score is too large, and on the default path (rerank off) it takes the front places of the merged BM25 list. When rerank is on, the final order is fully comparable. The number of queries grows with the number of kbs (in every round, one query per path per kb; when deepening, every kb is queried again). Left for later: when the bias becomes a problem, have the Store return the (N, df(t), avgdl) of each kb and let the engine compute one common IDF; when there are many kbs, run the kb queries in parallel, or query the vectors of several kbs with one SQL on PG. A kb that has no document written yet (does not exist) is skipped; an invalid name is still `InvalidRequest`. `search` on a single kb goes through the same flow, and its result does not change.
- **Graph retrieval** (S11, 2026-11-12, local mode only, PG only):
  - **Building**: when a kb has `graph.enabled`, each chunk gets one LLM call (role `graph`, local qwen3 by default, thinking off), which extracts entities (name, type) and relations (the two end entities, a one-sentence description) following `prompts/graph_*.j2`; the reply is held to a JSON schema (since S16, `GraphReply`: both lists `entities` and `relations` must be present, and are empty lists when nothing was found). The LLM sees only the chunk body, not the document title: when given the title, a local model takes the title as an entity of every chunk, and it becomes a hub connecting everything (measured). When the reply still does not fit the schema, it is asked once more with the same prompt, and if the second one still does not fit, the whole document write fails, with no other model tried (rule 5). When a relation endpoint is not in the entity list, it is added as an entity. Entities are merged by normalised name (NFKC, casefold, merged whitespace), so the same person in different documents is one node. The graph is written in the same transaction as the chunks; the extraction model and prompt are recorded in the kb identity (`chunker.graph`), so changing them needs a new kb. The Chroma backend does not support it, and it errors at config time.
  - **Query** (the `graph` path, no LLM call): **the graph sets the range, vectors rank.**
    1. **Seeds**: entities whose name (normalised) is a substring of the query, at least 2 characters; an entity whose name is contained in 2 or more other entity names does not count (category words such as "委员会" (committee), "council").
    2. **Expansion**: follow "mentioned together in the same chunk": entity → chunks that mention it → other entities in those chunks, for `graph.hops` hops (default 2, that is, the answer is two documents away from the entity the question names). An entity mentioned by more than `graph.hub_chunks` chunks (default 5) is a hub (a company name, a topic that runs through the whole text): it is not expanded through, and it does not bring chunks into the range. In each hop only the `graph.max_neighbors` new entities with the most connecting chunks are kept.
    3. **Ranking**: the chunks reached are ranked by cosine similarity to the query vector (`vector_query(…, among=reached chunks)`, an exact scan by primary key, not through HNSW). The scores are therefore comparable with those of the vector path.
    `where` limits the whole corpus: seeds are looked for only in the documents that are kept, and the graph walk goes only through them; a document that is filtered out neither appears in the result nor serves as a bridge (pointed out by Codex at the close-out of S11, 2026-11-12). In one retrieval the graph of each kb is walked only once; deepening rounds only re-rank. The seed rule and the hub rule are both in `graph.py`, and the store returns only data.
    It walks co-occurrence and not the extracted relations, because the local model missed many relations it could have written (the Chinese people sections have almost no relations); the relations are stored all the same, kept for a later global mode and for the answer context.
  - **Why it is scored this way** (2026-11-12 multi-hop eval, 12 questions of 2–3 hops, compared offline): the first version scored a chunk by "the sum of entity weights"; the chunk holding the seed was always highest and the answer was pushed back, multi-hop hit@5 was only 0.42, and the single-hop control questions also dropped from 1.00 to 0.25; finding seeds by vector brought a lot of noise. "The graph sets the range, vectors rank" gives multi-hop hit@5 0.92 and leaves single-hop unaffected. The hit@1 of multi-hop questions is 0 under every method: the one ranked first is usually the first link of the reasoning chain (the chunk holding the seed), and the answer follows right behind; when answering, both are given to the LLM, so for this kind of question hit@5 is what counts.
  - **Mode**: a new `mode: graph` (graph only). Which paths `hybrid` contains is decided by `retrieval.hybrid_paths`, default `[vector, bm25]` (behaviour unchanged); adding `graph` makes it multi-path RRF. Weighted fusion supports only vector + bm25, and configuring graph raises an error. Using `mode: graph` on a kb without a graph raises `InvalidRequest`.
  - **Cost**: building takes one LLM call per chunk; a query takes 2–3 more SQL statements plus one range-limited vector query. Seed matching scans the entity names of the whole kb; an index is added when the size makes it a bottleneck.
  - **Limits**: the question must write the name of the entity ("Atlas 项目的负责人" (the head of the Atlas project) can find "atlas 项目", but "Atlas" alone cannot); `hub_chunks` is an absolute number and must be tuned when the corpus grows.
  - **Not done** (to be decided in S11b): global mode (community partition + community summaries, to answer summary questions over a whole batch of documents), using the LLM at query time to extract keywords, or entity vectors, to find seeds, and merged summaries of entity descriptions.
- **Filter**: isolation by namespace and kb comes first; `where` is an equality filter of "key → scalar value" (str, int, float, bool), which takes effect in both the vector and the BM25 path; bool is not equal to a number. A value that is not a scalar (list, object, null) raises `InvalidRequest` (400), checked before the query is embedded, with the same rule in both backends (decided in S8, 2026-11-09).

## 6. Generation

- **Prompt templates** `prompts/answer_system.j2` + `prompts/answer_user.j2`:
  - the retrieved chunks are numbered `[1] … [k]` in order;
  - the model is asked to answer only from these materials and to mark the numbers it cites;
  - when the materials are not enough, it says so directly.
- **`Answer` contains**:
  - the answer text;
  - `citations`: which chunk ids were cited; the code checks that every number is in the retrieval result;
  - `model_id`: `backend/model`, **the model that actually answered** (read from the response since S15, by lite-model according to the backend: with direct Ollama it is the
    model name, through LiteLLM it is `x-litellm-model-name`); `None` when no model was called. The transitional field `response_model` of S14 is removed;
  - token usage.
- **No fallback chain**: when a call fails it raises `LLMError` (HTTP 502), and retries only on the same model; the error message does not carry the upstream text.
  A truncated answer (it reached `max_output_tokens`) is also an `LLMError`; half an answer is no longer returned as a complete one (S15, decided by the owner).
- **LLM calls are made by lite-model** (S15, lite-model design §12): the three roles — `answer`, `contextual`, `graph` — are three roles of the same
  lite-model config `llm`. `contextual` and `graph` are part of the index identity and are **not routable** (a LiteLLM routing alias is
  rejected at start-up); `answer` is.

```yaml
llm:                      # lite-model's ModelsConfig (its design §4), with lite-rag's defaults
  backend: ollama         # ollama | openai | anthropic | gateway | litellm, or a backend you name yourself
  model: qwen3:8b         # the top-level backend, model and call parameters tune only answer (as `llm` did before S15)
  max_output_tokens: 1024
  connect_timeout_s: 5
  roles:                  # contextual and graph default to ollama/qwen3:1.7b, no thinking (S17; graph 2048 tokens), each field can be changed
    graph: {model: qwen3:14b}
  backends:               # same name, field by field: an openai-compatible vendor uses the name openai, and the record is still openai/<model>
    openai: {base_url: "https://api.deepseek.com/v1", api_key_env: DEEPSEEK_API_KEY}
```

  The top-level `temperature`, `reasoning_effort` and `max_output_tokens` move to `roles.answer` (what answer writes itself wins): contextual and
  graph take values only from `llm.roles`; otherwise the context and graph of new chunks under the same index identity would be generated in a different way (Codex S15).
  The role id is `backend/model` (the configured model) and is written into the index identity (`chunking.contextual`, `graph`); it is byte-for-byte the same as the
  `provider/model` before S15, so existing kbs need no rebuild. The pre-S15 shapes (`llm.provider`, `llm.base_url`,
  `llm.api_key_env`, `llm.max_tokens_param`, `chunking.contextual.llm`, `graph.llm`) are rejected at start-up, with a message saying where they moved to.
  Keys are read only from environment variables.

## 7. HTTP service (`literag serve`)

FastAPI, default address `127.0.0.1:8700`. As with the library, every call goes through `RagEngine`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | No authentication needed; checks the storage and the embedding service |
| `PUT /v1/kbs/{kb}` · `GET /v1/kbs` · `GET/DELETE /v1/kbs/{kb}` | Create a kb (with its chunker and embedding model), list, view statistics, delete |
| `PUT /v1/kbs/{kb}/documents/{doc_id}` | Upsert one text document (`title`, `text`, `metadata`) |
| `POST /v1/kbs/{kb}/files` | Upload a file (multipart), returns `doc_id` |
| `DELETE /v1/kbs/{kb}/documents/{doc_id}` | Delete a document |
| `POST /v1/kbs/{kb}/search` | `{query, top_k, mode, rerank, where}` → hits (with the scores of each stage) |
| `POST /v1/search` | `{kbs, query, top_k, mode, rerank, where}` → query several kbs together (S10); the key must cover every kb, otherwise 403 |
| `POST /v1/kbs/{kb}/ask` | `{question, top_k}` → answer (with citations and sources; `model_id` is the model that actually answered, S15, see §6; the `response_model` of S14 is removed) |

- **Authentication** (S6, KB web-authz: deny by default): `Authorization: Bearer lrk_<prefix>_<secret>`. The server stores only the `sha256` of a key, configured in `server.keys` (YAML, or `LITERAG_SERVER__KEYS` set to a JSON list). Each key has two groups of globs, `namespaces` and `kbs` (`*` can cross `/`, for example `steward/*`), and a `mode` (`read` / `write`); there is no default namespace, and a key that does not write one has no access. `literag key create` generates a new key, prints it only once, and also gives the entry to add to the config; **no HTTP route can create a key**. When no key is configured the service is fully open, so `serve` is allowed to bind only to a loopback address.
- **namespace**: each request names it with the `X-LiteRag-Namespace` header (a namespace itself contains `/`, which is not clean in a path). A missing header returns 400; a key that does not cover this namespace or this kb returns 403, and a read-only key that writes also gets 403; no key, or an unknown key, returns 401 with `WWW-Authenticate: Bearer`. The server holds one `RagRuntime`, and each request takes an engine with `runtime.engine(namespace)`, sharing the store, embedder, chunker and models.
- **The only path left without authentication on purpose** is `GET /health` (with `?deep=true` it also embeds one cached probe query).
- **Error format**: RFC 9457 Problem Details (`application/problem+json`): `KbNotFound` / `DocumentNotFound` 404, kb config mismatch (`KbMismatch`) 409, caller errors (`InvalidRequest`: name, parameter, file type, filter value) 400, request validation failure 422, embedding / LLM failure 502, storage failure 503, the service's own config error (`ConfigError`: extra not installed, API key not set, model does not exist) 500. The details of upstream failures and service config errors are written only to the log and are not returned to the client (they may contain addresses, the vendor's original error text or environment variable names).
- **Request body limit**: every request is checked by size before parsing (JSON `server.max_body_mb`, default 10; uploads `server.max_upload_mb`), and over the limit it returns 413. Open mode (no key configured) accepts only a local Host (against DNS rebinding), and rejects with 403 a request that carries `Origin` (sent by a browser page) or proxy headers (`Forwarded`, `X-Forwarded-For`, `X-Real-IP`) (2026-11-20, same as lite-mem). `GET /v1/kbs` lists only the kbs within the `kbs` scope of the key.
- **Concurrency**: the handlers are all synchronous, FastAPI runs them in a thread pool, and the thread count is limited by `server.max_workers` (default 8); when rerank is enabled, the model is loaded at the start-up stage.
- **Upload**: `POST /v1/kbs/{kb}/files` (multipart, `file` plus an optional `doc_id`; default `file:<sha256(file name)>`, so uploading a file with the same name again is a replacement), size limit `server.max_upload_mb` (default 20).
- **Deployment**: the repo ships a `Dockerfile` (python:3.12-slim, non-root, `HEALTHCHECK`; runs `literag migrate` first at start-up), and environment variables point at PG and Ollama. CI builds the image every time and runs a smoke test (/health, 401, 403, 200), and **does not push** to any registry (decided 2026-11-08). The `deploy/compose.yaml` example contains lite-rag + PG (pgvector); Ollama runs on the host (`host.docker.internal`).
- **Installation**: `pip install "lite-rag[server] @ git+https://github.com/ArthurGao/lite-rag"`. It will be published to PyPI later.

## 8. Eval

- **Corpus**: a small corpus under `eval/`, one set each in Chinese and English, of public knowledge written by ourselves. The golden questions are kept in JSONL, one `{question, expected_doc_ids[], expected_chunk_ids?[]}` per line.
- **Comparison**: `literag eval --config a.yaml --config b.yaml` builds a temporary kb for each config and then runs all the questions. The results are listed side by side: hit@1, hit@5, MRR@10, and p50 and p95 of retrieval latency.
- **Runs on lite-eval** (since S12): `literag eval` converts the golden set and the corpus into a lite-eval dataset, and each config is one lite-eval run (`eval/lite-eval/runs/`, gitignored). Judging (evidence sentences) and the metrics are computed by lite-eval's retrieval pack, with definitions word for word the same as before S12 (lite-eval design §18.16). Two runs can be compared pairwise for significance with `lite-eval compare`; `--crosscheck` recomputes the ranking metrics with ranx. lite-rag keeps only the adapter `LiteRagTarget` (`src/lite_rag/eval.py`).
- **Whether the answer is grounded** (since S13, the model judge of lite-eval E5): the retrieval eval looks only at whether the evidence was found; the quality of an answer is judged by lite-eval's grounded judge: does the answer contain a claim that the sources it cites do not support. The judge must be calibrated before it can be used (lite-eval design §9, §18.24):
  - **Calibration set** (`eval/judge-grounded/datasets/grounded-golden/`, committed, generated by `eval/make_judge_grounded.py`, and a test checks it agrees with the generator): the qualified answer of each golden question = the sentence of the document that contains the evidence sentence, citing the paragraph it sits in as source [1], which is grounded by construction; then lite-eval's construction tool corrupts it in one way each: change a number, add a sentence from another document, replace it with the answer of another question (all unsupported), or remove the citation (still grounded, the judge does not judge citation marks). The inserted and swapped content is taken only from other documents of the same language (by the golden set, not guessed from characters).
  - **Calibration**: `lite-eval run eval/judge-grounded/suites/calibrate.yaml --config <judge>`; the judge itself is the Target, and the agreement rate and the detection rate of each kind of corruption are recorded; the runs are in `eval/judge-grounded/runs/` (gitignored).
  - **Real answers**: `literag eval-answers -c <config> --calibration <calibration run> --min-agreement <threshold>` runs `ask` on the golden questions (`LiteRagAnswerTarget`, output `{question, answer, sources, status, dropped}`) and scores with the judge that the calibration run measured; the judge's config is taken from the one entry in the calibration suite, lite-eval checks it item by item before any call, and refuses when the calibration is not enough. Finally every flagged answer is listed and read one by one (was the judge wrong, or is the answer really bad).
- **Whether the answer is correct** (since S14): each golden question has a `key_fact` (a case-insensitive regular expression, the one thing the answer must say), and `eval-answers`
  adds lite-eval's `regex` scorer next to the grounded judge. The labels are by construction: `key_fact` matches the qualified answer of that question (the evidence sentence in the calibration set),
  does not match the question itself (`load_golden` rejects it), and does not match the answers in the calibration set whose number was changed or that were swapped with another question's answer (tested). It cannot see direction ("磨粗" (grind coarser)
  and "磨细" (grind finer) both mention "细"), so **the main metric is that `key_fact` and grounded both pass**, not `key_fact` alone. `key_fact` is all or nothing:
  if only some questions have it, `eval-answers` refuses (otherwise this item would be switched off silently); if none has it, the report says "not checked".
- **No output cache** (since S14): the suite of `eval-answers` writes `cache: off`, so every run answers again (at S13 a second run directly
  reused the first run's answers and latency, and with that repetition nothing is measured). `literag eval` has been off since S12.
- **Golden set**: every project that uses lite-rag prepares 20–30 questions and marks which chunk the correct answer should come from. After switching any layer, run it again, compare hit rate and MRR, and use it to answer "is it better or worse after the change".
- **Usage**: this is the main loop of learning. Each time a chunker is changed, BM25 is added or rerank is turned on, run eval, and use the numbers to judge whether the effect improved.

## 9. Configuration

Read with pydantic-settings, with the environment variable prefix `LITERAG_`; a config file can also be given with `--config file.yaml`:

```yaml
database_url: null        # normally set through the environment variable LITERAG_DATABASE_URL, not written into the file
store: {backend: postgres, schema: lite_rag}   # or {backend: chroma, data_dir: ./lite_rag_data}
embedding: {provider: ollama, model: bge-m3:567m, base_url: null, batch_size: 32, max_concurrency: 4, query_prefix: null}
chunking: {strategy: recursive, unit: tokens, size: 800, overlap: 120, min_size: 50, contextual: {enabled: false}}
retrieval: {mode: hybrid, hybrid_paths: [vector, bm25], top_k: 5, candidate_factor: 4, fusion: rrf, rrf_k: 60, bm25: {k1: 1.5, b: 0.75}}
graph: {enabled: false, concurrency: 4, hops: 2, hub_chunks: 5, max_neighbors: 20}
rerank: {enabled: false, model: BAAI/bge-reranker-v2-m3-int8, top_n: 20}
# lite-model's ModelsConfig (§6, S15). The top level tunes `answer`; the index roles keep their own
# (qwen3:1.7b, thinking off, S17) until `roles` names them. A backend's base_url / api_key_env go
# under `backends`; the pre-S15 keys (llm.provider, graph.llm, chunking.contextual.llm) are refused.
llm: {backend: ollama, model: qwen3:8b, max_output_tokens: 1024, connect_timeout_s: 5,
      roles: {graph: {model: qwen3:1.7b}},
      backends: {openai: {type: openai, base_url: "https://…/v1", api_key_env: OPENAI_API_KEY}}}
cache: {enabled: false, roles: [answer, contextual, graph], ttl_s: null, schema: lite_cache}   # S18
lineage: {enabled: true, keep_text_days: 90}                 # S25 provenance ledger
trace: {enabled: false, query: text, retention_days: 90}     # S25 query traces (query: text | hash | none)
server: {host: 127.0.0.1, port: 8700}
```

## 10. Integration with llm-steward

llm-steward adds a `LiteRagProvider` that implements its `KnowledgeProvider` Protocol. It supports two ways of connecting: in-process, importing the library directly, or HTTP, calling the service. Both ways run llm-steward's contract tests.

| llm-steward call | lite-rag implementation |
| --- | --- |
| `ensure_collection(collection)` | The namespace is `steward/<namespace_id>` (one per llm-steward namespace), the kb name is `c-<collection_id>`, and the kb name is returned |
| `upsert_entry(entry, kb, external_id)` | `upsert_text(kb, doc_id=str(entry_id), title, content)`, returns `doc_id` |
| `remove(external_id)` | `delete(kb, doc_id)` |
| `search(query, kbs, top_k)` | One `search_kbs` (S10; over HTTP `POST /v1/search`), taking the highest score per `doc_id` |
| `health()` | The storage is available, and the embedding service is available |

- capabilities are `{entry_upsert}`. `doc_parsing` and the like are added when llm-steward P3 does documents.
- Governance rules are still guaranteed by llm-steward: only entries in effect are written, and retrieval results pass through its safety net.
- llm-steward and lite-rag can share the same PG instance, but each uses its own database or schema.

## 11. Project structure

```
lite-rag/
  pyproject.toml            # uv; extras: [fastembed] (embedding and rerank) [anthropic], later [server] [chroma] [pdf]
  Dockerfile  compose.example.yml
  src/lite_rag/
    config.py loaders.py chunking.py embedding.py keyword.py fusion.py rerank.py llm.py engine.py
    stores/ base.py pg.py chroma.py          # Store Protocol + two backends
    lineage/                                 # generic ledger (S25): imports no other lite-rag module
    provenance.py tracing.py                 # how RAG artifacts are recorded in the ledger; query trace (S25)
    migrations/                              # PG schema (literag migrate)
    server/ app.py auth.py schemas.py        # FastAPI (extra [server])
    cli.py prompts/answer_system.j2 prompts/answer_user.j2
  tests/   unit/ (fakes)  integration/ (PG: its own test schema; Chroma: temp dir)  -m ollama is optional
  eval/    corpus, golden questions, reports
  docs/lite-rag-design.md
```

## 12. Stages and acceptance

At the end of every stage, ruff, ruff format, mypy and pytest must all pass; every new test must have been seen to fail (red) at least once.

| Stage | Content | Acceptance |
| --- | --- | --- |
| S0 Environment | Install Ollama, pull `bge-m3` and `qwen3:8b`; create the database `lite_rag` and its role on the shared PG; check the APIs and versions of Ollama, the pgvector driver, Chroma and the rerank package against their docs, and lock the versions | `ollama list` shows the two models; `psql` can connect to `lite_rag`, and the `vector` extension is available; the versions are written into `pyproject.toml` |
| S1 Skeleton | uv, ruff, mypy, pytest, config, fakes, CLI skeleton, `literag migrate` | `literag --help` works; the migration runs through on the test schema |
| S2 Ingestion (PG) | Loader; chunkers: `fixed`, `recursive`, `sentence`, `paragraph`, `markdown`; embedders: `ollama`, `openai`-compatible, `fastembed`; `PgStore` (upsert and delete done in a single transaction), `upsert_text`, `delete` | Chunk boundary tests (Chinese and English); a repeated upsert does not rewrite; after a delete, chunks, embeddings and postings are all empty; fault injection: an error in the middle of a write rolls the whole document back |
| S3 Retrieval | BM25 (Python version and SQL version), RRF, three modes, scores of each stage, `literag search` | BM25 agrees with a hand calculation, and the Python and SQL versions agree; fusion order is correct; the real round-trip test under `-m ollama` passes |
| S4 Rerank and answering | cross-encoder, three LLM implementations, answer template, `literag ask`, citation check | FakeLLM test: an out-of-range citation is rejected; switching the provider needs only a config change |
| S5 Eval | Corpus, golden questions, `literag eval` comparison table | A comparison table run on real Ollama: fixed vs recursive vs markdown, vector vs hybrid, rerank on vs off |
| S5b Structured loading and advanced chunking | PDF loading: pypdf (default) and docling (optional, converts to Markdown); chunks record the page number; advanced chunking `semantic` (self-written, and compared with `chonkie`), `parent_child`, `contextual` | Loaders, chunking, storage and retrieval all have unit tests and integration tests (fake models); the real contextual runs through under `-m ollama`; a chunking comparison table and a PDF loader comparison table are produced with eval, and the results go into the KB |
| S5c More embeddings | embedders `voyage`, `cohere`, `gemini`, `sentence_transformers` | Each embedder has unit tests (MockTransport, recorded response structure); an embedding comparison table for local models (with a cost column); paid APIs run only after the user provides a key and agrees |
| S6 HTTP service | FastAPI, API key authentication, Problem Details, Dockerfile, compose example, CI image build | API tests: unauthorised returns 401, access to a kb out of scope returns 403; after `docker compose up`, upsert → search → ask completes; library and API results agree |
| S7b Optional BM25 backend | ParadeDB `pg_search` (inverted index, real BM25) as `store.bm25_impl: pg_search`; needs the ParadeDB image | Done only when a single kb exceeds about a hundred thousand chunks and the measured BM25 latency becomes a bottleneck; a test comparing the ranking with the self-written BM25 |
| S7 Chroma backend | `ChromaStore` plus a SQLite inverted index, and the write order and recovery logic | The same Store contract tests pass on both PG and Chroma; crash recovery test |
| S8 Close-out | `/simplify`, Codex review, fix the problems, then re-run all checks | The review conclusions are on record |
| S9 Integration with llm-steward | `LiteRagProvider` (in-process and HTTP), contract tests, update the llm-steward design doc | llm-steward's KB contract tests pass in both ways of connecting; manual end-to-end: an entry in effect can be retrieved, and after it is retired it cannot |
| S11 Graph RAG (local) | `graph.enabled`: at build time the LLM extracts entities and relations and stores them in three graph tables in PG (in the same transaction as the chunks); the `graph` retrieval path (entity named by the question → expand along co-occurrence → rank by vector within the range), `mode: graph` and `retrieval.hybrid_paths`; the multi-hop eval set `eval/golden-multihop.jsonl` | Extraction: FakeLLM unit tests (valid, with a code block, invalid JSON; endpoints added as entities; names merged; the prompt does not contain the title); expansion checked against hand calculation (hops, hubs, per-hop cap and ranking); PG integration: write, replace, and the graph disappears with the document after a delete; category words and single letters are not seeds; a multi-hop answer can only be found through the graph, and scores agree with the vector path; config conflicts (chroma, graph not enabled, weighted fusion) raise errors; the kb identity contains the extraction model; the real extraction under `-m ollama` parses; the eval before/after comparison goes into the KB |
| S12 Eval moves to lite-eval | `literag eval` becomes a thin wrapper on lite-eval: the adapter `LiteRagTarget`, the golden set converted to a dataset, one setup shared by the same index; the self-written judging and metrics are deleted; an optional extra `eval` pins lite-eval by commit | Equivalence: the frozen old implementation (`tests/legacy_eval.py`) and the new one, in the same process on the same store, give the same `answer_rank` and `doc_rank` question by question on 4 golden sets, and the overall and per-question-type hit@1, hit@5, MRR@10 and doc hit@5 differ by ≤ 1e-9 (FakeEmbedder in CI, bge-m3 with `-m ollama`); the scorer is equivalent on 2000 random hit sets; command arguments and tables unchanged |
| S13 Answer judge (lite-eval E5b) | Depends on lite-eval 0.3.0; the calibration set of the grounded judge (constructed, committed) and the calibration suite; `LiteRagAnswerTarget` and `literag eval-answers`; dataset and suite generation share `write_generated_dataset` / `write_suite` | Calibration set: the label of each corruption is correct by construction, the same seed gives the same set, the committed one agrees with the generator, and the inserted and swapped content is in the same language and not in the sources; FakeLLM + FakeJudge: answers are scored by the calibrated judge, flagged answers are listed, the judge is the one the calibration run measured, and when calibration is not enough it is refused before any answer is generated; the calibration agreement rate and the detection rate of each corruption for qwen3:8b (thinking off and on), the flag rate on real answers and the item-by-item judgements go into the KB |
| S14 Answer correctness and auto-routing | The `key_fact` of golden questions and the `regex` scorer of `eval-answers`; `response_model` (the model written in the LLM response) carried all the way to `ask` and the eval output; `eval-answers` turns off the output cache; four comparison configs (direct Ollama 8b, and small / workhorse / smart-router through the LiteLLM Proxy) | `key_fact` labels by construction (matches the qualified answer, does not match the question, does not match answers with a changed number or a swapped answer); all or nothing; cache off (the second run answers again); FakeLLM: `response_model` is taken from the response, the eval output carries it, and the report says so when not checked; four configs × 2 real runs, the answers that did not pass are read one by one, and the results go into the KB and lite-model design §15 |
| S15 Move to lite-model | LLM calls are now made by lite-model 0.2.0: the three LLM configs become the three roles `answer` / `contextual` / `graph` (the last two not routable, with models fixed in `host_roles`); `RoleLLM`; `model_id` becomes the model that actually answered, `response_model` is removed; truncation counts as an error; old config keys are refused with a note on where they went; the old client is deleted | The 5 requests recorded before the migration (defaults of the three roles, all parameters of an OpenAI-compatible vendor, Anthropic) are the same field by field; the index identity (the eval kb name) is unchanged; `literag eval` (contextual, graph) and `eval-answers` agree before and after; `-m ollama` passes |
| S16 Graph extraction uses structured output | The reply of the graph role is held to a JSON schema (lite-model `structured()`, `GraphReply`); a reply that still does not fit the schema is asked for once more with the same prompt, and if the second still does not fit, the whole document fails; `CODE_VERSION` 2 (the identity of a graph kb changed) | Unit tests: the request carries the schema, a reply that does not fit is asked for once more and only once, a failed call is not repeated, the log does not carry the reply; the recorded request body gains only `response_format`; a chunk that gave bad JSON 7 times in 10 before gives valid output 10 times in 10 with structured output; `literag eval` graph-hybrid finishes all 3 times |
| S17 Index roles use a small model by default | `contextual` and `graph` default to `ollama/qwen3:1.7b`, no thinking; `eval/configs/*-8b.yaml` write 8b back as a control | Unit tests: the default id and parameters of the two index roles, and the top-level settings do not affect them; the 8b control config resolves to the settings before S17; retrieval and answer evals (KB 2026-11-19) are no worse than 8b |
| S18 Connect lite-cache L1 | The `cache` extra (lite-cache v0.1.0 release wheel, off by default); the `cache` config (`enabled`, `roles`, `ttl_s`, `schema`, `question_log`), which uses lite-rag's `database_url` and supports PG only; when on, the runtime's lite-model registers `CacheHook`; the calls of `contextual`, `graph` and `answer` are all inside `call_context(scope="literag/<namespace>")` (lite-cache design §0, §9); `BoundedLLM.map` does `copy_context()` per item (the threads of lite-model 0.2.1's `Bounded.map` do not inherit the contextvar); `Completion` / `Answer` carry `cached`, and the CLI marks a cached answer; `ask()` has an optional question log (JSONL) | When off, equivalent to before (request bodies, index identity and answers unchanged, and the lite-cache schema is not created); when on: after changing the embedding, `reindex --into` a new kb, `contextual` / `graph` send no requests, and the context and graph produced are word for word the same as the first time; the second `ask` of the same question does not call the LLM, and `cached` is true; two namespaces do not hit each other; `contextual` also hits when concurrency is greater than 1; a role not in `roles` is not cached; turning it on when the store is not PG raises `ConfigError`; the question log writes no file when off |
| S19 Embedding goes through lite-model | `OllamaEmbedder` and `OpenAICompatEmbedder` are now called through `embed()` of lite-model 0.6.0 (batching, concurrency, halving on 400/413 and same-model retries are all its job); lite-rag's `EmbeddingConfig` is unchanged and generates lite-model's config; the prefixes (`_Prefixing`), the vector cache (`CachingEmbedder`) and the index identity stay in lite-rag; Voyage / Cohere / Gemini / fastembed / sentence-transformers are untouched | The index identity is unchanged (provider, model, `model_id`, prefixes); under `-m ollama` the vectors of `/v1/embeddings` and `/api/embed` are equal element by element (including the query / document prefixes); throughput is unchanged (the same batch of texts, default concurrency 4 × batch 32); Ollama sends no `dimensions` (it did not before either) |
| S20 Voyage through lite-model, Voyage rerank | `VoyageEmbedder` is now called through the `voyage` type of lite-model 0.7.0 (`input_type` by query / document, `dimensions` as `output_dimension`, halving and retries are lite-model's); `rerank.provider: voyage`: `VoyageReranker` goes through lite-model's `rerank()` (`/v1/rerank`, paid), default `rerank-2.5`; `-m voyage` opt-in | The index identity is unchanged (provider `voyage`, model, empty prefixes, dim); the request body gains only the SDK's `encoding_format: "base64"`; the reranker scores each item in input order, and `model_id` is `voyage/<model>`; a missing key raises `ConfigError` at build time; the fastembed reranker is unchanged and refuses Voyage-only fields; `-m voyage` fails, rather than skips, when there is no key |
| S21 Retrieval routing (measure the ceiling first) | `eval/route_ceiling.py` (reads only existing lite-eval runs: the ceiling when each question picks the right config, the best fixed config, latency); `mode: route`, `retrieval.router.modes`, the `route` role (routable, default ollama/qwen3:1.7b without thinking, at most 256 tokens); `Router` (the JSON schema of each router contains only the modes it offers, a deviation is asked again, and if still wrong it raises `BadReply`); `prompts/route_*.j2` (the question is put in a nonce block as material); `Hit.route`; the `route` span in the eval output; `eval/configs/route-*.yaml` (qwen3:1.7b, llm-gateway's gpt-sub / claude-sub-haiku) | The mode the route picks is the one actually run (the result agrees item by item with specifying that mode directly); one route call per search, none for a fixed mode; no call when there is no kb; a failed call, or two replies both outside the schema, raise errors; when the `route` mode contains graph and graph is not enabled, the config raises an error; lite-eval compares the route with the best fixed config |
| S22 Each kb picks its own embedder | `embeddings.<name>` (named embedder, `embedding` is `default`); `create_kb(kb, embedding=)`, `literag ingest/reindex --embedding`, `{embedding}` in `PUT /v1/kbs/{kb}`; when a kb is opened, the embedder is found again from the identity it recorded (provider, model, prefixes, then dim), with no migration added; `search_kbs` embeds the query only once per embedder, and the lists of the embedders are merged by RRF; a warning, once, when across embedders and rerank is not on; `literag eval --kb-per-dir`, the cross-lingual eval set `golden-xlang` | With only one embedder configured the results are unchanged item by item (scores are the raw similarity); a new kb uses the embedder given, changing the name on an existing kb raises an error, and an unknown name gives 400; writing an existing kb uses its own embedder; `reindex --embedding` changes the model; eval: a single model vs mixed vs mixed + rerank |
| S23 Cohere / Gemini through lite-model | `CohereEmbedder` and `GeminiEmbedder` are now called through the `cohere` / `gemini` types of lite-model 0.8.0; `_HTTPEmbedder` is deleted; `_LiteModelEmbedder._input_type(role)` lets each embedder decide whether the role is sent as `input_type` (Gemini does not send it when there is a prefix) | The existing vendor tests (request URL, auth header, request body, role mapping, batch limit, prefixes, missing key) pass without changing a single one, and the index identity is unchanged |
| S24 Harder cross-lingual eval | `eval/make_xlang_hard.py` generates `corpus-xlang-hard/{en,zh}` (15 near-identical tier / region / version documents) and `golden-xlang-hard.jsonl` (36 questions, 24 of them cross-lingual, the evidence of each question in only one document); `vector-rerank.yaml`, `embed-qwen3-0.6b-rerank.yaml`, `xlang-en-bge-zh-qwen3*.yaml` | Answer: does choosing different embedding models for different corpora do better than the best single model |
| S25 Lineage | S25a the generic ledger `lite_rag.lineage`; S25b ledger records on the write path, `chunks.lineage_key`, backfill; S25c query trace (`search_traced` / `search_kbs_traced` / `ask` / `record_cited` / `trace show`); S25d `lineage gc`, `lineage show/upstream/downstream`, docs | See the S25 record below and spec §9: changing one passage produces only a new version of that passage; deletes cascade; a trace says whether the data used at that time is stale; `lite_rag.lineage` does not import other lite-rag modules; existing retrieval and eval behaviour unchanged |
| S10 Multi-kb retrieval | `search_kbs` and `POST /v1/search` (for llm-steward P4 joint retrieval): embed only once, fuse only once, rerank only once | One embedding; the RRF scores across kbs can be hand-calculated from the merged ranks; the same doc id is returned from both kbs; rerank is called only once; a single kb gives the same result through `search_kbs` and `search`; over HTTP every kb needs authorisation (403); llm-steward contract tests pass |

- **S15 Move to lite-model (2026-11-18, the second host of lite-model design §12)**: LLM calls are now made by lite-model 0.2.0.
  - lite-model 0.2.0 was released first (the owner's choice): a host sets `host_backends` / `host_roles` on a subclass of `ModelsConfig`, and the config is layered field by field on top.
    Without it, any `LITERAG_LLM__ROLES` would knock out the whole default of graph (no thinking, 2048 tokens).
  - `llm.py` keeps only `RoleLLM` (one role against the shared lite-model), `Completion` and `LLMError` (502); `PROVIDERS`, `ProviderProfile`,
    `_Client`, `OpenAICompatLLM`, `AnthropicLLM`, `BoundedCalls` and `build_llm` are deleted. `Message` is lite-model's. `FakeLLM` is based on lite-model's
    `FakeModel`, and the record becomes `fake/fake-model`. The concurrency limit now uses lite-model's `Bounded` (`RoleLLM.bounded()`). `openai` and
    `anthropic` are removed from the dependencies (lite-model pins them).
  - Request bodies: before the migration, 5 requests were recorded with the old client (the defaults of answer, contextual and graph, all parameters of an OpenAI-compatible vendor, Anthropic), kept in
    `tests/fixtures/llm_requests.json` (`tests/data/` was blocked by the `data/` in `.gitignore` and almost did not get committed); after the migration they are the same field by field.
  - Index identity: the role id is still `ollama/qwen3:8b`; the eval kb names (the hash of the index identity) computed by the old code and the new code are exactly the same for the 4 configs, and existing
    kbs need no rebuild.
  - Found and fixed during the migration: (1) the top-level `backend`/`model` used to govern only answer and now move every role — `host_roles` pins contextual and graph to
    `ollama/qwen3:8b`; (2) the old keys `graph.llm` and `chunking.contextual.llm` were **silently ignored** (the graph would be built with the default model) — they are now refused with a note on where they went,
    and the same for `llm.provider` and the like; (3) Codex: the top-level `temperature` and `reasoning_effort` would also reach contextual and graph (under the same index identity the content would be generated
    in a different way) — the top-level call parameters move to `roles.answer`, consistent with `llm` governing only answers before S15. A failing test was written first for each.
  - Behaviour changes (decided by the owner): a truncated answer is an `LLMError`; `Answer.model_id` is the model that actually answered (read from the response, through LiteLLM it is
    `x-litellm-model-name`), and the `response_model` of S14 is removed; `max_tokens_param` is removed (lite-model's profile decides).
  - eval (local Ollama):
    - `eval-answers` (llm-ollama-8b, judge calibrated with think): both times grounded 1.000, key fact 0.969, "answered correctly" 0.969, 0 errors, and no
      answer was truncated; before the S14 migration both times were 0.938. The question that differs is q12 ("资料里没有" (not in the materials)) both times.
    - `literag eval`: graph-hybrid before the migration 0.875 / 0.938 / 0.915 (hit@1 / hit@5 / mrr@10, the same in 3 runs), and the run that succeeded after the migration is exactly the same;
      chunk-contextual 0.875 / 0.969 / 0.912 → 0.906 / 0.969 / 0.928 (one question different).
    - **Graph extraction occasionally returns bad JSON and the whole ingest fails**: 2 of 3 runs after the migration, 0 of 3 before it. Compared item by item (concurrency 1, 49 chunks): the old client
      differs from itself in 22 replies between two runs, and between old and new it is also 22 — qwen3 on Ollama is not deterministic even at temperature 0, and the new client sends the same. It is not a
      problem of the migration but a risk that has always existed: the bad JSON of one chunk fails the ingest of the whole document. Follow-up: graph extraction retries once on a parse error (a separate stage,
      as it changes behaviour).
  - Mutations: 13 places (role pinning, graph no thinking, token cap, error conversion, model_id, routable, old-key refusal, billing, the two identities, the record, FakeLLM),
    all caught.
  - /simplify accepted: `test_ollama.py` still used the deleted `build_llm` (it runs only under `-m ollama`, so the default run does not reach it); billing is decided by the channel lite-model resolves
    (an ollama type with `channel: api` no longer counts as free); `LLMConfig.target_of()` is the only place that calls lite-model `target()`;
    the answer run record records the target's fields by exclusion; an outdated comment. Rejected: merging the two names `LLM` / `RoleLLM` and `BoundedLLM` (types already pass),
    collecting the index identity in one place in `Settings` (the duplicated assembly predates this change), eval-answers building lite-model twice when contextual / graph are on (rare),
    recomputing the role id on every read (tens of microseconds), merging the helpers in the tests with the old-key check (small).
  - Codex accepted 1 (top-level call parameters go only to answer, see above); 1 did not hold up on checking: "a typo under `llm` is silently ignored" — lite-model's config forbids
    extra fields, and `llm.reasining_effort` errors at start-up.

- **S16 Graph extraction uses structured output (2026-11-19)**: a follow-up noted in S15.
  - Cause: after the migration, the graph-hybrid ingest failed in 2 of 3 runs (bad JSON). Following the plan, only "ask again on a parse failure" was added first: 1 of 3 still failed —
    the same chunk (`茶叶分类#1`) was bad both times. Asking about this chunk alone 10 times gave bad JSON 7 times (`{"source": "中国, "target": …`, the model dropped
    a quote, and there were only two or three kinds of reply), and the three recordings of S15 happened to be good ones. Retrying once only brings the failure rate of this chunk from 70% down to about 50%, which is not a fix.
  - Changed to the root cause: the graph role uses lite-model's `structured()`, and `GraphReply` (`entities[{name, type}]`, `relations[{source,
    target, description}]`) is sent to the model as a strict JSON schema (Ollama constrains decoding by the schema). The same chunk is valid 10 times in 10.
    The model's same slip is still there, but it now lands inside a string (4 of 10 runs add an entity called "中国," (China,)): one node is wrong in content, instead of the whole document failing.
  - One retry is still kept: when the reply does not fit the schema (lite-model's `BadStructuredReply` → `BadReply`), it is asked again with the same prompt, and if the second still
    does not fit it raises `GraphError` and the whole document fails; a failed call (`LLMError`) is not repeated here (lite-model has already retried by `max_retries`). Every retry logs
    one warning (the chunk id, without the reply content).
  - The lenient parsing is deleted (code block, text before and after, hand-written field checks): the schema guarantees the shape, and `to_graph` only normalises (drops empty names, merges, adds endpoints).
    The prompt is unchanged (the template part of `prompt_digest` is unchanged), `CODE_VERSION` 1 → 2: the request gains `response_format`, the identity of an existing graph kb no longer
    matches, and it must be rebuilt (llm-steward does not use graph retrieval; eval kbs are rebuilt every time).
  - `FakeLLM` (`lite_rag.testing`) supports structured calls: the scripted text is validated against the schema, and when it does not fit it raises lite-model's `BadStructuredReply`,
    the same as a real endpoint (lite-model also uses `model_validate_json`).
  - Also fixed: the concurrency test of S15 covered `complete()`, while `bounded()` does not go through it — the test verified nothing. Changed to wait on a barrier in the script (with concurrency 1
    it times out, red first).
  - 4 mutations (remove the retry, retry on any `LLMError`, map `BadStructuredReply` to `LLMError`, the schema no longer requires `relations`),
    all caught.
  - eval (local Ollama, `literag eval` graph-hybrid): with only the retry, 1 of 3 runs failed (still bad after the retry) and the other 2 gave 0.875 / 0.938 / 0.915; with
    structured output all 3 runs finished, each at 0.875 / 0.938 / 0.915 (hit@1 / hit@5 / mrr@10, the same as before and after S15), and not one retry was used.
    8 `-m ollama` tests pass. (One intermediate re-run collided with the previous group, still running, on the same schema, and they deleted each other's kb; those two groups do not count.)
  - /simplify: accepted — remove `PARSE_ATTEMPTS` and the loop (fixed at one); altitude pointed out that the root cause was the unconstrained output (the change above). Rejected — move
    `_replies` into `FakeLLM` (only one test file uses it).
  - Codex: round one, 1 finding (the retry changes how the graph is generated and should enter the identity) — after the change to structured output `CODE_VERSION` was already raised; round two, 1 finding that does not hold up:
    "the fake's validation is looser than the schema sent to the model" — the real path is also `model_validate_json`, and both are equally loose.
- **S17 Index roles use a small model by default (2026-11-19)**: after all modules moved to lite-model, qwen3:1.7b and qwen3:8b were compared role by role
  (KB `global/llm-evals/evals/2026-11-19-lite-small-model-per-role.md`, local Ollama, 2 runs per group).
  - Retrieval: contextual with 1.7b builds the kb 3.6× faster (54 s vs 190 s), graph 5.7× faster (70 s vs 401 s); hit@5 is unchanged (0.969 / 0.938), but
    hit@1 drops by 0.06–0.11: for 4 / 3 questions the correct chunk fell from rank 1 to rank 2–3, the same in both runs.
  - Answers (8b answers, calibrated grounded judge, "correct" = key fact and grounded): contextual 0.938 ×2 (built by 8b: 0.969 / 0.938),
    graph 0.969 ×2 (built by 8b: 0.906 / 0.938); the baseline without an index 0.938 ×2. Every wrong question was read: q12 and q04 were wrong already in the baseline, and there is no new wrong question;
    one answer of q04 was truncated at the output limit of the answer role (a problem of the answer role, not of the index). The top 5 all go to the answering model, so a rank change from 1 to 2 does not affect the answer.
  - So the defaults of `contextual` and `graph` are changed to `ollama/qwen3:1.7b`, `reasoning_effort: none` (contextual had none set before, and now it is the same as the measured config).
    answer is unchanged (8b). The role id enters the index identity: an existing contextual / graph kb that used the defaults no longer matches and must be rebuilt, or `qwen3:8b` must be written back in
    `llm.roles` (`eval/configs/*-8b.yaml` do exactly this, for comparison). lite-mem's `extract` is not changed: 1.7b leaves more stale facts, and is not faster.
  - `ollama pull qwen3:1.7b` is needed first. A caller that reads only the first hit (without going through answering) will feel the change in rank; in that case write 8b back.
- **Fixes after the cross-repo check-up (2026-11-20)**: three fixes made after the whole of lite-ai was run through, each with a failing test written first. (1) `CREATE EXTENSION vector`
  gets `SCHEMA public`: when the role and the schema have the same name (`lite_rag`), `"$user"` comes first in the search_path, and on a new database the extension would be installed into the application schema
  and be deleted together with it when the schema is dropped; lite-mem and llm-steward have long written it this way. (2) When PgStore is opened, it checks for migrations not yet run and, if there are any, raises an error and hints
  `literag migrate` (migrations are still run explicitly, not automatically); the local application schema had been behind by the three migrations 0004–0006 without anyone noticing.
  (3) `errors.redact` uses the same rules as lite-mem: `password = x` (spaces around the equals sign), quoted values, prefixes such as `PGPASSWORD`, and passwords in URL parameters.
  In addition, open mode gets the Origin and proxy header checks (§7), made into a middleware alongside `TrustedHostMiddleware` (the rules of the whole open mode are in one layer).
  - /simplify: `pending()` is the same as lite-cache's (lists only file names, does not read SQL); the Origin check moved from a per-request dependency to the middleware; the tests use
    `MagicMock(wraps=…)` to record the SQL sent. Not changed: the check sits in `PgStore.__init__` and not in `open_store` (it is together with `check_vector`, and
    callers that construct `PgStore` directly are covered too); two more round trips at open (each process opens only once).
  - Codex accepted 2 (failing tests first): `redact` also masks `Authorization: Bearer …`, `api_key=`, `token:` and `secret=`;
    open mode also rejects `Via` and `X-Forwarded-Host/-Proto/-Port/-Prefix` (kept in step with lite-mem).

- **S19 Embedding goes through lite-model (2026-11-22)**: decided by the architecture review (2026-11-21): the HTTP embedders are taken into lite-model. lite-mem moved in its S14;
  lite-rag's embedders have more features (concurrency, halving on 400/413, a configurable batch size, `dimensions`), so lite-model was first brought up to par (0.6.0), and only
  the ollama and openai kinds were moved. lite-model does not support the request formats of Voyage, Cohere and Gemini, and fastembed and sentence-transformers run in-process, so those stay as they were.
  - User config is unchanged: the keys and defaults of `LITERAG_EMBEDDING__*` did not change, and `_LiteModelEmbedder` generates lite-model's `ModelsConfig` from them (the backend name is the
    provider name, and openai is keyless when `api_key_env` is not set).
  - The index identity is unchanged (fixed by a test first); Ollama now goes through `/v1/embeddings`, and in the `-m ollama` test the vectors equal those of `/api/embed` element by element, prefixes included; the test turned red when the output was deliberately
    put in reverse order.
  - Throughput: 130 README paragraphs, default concurrency 4, batch 32, two groups before and two after run alternately, 3 runs per group, and the medians are all 4.8 seconds.
  - Old tests changed on purpose: in `test_embedding.py` the wire format of Ollama / OpenAI is changed to the request and reply of `/v1/embeddings`. The retry test no longer checks lite-rag's own
    backoff seconds (the backoff is now the SDK's), only that the same model is retried, the count is right, and 4xx is not retried. New: halving on 400/413, Ollama sends no `dimensions`.
  - Behaviour change: an OpenAI-compatible endpoint without a key (TEI, vLLM) now receives `Authorization: Bearer unused`, which is lite-model's placeholder, because the OpenAI
    SDK insists on a key; it was not sent before. These services ignore it when no authentication is configured; a test guarantees that what is sent can only be the placeholder, never a key from the environment.
    Error messages also no longer carry the upstream text (lite-model rule 8).
  - Close-out: /simplify accepted 6:
    - the dim probe and check are extracted into `_ProbedDim`, shared by the HTTP and lite-model embedders;
    - `_OpenAIShape` was left with only the Voyage subclass and is merged back into `VoyageEmbedder`;
    - the differences between providers become the two hooks `_role` / `_backend`, and the `sends_dimensions` flag is removed;
    - building the lite-model client and converting errors use `llm.build_model` (with a new `roles` parameter), shared with the LLM side;
    - the default Ollama address is left to lite-model's profile, and `OLLAMA_URL` is deleted;
    - a dead alias in the tests is deleted.
  - Codex: no findings. Checks: pytest 564 passed; `-m ollama` 10 passed; ruff and mypy pass.
- **S20 Voyage goes through lite-model (2026-11-22)**: the user wanted Voyage (the embedding provider Anthropic recommends), and wanted lite-mem and llm-steward to be able to use it later too,
  so the `voyage` type was first added in lite-model 0.7.0 (`input_type` of embed, `rerank()`, its design §3.3), and then `VoyageEmbedder` here was switched over.
  - `VoyageEmbedder` is now a `_LiteModelEmbedder`: with `input_type = True` the role of the call (document / query) is sent as lite-model's `input_type`;
    the batch size is still capped at 1000; `base_url` / `api_key_env` are configurable as before. The index identity did not change: provider, model, empty prefixes and dim are the same attributes.
  - Old tests changed on purpose (`test_embedding_vendors.py`): the request body gains `encoding_format: "base64"` (the OpenAI SDK adds it itself when it is not given and decodes; Voyage
    does not accept "float"), and replies with JSON floats are still read; `_base_url` is replaced by the `endpoint` of S19; error messages match the error type names of lite-model
    (`BadRequestError`, `AuthenticationError`), because lite-model's error messages do not carry the upstream text (its rule 8), so there is no more "400" / "401". The request sequence test for halving
    passes as it was (5 → 2 + 3 → 1 + 2).
  - Voyage rerank: `RerankConfig.provider` gets `voyage`, and `api_key_env`, `base_url`, `timeout_s`, `max_retries` are for voyage only (fastembed raises an error if they are set),
    and when model is not written it is `rerank-2.5` (the default model name of fastembed is not sent to Voyage). `RerankError` is 502. The test is `tests/unit/test_rerank_voyage.py`;
    the reply is deliberately in reverse relevance order, to check that the scores come back to input order; when the import fails, everything is red.
  - `-m voyage` (`tests/integration/test_voyage.py`): one real embed (document and query) plus one rerank; it fails rather than skips when there is no `VOYAGE_API_KEY`;
    there was no key this time, so it was not run. The format of the contract tests comes from the Voyage API reference of 2025-12-10, and waits for a real call to confirm.
  - /simplify accepted: the Voyage backend entry is shared by the embedder and the reranker through `voyage_backend()`; `dimensions` is sent by `_LiteModelEmbedder` according to
    `sends_dimensions` (False for Ollama, which still does not send it); an unscored text in the rerank reply raises `RerankError` rather than being filtered silently; fastembed also rejects
    `timeout_s` / `max_retries`. Rejected: one more layer for the construction of `ModelsConfig` (the duplication is small); having lite-model's profile decide `input_type` and removing the flag on the
    class (whether to send by role is lite-rag's choice for each embedder).
  - Codex: nothing that needed a change; one small suggestion held up and was fixed (red then green): `provider: voyage` with fastembed's `cache_dir` set also raises an error.
  - Dependencies: lite-model 0.7.0, lite-cache 0.3.2 (its wheel also points to lite-model 0.7.0). Checks: pytest 570 passed (PG), `-m ollama` 10 passed,
    mypy and ruff clean.
- **S21 Retrieval routing (2026-11-22)**: the user wanted "to pick the retrieval method with a small model first". The embedding is part of the index identity and cannot change per request, so the route picks the retrieval mode.
  A measurement of 2026-11-11 (KB eval-results) said that with bge-m3, picking vector or BM25 correctly for each question only equals using vector alone, so `mode=auto` was not built then. This time the ceiling was measured first.
  - Ceiling (`eval/route_ceiling.py`, three golden sets with 60 questions, six configs: vector, bm25, hybrid, hybrid+rerank, graph-hybrid, graph-only):
    on the 32 golden questions and the 12 exact questions vector already hits all, so picking per question cannot raise it; rerank did not help a single question, and only added about 1 second. On the 16 multihop questions: the best fixed config
    graph-hybrid gets 12, and the ceiling of picking per question is 15. In total: vector 54/60, the per-question ceiling 59/60, picking by question type (multi-hop goes to graph-hybrid, the rest to vector) 56/60. It passed the threshold (≥ 2 questions).
  - Design: the route picks only between `vector` and `hybrid` (vector + bm25 + graph). Graph-only is not offered: in the ceiling table it gets only 13/32 on single-hop questions, and the cost of a wrong pick is too high;
    hybrid with graph is only 2 questions behind on single-hop questions. Rerank does not enter the route: it did not improve any question.
  - Result (qwen3:1.7b route, in the same eval and with the same graph index as graph-hybrid; vector is the mode-vector run of the same day): route 56/60 (32, 12, 12),
    graph-hybrid 53/60, vector 54/60. All the route's picks are right: all 44 single-hop questions pick vector, all 12 multi-hop questions pick hybrid (in the multi-hop set, 4 keyword questions picked both ways, and both hit).
    That is exactly the ceiling of picking by question type; the remaining 4 multi-hop questions are ones graph retrieval itself did not find. lite-eval compare is not significant anywhere (McNemar p ≥ 0.5, too few questions).
    Cost: one more model call per search, p50 about 650–750 ms (local qwen3:1.7b), against 4–15 ms before.
  - Conclusion: the decision of 2026-11-11 is not overturned, and there is no routing by default; `mode: route` stays as an opt-in, suitable where the questions include multi-hop ones and you do not want to send every question through graph retrieval.
    Comparison on llm-gateway (2026-11-22, after the user started the gateway; four configs in the same eval and with the same graph index):
    - gpt-sub: 56/60 (32, 12, 12), the picks are all right as with qwen3:1.7b (in the multi-hop set the 4 keyword questions all picked vector), p50 1.5–1.7 s, more than twice that of the local qwen3:1.7b.
      Conclusion: a larger model has nothing to improve on this set of questions, and the local small model is enough.
    - claude-sub-haiku: all 60 calls failed. The gateway returned 200, but said through `X-LLMGW-Ignored-Params` that the claude-agent-sdk backend does not support an output cap
      (`max_completion_tokens`), and lite-model raised `IgnoredParameter` by its rules (plus one 500). The route raised an error and did not fall back to a default mode, which is exactly the behaviour the design wants.
      To use the gateway's Claude profile in lite-rag, it must be possible to set the `max_output_tokens` of one role to "unlimited": the current `LLMConfig` cannot do it
      (writing null on a role inherits the top-level default 1024, and writing null at the top level is moved to answer by the S15 rule).
      Fix (the user agreed, 2026-11-22): a top-level null stays at the top level, and only non-null values are moved to answer by S15. The result for answer is the same as before (a null moved over before was also
      inherited from the top level); the index roles and route use their own values as before, and only a role that writes null itself becomes unlimited. Test `test_a_top_level_null_cap_lets_a_role_run_uncapped`
      (red then green). `route-gateway-claude-haiku.yaml` now writes `max_output_tokens: null` at both the top level and route.
    - claude-sub-haiku re-run (after the fix, in the same eval as graph-hybrid): on the 57 questions that were answered all picks are right; 3 calls were still a gateway 502 after retries
      (in the gateway log this time, 21 of about 120 requests were 502, `profile=-`, most caught by lite-model's same-target retry), and these 3 searches raised errors, with no fallback to a default mode.
      Hits 53/60 (29, 12, 12), p50 3.2–3.6 s.
    - Summary: none of the three route models picked wrongly on the questions they answered; the local qwen3:1.7b is the fastest (0.7 s), gpt-sub 1.6 s, claude-sub-haiku 3.4 s with intermittent 502s. The default stays
      the local qwen3:1.7b.
  - Tests: `tests/unit/test_route.py`, `tests/integration/test_route.py`, all red when the import fails; the span test added later turned red after the kind of the span was broken.
  - /simplify accepted: the schema of each router contains only the modes it offers (a deviation is rejected by the schema, and the structure is the same as the two attempts of graph); `_answered()` is shared;
    `Route.mode` uses `RouteMode`, removing the cast; `router()` no longer takes the lock once built; a repeated sentence in the prompt uses a variable; the ceiling script uses lite-eval's `load_run`.
    Rejected: running the route and the query embedding concurrently (a local embedding takes tens of milliseconds and is cached, so the gain is small); rendering the prompt only once (every render needs a new nonce);
    moving route from each Hit to a return type (it would change the public interface `search() -> list[Hit]`, which llm-steward uses).
  - Codex: two P2. One held up, fixed after a failing test first: `search_kbs` called the route even when none of the kbs exists. One rejected: "when `mode: route` offers only vector / bm25,
    graph in `hybrid_paths` should not require graph.enabled": `hybrid_paths` is also used by a per-call `search(mode="hybrid")`, and this check has always been independent of `mode`.
    Cache: `route` is not in the default of `cache.roles`; to cache the route, add it.
  - Checks: pytest 584 passed (PG), mypy and ruff clean.
- **S22 Each kb picks its own embedder (2026-11-22)**: the next stage after S21, decided by the user: the embedder is given when the kb is created (not picked automatically), and a new cross-lingual corpus is written.
  - Approach: `Settings.embeddings` holds the extra named embedders; the runtime holds an `Indexer` (embedder, token counter, chunker) per name, and a non-default one is built only when first
    used (a service that queries only default kbs needs no key of another vendor). The kb table already recorded provider, model, dim and prefixes, so no migration is added: when a kb is opened, the configured embedder is found again from the recorded
    identity (`RagRuntime.signature` only reads the config and builds no embedder; dim is compared only when the model is the same but the dimension differs). If none is found it raises `KbMismatch`, and the message
    lists the configured names; with only default configured, behaviour is as before. `search_kbs` groups by embedder and embeds the query once per group, merges within a group by similarity (as before),
    and across groups by RRF; `Hit.scores["vector"]` is still its similarity under its own model. The context budget of the contextualizer still uses the default's counter (it affects only the truncation of
    the LLM input and does not enter the index identity).
  - Cross-lingual eval set (`eval/corpus-xlang/{en,zh}`, 4 fictional policy documents each; `golden-xlang.jsonl` 24 questions: Chinese asking about English documents 8, English asking about Chinese documents 8, same language 8;
    a new field `doc_lang`, by which the report groups). `literag eval --kb-per-dir`: one kb per subdirectory, and when the subdirectory name equals a named embedder, that embedder is used.
    Results (hit@1 / hit@5 / MRR@10, re-run with the final code, numbers agree):

    | Config | hit@1 | hit@5 | MRR@10 | p50 ms |
    | --- | --- | --- | --- | --- |
    | bge-m3 (both kbs) | 1.00 | 1.00 | 1.000 | 7 |
    | Qwen3-Embedding-0.6B | 0.92 | 1.00 | 0.951 | 8 |
    | MiniLM | 0.92 | 1.00 | 0.918 | 5 |
    | multilingual-e5-large | 0.58 | 0.83 | 0.683 | 7 |
    | Mixed: en with Qwen3, zh with bge-m3 | 0.46 | 1.00 | 0.717 | 10 |
    | Mixed: en with e5, zh with bge-m3 | 0.50 | 1.00 | 0.750 | 8 |
    | Mixed Qwen3 / bge-m3 + rerank | **1.00** | 1.00 | 1.000 | 813 |
    | Mixed Qwen3 / bge-m3, hybrid | 0.42 | 0.92 | 0.587 | 14 |

    How to read it: bge-m3 alone is all correct, this set is too easy for it, and mixing cannot do better. When mixed, hit@5 is unchanged but hit@1 drops to half: the first place of each of the two kbs ties in RRF
    (both 1/(k+1)), and when sorted by key the en kb always comes out first; for the doc_lang=zh questions hit@1 is all 0. Rank fusion cannot tell which kb's first place is better, a risk that was
    foreseen in the plan. Cross-encoder rerank makes the scores of the two sides comparable and restores 1.00 (+0.8 s); hybrid with BM25 is worse instead.
  - Conclusion: the capability stays (let kbs of Voyage and the like coexist with local kbs, and change models with `reindex --embedding`); `search_kbs` across embedders should turn on rerank.
    The engine warns once when across embedders and rerank is not on (`RagRuntime.warn_once`).
  - Tests: `tests/unit/test_kb_embedders.py`, `tests/integration/test_kb_embedders.py`, `test_eval.py` (`--kb-per-dir`, the xlang golden set, `doc_lang`, billing, kb name),
    `test_cli.py`, `test_server.py`; all seen red first. The first version of the "same as before with only one embedder" test was too weak (it still passed after fusion was changed to always use RRF),
    and it was changed to compare item by item with cosine similarity and then verified to turn red.
  - /simplify accepted: `Indexer`/the default attributes are stored once (`embedder`/`counter`/`chunker` become derived attributes); the `KbMismatch` text is composed in `_mismatch`; `KbInfo.signature()`;
    `DEFAULT_EMBEDDING` moves to `types.py`; the write path of `_open` does one `get_or_create_kb`, and skips when checked and the name is the same; `create_kb` no longer reads once more; `_search` no longer
    checks the number of embedders in two places (with a single group it is just a merge by similarity); `warn_once` uses the runtime's lock; eval builds only the named embedders it uses, and treats it as one group when there is no grouping.
    Rejected: storing the embedder name in the kb table (needs a migration; matching by identity survives a rename); a signature cache (the cost is small); the contextualizer using a counter per kb (it affects only the budget);
    removing `RagEngine._chunker` (`test_parent_child.py` changes it directly to simulate an underestimate, and by the rules an existing test is not changed, so it stays, with a note).
  - Codex: two P2, both held up, fixed after failing tests first: the billing of `--kb-per-dir` looked only at the default embedder (a named paid embedder would be marked free); the kb name
    used the directory name directly, which could collide or be invalid, and was changed to a number by position. A gap of S21 was found on the way: with `mode: route` the billing did not count the route role, which was added, test first.
  - Checks: pytest 606 passed (PG), `-m ollama` 10 passed, mypy and ruff clean. llm-steward uses only `upsert_text` and `search_kbs`, whose interfaces are unchanged, so its pin is not raised this time.
- **S23 Cohere / Gemini go through lite-model (2026-11-22)**: the two HTTP embedders left after S19 (ollama / openai) and S20 (voyage). lite-model 0.8.0 added for them the
  embed-only `cohere` and `gemini` types (its design §3.4).
  - `CohereEmbedder` and `GeminiEmbedder` become subclasses of `_LiteModelEmbedder`; the batch limits 96 / 100 are as before; `vendor_backend()` is shared by voyage, cohere, gemini and the Voyage reranker;
    `input_type` changes from a boolean on the class to a method `_input_type(role)`, and Gemini returns None when the model profile has a prefix (no taskType is sent, as before).
  - `_HTTPEmbedder`, `_retryable` and `_too_large` are deleted (nobody uses them any more).
  - Verification: the Cohere / Gemini tests in `test_embedding_vendors.py` pass without changing a single one (URL, `authorization` / `x-goog-api-key`, request body, `input_type` / `taskType`,
    `outputDimensionality`, batch limit, the prefix of gemini-embedding-2, error on a missing key), so the requests and the index identity did not change.
  - Codex: both held up and were fixed: the lock file is raised to lite-model 0.8.0 (and lite-cache 0.3.3) in the same change; the passage in the design that described `_HTTPEmbedder` as the current approach is rewritten as history.
    Dependencies: lite-model 0.8.0, lite-cache 0.3.3. Checks: pytest 606 passed, `-m ollama` passes, mypy and ruff clean.
- **S24 Harder cross-lingual eval (2026-11-22)**: the cross-lingual set of S22 was too easy for bge-m3 (24/24) and could not answer "is choosing the model by corpus better". In the new set each policy has
  several almost identical versions (three tiers of a cloud service, two years of a backup policy, machine rooms in three regions, two tiers of procurement, leave for two kinds of employees), differing only in a number or a condition; the question names the version, and
  the evidence sentence is in that one document only (the generation script asserts it, and a test checks it again). 36 questions: Chinese asking about English 12, English asking about Chinese 12, same language 12. `--kb-per-dir`, two kbs en / zh.

  | Config | hit@1 | hit@5 | MRR@10 | doc_lang=zh hit@1 | p50 ms |
  | --- | --- | --- | --- | --- | --- |
  | bge-m3 | 0.94 | 1.00 | 0.968 | 0.889 | 15 |
  | Qwen3-Embedding-0.6B | **0.97** | 1.00 | 0.986 | 0.944 | 6–14 |
  | multilingual-e5-large | 0.78 | 0.92 | 0.834 | 0.833 | 7 |
  | MiniLM | 0.72 | 1.00 | 0.833 | 0.667 | 5 |
  | bge-m3 + rerank | 0.97 | 1.00 | 0.981 | 0.944 | 813 |
  | Qwen3 + rerank | 0.94 | 1.00 | 0.968 | 0.889 | 786 |
  | Mixed: en with bge-m3, zh with Qwen3 (the best for each language) | 0.50 | 1.00 | 0.743 | 0.000 | 7 |
  | Mixed + rerank | 0.94 | 1.00 | 0.968 | 0.889 | 815 |

  How to read it: on English documents bge-m3 and Qwen3 are both all correct, and on Chinese documents Qwen3 is best, so "each uses its own best" is en bge-m3 + zh Qwen3. But without rerank it is 0.50
  (the tie of S22: the first places of the two sides have the same score, and en always comes first); with rerank it is 0.94, which is not as good as Qwen3 alone (0.97). Rerank itself pushed one Chinese question down in rank on this set.
  lite-eval compare: Qwen3 against mixed + rerank, against Qwen3 + rerank, and against bge-m3 each have 3 discordant questions, McNemar p = 1, not significant.
  Conclusion: choosing different models for different corpora did not beat the strongest single multilingual model; the default is still one embedder. The value of choosing the embedder per kb (S22) is to let kbs of different models
  coexist and to make changing models easy, not to raise hits. The default model is not changed (bge-m3: switching to Qwen3 would rebuild the index of every kb, and on 36 questions it is 1 question better, not significant, so whether it is worth it is the user's decision).
  - Test: `test_the_hard_cross_language_set_matches_its_corpus_and_its_generator` (seen red first: the doc of one question was changed to its sibling document).

- **S25 Lineage (2026-11-24/25)**: the owner wanted lite-rag to become a generic module, with answer lineage. After checking the code it turned out that "incremental reprocessing" already exists (document hash skip,
  `embedding_cache_<dim>`, lite-cache L1, foreign-key cascade delete), so the "ledger" approach was chosen: the generic layer only records and queries and does not drive computation; lite-rag's flow is unchanged, and the ledger is written inside the existing transaction.
  Rule 3 (changing the model = a new kb) and rule 4 (fixed chunk ids, whole-document replacement) are unchanged. The decisions and reasons are in the spec; the implementation plan is `docs/superpowers/plans/2026-11-25-lineage.md`.
  - Caught by tests and fixed: the trace time used to be taken after the model answered (a document changed during the answer would show "not recorded"); the race between reading the chunks and taking the time (Codex, changed to use `started_by`).
  - Security review: the source text of a deleted artifact is cleared at once (it used to wait for gc).
  - Trace storage: each trace is about 2×`top_k` (rerank off) to 2×`top_n`+`top_k` (rerank on) rows; `in_context` is not stored and is derived at read time: with rerank off (the default) the trace rows of a search are halved, and those of an ask are about a third fewer.
    Estimate: 10 thousand a day, kept 90 days, rerank on, about 5–10 GB on PG.
  - Added before the merge: `record_cited` once per trace; `literag trace purge`; an ask whose model failed also writes a trace (`failed`); `record_many` batch writes.
  - Not done: edges from a stable chunk to an old document version are not pruned (that is the lineage itself);
    an HTTP query interface for the ledger; llm-steward switching to `search_kbs_traced` + `record_cited` is for after it raises its pin.
  - Close-out records (simplify, Codex): in the commit message of each stage; for S25d see its commit.

## 13. Open questions

- The licenses of each dependency (Ollama, pgvector, Chroma, jieba, bge-m3, bge-reranker, qwen3): checked one by one when the versions are locked, and written into the README.
- How to package rerank: fastembed's cross-encoder or sentence-transformers. To be confirmed from the docs before S4 starts. The former is lighter and needs no torch.
- Which library to use for PDF plain-text extraction: to be confirmed at S2.
- ~~PG driver and async~~ → decided at S1 (2026-11-08): sync core + psycopg 3 (`psycopg[binary,pool]==3.3.8`); vectors are passed in as `float8[]` and converted to `vector` / `halfvec` in SQL, so the `pgvector` Python package is not needed; the S6 service layer calls it in a thread pool, and the concurrency limit is configurable.
- S2 implementation notes (2026-11-08):
  - One `embeddings_<dim>` table per dimension, with an HNSW (cosine) index; dimensions ≤ 2000 use `vector`, 2001–4000 use `halfvec` (the limit of pgvector HNSW), and larger is refused.
  - A document is written in one transaction that locks the `kbs` row of that kb, so writes to the same kb are serial and the BM25 statistics cannot be miscounted. Chunks, vectors and postings are all written in bulk with COPY, one round trip per table; the old version is deleted first, then one UPDATE writes the statistics' increase and decrease into `kbs` at once. The `search_path` of each connection in the pool is set only once.
  - A kb records the embedding it was built with (provider, model, dimension, query prefix, document prefix) and the chunking identity; opening it with a different config is refused, with a hint to keep the original config or use a new kb name. The comparison logic is in the engine (`KbInfo.differences`), and the store only does "fetch it if it exists, create it if not". The chunking identity is declared by `ChunkingConfig.identity()` (it lists only the fields that affect the chunking result; a field added later must be added to it explicitly, so that existing kbs do not become unopenable), plus the counter's identity: `chars`, `estimate` or `hf:<tokenizer>`. `literag reindex` is left to S5b.
  - The namespace and the kb name are passed to the store through `KbRef`; the physical key `<namespace>:<kb>` is generated only in `KbRef.key`, and the CHECK constraint on the `kbs` table guarantees `name = namespace || ':' || kb`.
  - The vector cache is a separate `VectorCache` interface (`PgStore` implements it too), wrapped on by `build_embedder(config, cache=store)`; the cache tables are built separately from the embedding index tables, so reading the cache does not create an HNSW index as a side effect.
  - Chunking is counted with the embedding model's own tokenizer (Hugging Face `tokenizers`, no torch needed). The tokenizer follows the model profile: bge-m3 uses `BAAI/bge-m3`, nomic uses `nomic-ai/nomic-embed-text-v1.5`, a model name of the form `org/name` takes itself as its tokenizer; an unknown model uses an estimate. The config `embedding.tokenizer` can override it, and setting `""` means use the estimate. fastembed also goes through the same prefix table (with its plain `embed`), so the prefix recorded on the kb is the prefix actually used.
  - The `markdown` strategy puts the heading path in front of the chunk before embedding and BM25; `chunks.text` keeps the slice of the original text, used for citations.
  - PDF extraction is postponed to S5b, and S2 supports only txt and md.
- S6 implementation notes (2026-11-08):
  - Decisions (chosen by the user on 2026-11-08): the namespace goes in a request header, and keys are scoped by glob; the documents table stores the original text to support reindex; the image is only built and not pushed; all kbs follow the service's own config (per-kb config is left for later).
  - **reindex**: migration 0005 adds `text` and `pages` to `documents`. `literag reindex <kb> --into <new kb>` rebuilds from the original text into a new name with the current config, and the original kb is untouched; switching means having the client use the new name (replacing in place would need `ON UPDATE CASCADE` on the foreign keys of all kbs, not done for now). If the target already exists, or some document has no original text (written before 0005), it refuses at once and writes none; a failure in the middle deletes the half-built target.
  - Not done in S6: per-kb config; in-place reindex; running the vector and BM25 paths in parallel (retrieval now takes about 4 ms); publishing the image.
  - **Close-out /simplify (2026-11-08)**, changed: the error classes carry their own HTTP status and "whether the message may be returned to the client" (`status` / `expose`), the server is left with a single handler, and no longer depends on an ordered mapping table; `KbNotFound.of()` unifies the wording of "the kb does not exist"; upload goes through `RagEngine.ingest(doc_id=, source=)`, the document id rule is in only one place, `loaders.doc_id_for`, and upload counts while writing the temporary file and stops at once when the limit is exceeded; the retrieval parameters are defined only once; the authentication dependency becomes async and returns only the engine, each request builds a namespace copy on the spot (a shallow copy, with a cost comparable to a dictionary lookup), and there is no cache a caller could blow up; "without a key it may bind only to loopback" is handled by `server.check_bind` (both `run()` and the CLI pass through it); the connection pool size matches `server.max_workers`; reindex first counts the documents missing original text with one SQL, then reads by `doc_id` page by page, without reading the original text of the whole kb into memory. Not changed: splitting the engine into "a runtime not bound to a namespace + a view per namespace" (a change beyond this stage; S8 has split it into `RagRuntime` + `RagEngine`); `create_kb` reading the kb row once more (a low-frequency operation).
  - **Codex review (2026-11-08)**, all 5 held up and were fixed, each with a red-then-green regression test: the upload size limit used to be checked in the route, but FastAPI had already read the whole multipart request and written it to disk before the route ran; now an ASGI middleware rejects by `Content-Length` before parsing, counts while receiving when there is no length, and returns 413 (the limit plus 8 KiB for the multipart boundary); reindex checked "the target does not exist" before writing, so if someone created a kb with the same name in between it would write into it, and on failure delete it; now one `INSERT … ON CONFLICT DO NOTHING` creates the target atomically, and it refuses if it was not created; an upload error carried the client's file name verbatim, and for an unsupported file type even the temporary path on the server; now the wording is always fixed; upstream errors that are not returned to the client have the credentials in URLs and `password=` removed before being logged; the CI smoke test kept going when the service never became ready, and now it fails on timeout and checks that the container is still running. Codex found no path past the namespace or kb authorisation.
  - **Post-commit security review (2026-11-09)**, all 3 held up, fixed separately: open mode without a key relied only on binding to loopback, but a web page in a browser can reach loopback under its own domain by DNS rebinding; now open mode accepts only the Hosts `localhost`, `127.0.0.1` and `[::1]` (Starlette `TrustedHostMiddleware`); the request body limit used to cover only uploads, JSON request bodies had no limit, and FastAPI reads the whole body before authentication; now every request has a limit (`server.max_body_mb`, default 10, uploads still use `max_upload_mb`) and over the limit it returns 413; `GET /v1/kbs` used to list all kbs under the namespace, which told the caller the names outside the key's `kbs` scope; now it lists only what the key can access.
  - End to end (`deploy/compose.yaml`, Ollama on the host) it was found that the image had no `pdf` extra, so uploading a PDF would fail; the image now carries `server` and `pdf` (443 MB). docling would bring several GB of torch, so it is not put in the image; build your own if you want it.
- S11 implementation notes and close-out (2026-11-12):
  - Decisions (chosen by the user on 2026-11-12): lite-rag covers the mainstream RAG methods as far as possible; the first version of Graph RAG does only local mode, and only PG.
  - **Two versions of scoring**: the first version, "sum of entity weights + entity vectors to find seeds", was worse than vector alone on multi-hop questions (see §5); after comparing several scorings offline it was changed to "entity named by the question → co-occurrence expansion → rank by vector within the range", and the entity vector table was deleted. The numbers are in the KB `projects/lite-rag/reference/eval-results.md`.
  - **/simplify changed**: `Contextualizer` and `GraphExtractor` share `llm.BoundedCalls` (the concurrency limiting is left in one copy); the seed rule moves from SQL back to `graph.seeds`; the graph query is split into the `GraphStore` Protocol, so Chroma no longer needs methods that can only raise errors; the graph of each kb is walked only once per retrieval; `among` is an exact scan by primary key; whether the graph counts into the kb identity is decided uniformly by `GraphConfig.index_identity()` (shared by engine and eval); the unused `doc` parameter, the redundant `doc_id` column and two unused relation indexes are removed; the tests are constructed through `support.make_engine`.
  - **/simplify not changed**: the category-word check scans the entity names on every query (precompute when the size calls for it); contextual and graph extraction run in series (the two LLM passes could run in parallel, beyond this stage); the two ways of writing the prompt fingerprint and the barrier stand-in in the tests were not merged (the gain is small).
  - **Codex review**: two held up and were fixed, each with a red-then-green test: `where` used to filter only the final result, and a filtered-out document could still serve as a bridge when walking the graph; now seeds and expansion look only at the documents that are kept; a reply missing the `entities` or `relations` field was written as an empty graph, and now raises an error. One not accepted: the kb identity does not record the `base_url`, temperature and so on used for extraction, for the same reason as contextual — the identity records the model and the prompt, not sampling and transport parameters (eval likewise treats `base_url` as a transport parameter).
  - This stage changed one existing test: the migration list in `test_migrate.py` gains `0006_graph` (a new migration is a change of spec). `test_a_metadata_filter_applies_to_the_graph_path`, written earlier in this stage, verified the old semantics that Codex overturned, and is replaced by `test_a_metadata_filter_also_scopes_the_walk`.
- S10 close-out (2026-11-09):
  - **/simplify changed**:
    - `_search` takes the list of already opened kbs directly, and "raise an error or skip when a kb does not exist" is decided by `search` and `search_kbs` each, so the `skip_missing` parameter and `_readable` are removed; "a path was filled up" is computed after the merge.
    - The key type of multi-kb retrieval `Key = (kb index, chunk id)` is defined only once, in `ranking.py`, and `fusion.py` refers to it.
    - Authentication: `Authorizer.check` takes all the kbs the request involves (in the path or in the body), and the glob check is left in one place; the kbs of `POST /v1/search` are authorised by the dependency `ReadKbs`, through the same entry as the kb in a path, and the route no longer checks by hand (mutation check: after removing this check `test_a_multi_kb_search_needs_access_to_every_kb` fails).
    - `Hit.kb` becomes a required keyword argument and no longer defaults to an empty string; the two retrieval routes share the `_hits` serialisation.
  - **/simplify not changed**: re-querying only the kbs that were filled up when deepening, and fetching the records of several kbs in one query (`WHERE name = ANY`): at the size of a few kbs the gain is negligible, noted in §5.
  - **Codex review**: no problems found.
- S9 implementation notes (2026-11-09):
  - llm-steward's `LiteRagProvider` (`backend/src/llm_steward/providers/literag.py`) maps an llm-steward namespace to `steward/<namespace_id>`, a collection to the kb `c-<collection_id>`, and an entry to a document with `doc_id = entry_id`. It goes over HTTP by default (chosen by the user on 2026-11-09); in-process is an optional extra, and the dependency on lite-rag is locked to a git commit (chosen by the user).
  - Changes on the lite-rag side: add `py.typed`, so that llm-steward's mypy strict can check the calls into lite-rag; `redact` (removes credentials from error messages) moves from `server/app.py` to `errors.py`, because llm-steward also needs it when logging in-process (llm-steward's Codex review found that logs might contain a database password).
  - Acceptance: llm-steward's KB contract tests pass both in-process and over HTTP (the real `create_app`, through ASGI, with API key on), plus one end-to-end test through llm-steward's governance layer; manual end to end with the real `literag serve` (Ollama `bge-m3`) and `steward serve`: an active entry can be retrieved, after it is retired it cannot be retrieved and is deleted from lite-rag in about 2 seconds, and after lite-rag is stopped llm-steward's `/health` reports degraded.
- S8 close-out (2026-11-09):
  - Two old cases that ran only on PG and duplicated the contract tests were deleted (`test_delete_kb_removes_everything`, `test_documents_are_read_page_by_page`), decided by the user on 2026-11-09.
  - **Three changes decided by the user**:
    - The engine is split into `RagRuntime` (built only once, holding the store, embedder, chunker and models) and a per-namespace `RagEngine`. The shallow copy of `for_namespace`, the `server` namespace that `serve` made up, and the three duplicated assembly codes in `from_settings`, eval and the tests are removed; the connection pool size is passed in by `serve`, and the core no longer reads the server config. Models are loaded lazily under a lock, and concurrent requests build them only once.
    - Metadata filtering is unified in both backends as "scalar equality", checked by the engine before the query is embedded (`stores.base.check_where`).
    - The error classes are split: `InvalidRequest` (400, message returned) is the caller's fault; `ConfigError` becomes the service's own config error (500, logged only).
  - **/simplify changed**:
    - the prefix lookup is left in one place, `model_profiles.prefixes_for`;
    - vector normalisation is left in one place, `chunking.unit`;
    - sentence splitting is left in one place, `chunking.sentences`;
    - the citation check parses the numbers only once;
    - CLI JSON output uses one helper;
    - loopback addresses are listed only once;
    - "the kb does not exist" in the store uniformly uses `KbNotFound.of`;
    - `warm_up` also preloads the jieba dictionary (about 1 second); the embedding model is not preloaded, so that every start does not call an API that may cost money.
  - **contextual becomes concurrent**: `chunking.contextual.concurrency` (default 4, not in the kb identity). A runtime has only one contextualizer, and a semaphore limits the total number of concurrent LLM calls, so concurrent ingestion on the server does not multiply them (pointed out by Codex).
  - **/simplify not changed**:
    - pg and chroma sharing the row encoding / paging helper: the SQL and the JSON encoding differ, and splitting it out would separate the column names from the statements;
    - removing `count_missing_text`: without it, a failure would come after some documents have been re-embedded;
    - eval setting the schema only once: only `--schema` going through `load_settings` gets validated;
    - unifying the hash of the content hash and of the eval kb name: the content hash is already stored in the database, and changing the format would make all documents be treated as changed;
    - a store backend registry: there are only two backends;
    - an LLM provider registry;
    - unifying the Chunker Protocol by id;
    - autocommit for PG reads, an in-process LRU for query vectors, running vector and BM25 in parallel, a Chroma read lock, keeping the N / avgdl counters on SQLite: all of these pay off only with a large corpus or high concurrency; noted here, to be done when needed.
  - **Codex review (the whole library, 2026-11-09)**:
    - The upload limit was decided by the path suffix `/files`, so any path ending with it could get the larger limit: changed to an exact match of `/v1/kbs/{kb}/files`.
    - The number of contextual threads multiplied with the number of requests: changed to a shared semaphore.
    - Deleting a document that does not exist reported that the kb does not exist: `DocumentNotFound` is added.
    - Two concurrent upserts of the same content might both return `skipped=False`: the stored result is the same, and `Upserted.skipped` only notes that this is best effort.
    - One that did not hold up: Codex said "a write key scoped to a kb can create a kb outside its range through `PUT /v1/kbs`", but there is no kb-creating route without a kb, and `PUT /v1/kbs/{kb}` checks the glob. A regression test was added (`test_a_write_key_cannot_create_or_write_kbs_outside_its_globs`), and a mutation check can catch it.
  - **Codex round two (after the S8 changes)**: one held up and was fixed: `RagRuntime.from_settings` closed the store only when the embedder / reranker failed to build, and when building the chunker (in `RagRuntime.__init__`) failed the store was not closed; now the whole build is in the same `try`, with a red-then-green regression test. No other problems were found.
- S7 implementation notes (2026-11-09):
  - Store contract tests `tests/contract/`: the `store` fixture is parametrised by `pg` and `chroma`, and goes only through the `Store` / `VectorCache` interface (kb, document, chunk, vector and BM25 queries, vector cache), plus a few engine-level cases (three retrieval modes, filter, parent_child, reindex). The two backends share one Protocol `CachedStore`; `stores/factory.py` opens and migrates by `store.backend`.
  - Chroma-specific tests (`tests/integration/test_chroma.py`): a crash is simulated at each step boundary (without triggering the cleanup of `except Exception`), confirming that after a crash the document is invisible, after reopening both sides are cleaned up, and it can be ingested again; also tested: immediate cleanup on failure, leaving it to the next open when the cleanup also fails, batched writes, filter limits.
  - When `literag eval` uses chroma, the eval kbs go in the subdirectory `<data_dir>/<--schema>`, apart from the application data.
  - **Close-out /simplify (2026-11-09)**, changed: the backend-dependent choices (open, migrate, the data location `location()`, the isolated location for eval `evaluation_store()`) are all collected into `stores/factory.py`, and the CLI and config classes no longer branch by backend; eval decides "the same store" by comparing only the fields that decide the location (PG used to compare `data_dir` too); the default value and validation of `bm25_impl` are merged into one validator; the metadata of each document is parsed only once when BM25 filters; upsert deletes the old vectors by id; the contract tests open the store through the factory. Not changed: moving the vector query outside the lock (a concurrent kb delete would make the query raise an error, and serial reads in a single-process backend are acceptable); caching N and avgdl in memory (they must be invalidated on every write, and computing them on the spot at learning scale takes only milliseconds); sharing the paging and row-conversion helper with pg.py (the SQL on the two sides differs); two tests that overlap old cases tested only on PG are not deleted for now.
  - **Codex review (2026-11-09)**: no problems found. A document that is not complete is never read, and recovery at open cleans both sides; the BM25 statistics and the filter agree with PG; the store's lock serialises the operations in the service thread pool.
  - **Chroma vs PG comparison** (eval corpus of 32 questions, the same set of configs): the three rows mode-vector, mode-bm25 and chunk-markdown have exactly the same hit@1, hit@5 and MRR@10 (0.979 / 0.823 / 0.930), which shows that the vector and BM25 rankings of the two backends agree.
- S5c implementation notes (2026-11-08):
  - The hosted embedders (voyage, cohere, gemini) go straight over HTTP, sharing the batching, concurrency and retries of `_HTTPEmbedder` with the Ollama and OpenAI-compatible interfaces; `_call(batch, role)` hands the document / query role to the subclass, which sends it with each vendor's parameter name. The request and response formats were checked against the official API references of 2026-11-08, the unit tests replay these formats with MockTransport, and the tests never call a real service.
  - `embedding.base_url` no longer defaults to Ollama's address but to `None`, taking each provider's default (`EMBEDDING_URLS`). Before, `provider: openai` without a base_url would hit Ollama, and now the default is OpenAI. The kb identity does not contain base_url, so existing kbs are not affected.
  - sentence-transformers 6.2.0: torch is imported only when the model is first used (the unit tests drop from 27 seconds to 0.3 seconds, and CI is not required to install this extra); 6.x renamed `get_sentence_embedding_dimension` to `get_embedding_dimension`, and the old name gives a deprecation warning.
  - **Embedding comparison** (32 questions, markdown 120/20, vector retrieval only, each model chunked with its own tokenizer; ranx check agrees):

    | Model | Dimensions | hit@1 | hit@5 | MRR@10 | Cost |
    | --- | --- | --- | --- | --- | --- |
    | ollama `bge-m3:567m` (default) | 1024 | 0.97 | 1.00 | 0.979 | 0 (local) |
    | fastembed `intfloat/multilingual-e5-large` | 1024 | 0.94 | 1.00 | 0.969 | 0 (local) |
    | sentence-transformers `Qwen/Qwen3-Embedding-0.6B` | 1024 | 0.91 | 1.00 | 0.953 | 0 (local) |
    | fastembed `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 0.69 | 0.97 | 0.815 | 0 (local) |

    The default is still bge-m3. e5-large is only one question behind and can serve as an in-process alternative that does not depend on Ollama; MiniLM is small (0.22 GB), but on keyword questions hit@5 drops to 0.94 and hit@1 is only 0.69, so it is not suited as the main retriever. The paid models (voyage, cohere, gemini, OpenAI) were not run: they run only after the user provides a key and agrees, and the cost follows each vendor's price list at that time. 32 questions are too few; they can only rule out models that are clearly worse and cannot separate the top three.
  - **Close-out /simplify (2026-11-08)**, changed: each provider's default address, key environment variable and per-batch limit become attributes of its own embedder class (`default_url`, `key_env`, `max_batch`), `build_embedder` now looks up the `EMBEDDERS` table by provider name (a test guarantees that every provider in the config has an implementation), and the config no longer has default addresses that branch by provider; keys are always obtained through `api_key()`, and when none is configured the request carries no auth header; Voyage inherits the OpenAI-compatible implementation, with only an extra `input_type` and a different dimension field name; Gemini's rule is unified as "use the prefix if the model config has one, otherwise send `taskType`", so an unknown Gemini model does not end up with neither; the logic that adds the prefix is written only once in the `_Prefixing` base class, shared by the HTTP and in-process implementations. Not changed: the default number of items per batch of the hosted APIs (Voyage also limits the total number of tokens per request, to be fixed after measuring); the one call at start-up that probes the dimension when `dimensions` is not configured (the kb identity check must know the dimension before writing, once per process; configuring `dimensions` saves it).
  - **Codex review (2026-11-08)**: only one P2, which held up and was fixed: besides at most 1000 items per call, Voyage also limits the total number of tokens per request (which varies by model, for example 120 thousand for `voyage-4-large`), and a batch of long chunks would get a non-retryable 4xx. Now when a batch is rejected with 400 / 413, all HTTP embedders split it in half and resend the parts separately, down to a single item; if a single item is still rejected it raises an error, and other client errors (for example 401) are not split. Codex checked that Gemini's request fields, roles, dimensions and prefixes agree with the official docs, and the change of the default `base_url` is noted in this section.

- S5b implementation notes (2026-11-08):
  - **Page numbers**: `load_file` returns `Loaded(title, text, pages)`, where `pages` is the start offset of each page in the full text (pages are joined by blank lines). A chunk's `page` is the page its start falls on (`bisect`); for a document without pages it is empty, and page numbers are never invented. The content hash includes them only when there are pages, so the hash of existing documents does not change. Migration `0004_pages_parents_context` adds `page`, `parent_n` and `context` to `chunks`, and creates the `parents` table (deleted by cascade with the document).
  - **docling 2.148 measured** (a generated sample of three pages, no OCR): the first run takes about 1 minute, most of it loading the models; every heading level is output as `##` (levels are flattened); the Chinese heading "退货规则" (return rules) is treated as an ordinary paragraph. `export_to_markdown(page_break_placeholder=…)` can keep the page breaks.
  - **chonkie 1.7.0 measured**: `threshold` is a fixed cosine similarity, not a percentile; similarity is first computed over a 3-sentence window, and the similarity sequence must have at least `filter_window` (5) values before it looks for cut points, so text of fewer than about 8 sentences is not cut by meaning and is cut only by size. On a text of 10 sentences about coffee + 10 sentences about tax, its cut falls after the boundary, while the self-written version cuts exactly at the boundary. The default separators have no Chinese sentence-end punctuation, which the adapter adds.
  - **Eval results** (bge-m3, 32 questions, 120/20 tokens, hybrid, ranx check agrees; identical bit for bit to the S5 baseline, which shows that eval is reproducible):

    | Config | hit@1 | hit@5 | MRR@10 | paraphrase hit@5 |
    | --- | --- | --- | --- | --- |
    | markdown (S5 baseline) | 0.91 | 0.94 | 0.930 | 0.88 |
    | semantic (self-written, 57 chunks) | 0.91 | 0.94 | 0.925 | 0.88 |
    | semantic (chonkie) | 0.91 | 0.94 | 0.930 | 0.88 |
    | parent_child 120/480 | 0.91 | 0.94 | 0.927 | 0.88 |
    | contextual (qwen3:8b) | 0.88 | 0.97 | 0.907 | 0.94 |
    | markdown 40 | 0.72 | 0.94 | 0.814 | 0.88 |
    | parent_child 40/120 | 0.88 | 0.97 | 0.916 | 0.94 |

    - The sections of this corpus are all short: after cutting by heading almost every section is under 120 tokens. So the chunks cut by the chonkie version and by parent_child 120/480 are the same as markdown chunk by chunk (49 chunks each), and the same or similar results are decided by the corpus and do not show that the strategy itself is ineffective. parent_child 120/480 is 0.003 below markdown, with only q12 differing: it takes 50 candidates per path (candidate pool 10 × about 5 children per parent), markdown takes 40; the extra chunk ranks 41 on the vector path and 8 on BM25, appears on both paths, and its RRF fused score then exceeds the answer chunk that appears only on the vector path, pushing the answer (originally rank 10) out of the top 10. **The result of RRF depends on how deep each path goes**: the deeper it goes, the more easily a chunk that "appears on both paths" climbs up.
    - The effect of parent_child shows on small chunks: with child chunks of the same 40 tokens, returning the parent raises MRR from 0.814 to 0.916; but it does not exceed using markdown with 120 tokens directly (0.930). Note that a parent is longer and more likely to contain the original evidence text, which favours parent_child.
    - contextual gains some and loses some (hit@5 +1 question, hit@1 −1 question), and a difference of 1 in 32 questions does not count; the cost is one LLM call per chunk at build time (about 3 minutes for the main corpus).
    - semantic: it brought no improvement, and the self-written version is kept (it does not depend on an extra library); chonkie cannot be told apart from the self-written version on this corpus and its cut points are inaccurate on long text, so **it is not made the default implementation**; the adapter is kept as an optional comparison (`[chonkie]`), to be evaluated when there is a corpus of long documents.
  - **PDF loader comparison** (two generated structured PDFs, 10 questions; several sections have the same small headings, and the body text does not contain the section name):

    | Config | hit@1 | hit@5 | MRR@10 |
    | --- | --- | --- | --- |
    | pypdf + recursive | 0.60 | 1.00 | 0.767 |
    | docling + recursive | 0.70 | 1.00 | 0.833 |
    | docling + markdown | 0.60 | 1.00 | 0.750 |

    docling + markdown did not win, and the reason is exactly that the headings are flattened: the `heading_path` of the three "退款期限" (refund period) sections is only "退款期限" in all of them, and the higher-level headings that tell them apart (flights, hotels, trains) are lost; the 120-token chunks of recursive often carry the line "## 机票" (flights) along instead. 1 question in 10 is already 0.1, and generated PDFs are cleaner than real PDFs, so the conclusion is only: **the default is still pypdf**, and docling stays optional; to really use the structure, a parser that can restore heading levels is needed (for example by inferring from font size), which is beyond v1 (§1).
  - eval: the corpus is loaded separately by PDF loader, and the loader enters the index identity; before comparing against the evidence text, Markdown marks (heading symbols, bold, escapes) are removed, and so is the whitespace between Chinese characters (a PDF breaks lines in the middle of a sentence).
  - **Close-out (2026-11-08)**: /simplify (4 angles: reuse, simplification, efficiency, layering) changed: prompt rendering and the nonce guard are merged into `lite_rag.prompts.render_pair`; the kb identity of contextual uses `contextual.prompt_digest()` directly; the heading regex is left in one copy, `loaders.ATX`; the docling converter uses `@cache`; `embedding_text(chunk, header)` decides the indexed text uniformly; parent_child first dedups by parent, then cuts to the candidate pool, then reranks (rerank no longer scores 4 times as many), the engine no longer checks the strategy name and the chunker declares `children_per_parent` instead (taking overlap into account), and the dedup key uses `parents.n`; contextual counts tokens only once per document; semantic normalises each vector only once. Not changed, and why: pypdf's Chinese line breaks are not merged in the loader (the line break between a heading line and the next line cannot be told from a break in the middle of a sentence, and merging would stick the heading into the body; they are normalised only when eval compares); the parent text is not split into a second query (one search is at most a few hundred rows, and one more round trip is not worth it); concurrent contextual calls are postponed to S5c (the concurrency of local Ollama is decided by the server); `semantic` is not put into `STRATEGIES` (this registry holds the strategies that need no model, and the generic tests iterate over it); the identity fields are not changed to a table lookup (there are only 3 conditions, and a derived value such as `effective_parent_size` must be handled); `run_configs` still accepts a dict or a function (existing tests use a dict).
  - **Codex review (2026-11-08)**, fixed: `children_per_parent` is only an estimate, and when the search cannot fill the candidate pool after dedup and some path was filled up, it doubles the depth and fetches again (this also covers the race between retrieval and deletion); `top_k=0` is no longer treated as unspecified, and less than 1 raises an error; the identity of contextual gains `CODE_VERSION` (when code rules such as window cutting and reply cleaning change, the version must be raised); the rerank test is changed to assert that what is sent for scoring is one child per parent and that the final order follows the rerank scores; chonkie gained tests for `unit: chars` and for an overlong string without break points. Dismissed, and why: a cross-document parent wrongly deduped (the primary key of `documents` is `(kb, doc_id)`, retrieval is inside one kb, and `(doc_id, n)` is already unique); a nonce tag that the body can close (`_material` escapes every `</source…`, and an existing test covers it); chonkie going beyond `size` in chars mode (it uses lite-rag's own counter and `_fit` is applied afterwards, verified by a new test); an empty page misaligning the page numbers (there is a separator between pages and start points are not duplicated, and a test was added to lock it); semantic sending an overlong sentence to embed (every sentence first passes `_fit`, down to the token window); a change of identity of existing semantic kbs (there was no semantic before S5b); golden questions validated against only one loader (after the simplify they are validated against each loader).
- S5 implementation notes (2026-11-08):
  - **The golden set is annotated by "evidence text"**, not by chunk id: each question states the document where the answer is and a passage of original text. A hit counts as correct only when it both "comes from this document" and "contains this passage" (compared after unifying whitespace). This way the same set of questions can compare different chunking methods; otherwise the annotation would become invalid whenever a chunk id changes. When the golden set is loaded, it is checked that each passage really appears in its document.
  - **Corpus**: 8 self-written documents under `eval/corpus`, 4 in Chinese and 4 in English, of public common knowledge, plus one fictional policy document. `eval/golden.jsonl` has 32 questions in all, 4 per document: 2 ask with keywords from the text, 2 ask in other words.
  - **`literag eval -c a.yaml -c b.yaml`**: outputs hit@1, hit@5, MRR@10, document-level hit@5 and p50/p95 latency, and hit@5 is also broken down by question type. Several design points (S5 simplify, 2026-11-08):
    - Configs with the same index config (chunking + embedding) share one kb, built only once, and then each queries with its own retrieval and rerank settings. The kb name is taken from the hash of the index config.
    - All configs share the same store, and the embedder and reranker are built once per config.
    - Eval uses its own schema `lite_rag_eval` (migrations run automatically), does not touch application data, and the kbs are deleted at the end.
    - The config file takes priority over the `LITERAG_*` environment variables and `.env` (`load_settings(file_wins=True)`); otherwise the settings in a developer's `.env` would silently override the config file, and several configs would in fact be measuring the same settings. Every report records the settings actually in effect at the time.
    - Latency counts only retrieval (plus rerank) itself: the query vectors of all questions are computed before timing, followed by one warm-up search. On the first run fixed measured 40 ms, because the first config had to compute a query embedding on the spot for each question while the later configs read the cache directly, and it was not "the first load of the tokenizer" as written at the time, which has been corrected.
    - The retrieval depth is fixed at 10 (to compute MRR@10). In hybrid mode each path takes 10 × candidate_factor candidates, a deeper pool than with the default top_k=5.
    - Fixes after the Codex review (2026-11-08):
      - When judging "whether the index is the same", all the other embedding settings take part in the comparison, except the settings that affect only how data is fetched (batch size, concurrency, retries, timeout, cache). So even with the same model name, a different endpoint counts as a different index.
      - A kb is registered before the build starts, so that when the build fails, the half-built kb is also cleaned up. In the cleanup the kbs are deleted one by one, the original error is raised as usual, and problems met during the cleanup are attached to it as notes.
      - p95 is computed with the nearest-rank method, so the result is always an actually measured value.
      - All configs must use the same database; when `--config` is not passed, a clear error is given.
      - The evidence text of 8 questions was rewritten (q10, q11, q19, q24, q25, q26, q28, q30), and now each passage contains both the object asked about and the answer, for example "有害垃圾……例如废电池" (hazardous waste … for example used batteries), and no longer just a list of examples.
  - **The metrics were checked independently**: `eval/crosscheck_ranx.py` recomputed hit@1, hit@5 and MRR@10 of 6 configs with ranx 0.3.21, exactly the same as our results. ranx's dependencies are too heavy (pandas, numba and so on), so it was not added to the project dependencies, and is run temporarily with `uv run --with` when needed. (Replaced since S12 by `literag eval --crosscheck`, and the script is deleted.)
  - **First measurement** (bge-m3 + qwen3 run locally, chunk size 120 tokens / overlap 20):

    | Config | hit@1 | hit@5 | MRR@10 | paraphrase question hit@5 | p50 ms |
    | --- | --- | --- | --- | --- | --- |
    | fixed + hybrid | 0.88 | 1.00 | 0.919 | 1.00 | 5 |
    | recursive + hybrid | 0.84 | 1.00 | 0.905 | 1.00 | 4 |
    | markdown + hybrid | 0.91 | 0.94 | 0.930 | 0.88 | 5 |
    | markdown + vector only | 0.97 | 1.00 | 0.979 | 1.00 | 3 |
    | markdown + BM25 only | 0.75 | 0.91 | 0.823 | 0.81 | 2 |
    | markdown + hybrid + rerank | 0.97 | 1.00 | 0.979 | 1.00 | 1061 |

    Conclusion: on this small corpus, vector-only retrieval does as well as adding rerank; hybrid is instead dragged down by BM25, and the problem is concentrated in the questions asked in other words, where the wrong results that BM25 brought in took the place of the correct answer through RRF. Rerank can make up the loss, at the cost of about 1 more second per query. 32 questions are too few, and these gaps may not be stable; the only thing that can be said fairly surely is that with BM25 only, the paraphrase questions get clearly worse. The default config is not adjusted for now, to be decided after the corpus and questions are expanded.
  - **What is written ourselves and what uses a library** (decided 2026-11-08): chunking, BM25, fusion and the eval metrics are implemented by ourselves, since these parts are exactly the focus of learning in this project, and they also need to fit closely with exact offsets, kb identity information, namespaces and writes in the same transaction; the low-level parts that are easy to get wrong all use mature libraries, including pgvector, PostgreSQL, jieba, tokenizers, fastembed, Ollama and the official SDKs. For semantic chunking, chonkie is evaluated first (S5b); `pg_search` is added as a BM25 backend when the data grows (S7b); no whole framework such as LangChain or LlamaIndex is introduced.
- S4 implementation notes (2026-11-08):
  - **Rerank**: uses fastembed's `TextCrossEncoder` (extra `[fastembed]`), with the default model `BAAI/bge-reranker-v2-m3-int8` (Apache-2.0, multilingual, about 0.6 GB). `jinaai/jina-reranker-v2-base-multilingual` is under the CC-BY-NC license and is not used as the default. Rerank is off by default, and can be turned on with `search(rerank=True)`, `--rerank` or the config item `rerank.enabled`. When on, the candidates ranked in the first `rerank.top_n` (default 20) of the fusion result are handed to the cross-encoder for scoring; the text sent to the model contains the chunk's title and heading path. Every hit records `scores["rerank"]` and `ranks["rerank"]`.
  - **LLM**: `ollama` and `openai` share the OpenAI SDK, with only the base URL different, and Ollama uses the local `/v1`. `anthropic` uses the official SDK (extra). Some details that were measured or checked:
    - The differences between providers are recorded in the table `llm.PROVIDERS` and not written as code branches: the default base URL, the default key variable, which field the token cap uses, and the default temperature.
      - Ollama does not recognise `max_completion_tokens`, so it gets `max_tokens`; OpenAI gets `max_completion_tokens`. When an individual OpenAI-compatible endpoint recognises only the other field, override it with `llm.max_tokens_param`. The config item itself is called `max_output_tokens`.
      - Default temperature: 0 for Ollama; for OpenAI it is not sent by default, because its reasoning models reject a non-default temperature.
      - anthropic does not support `temperature`, `reasoning_effort` and `max_tokens_param`; if they are configured, an error is raised when the config is loaded, and they are not dropped silently.
      - `reasoning_effort` accepts only a fixed set of values, and a wrong one raises an error at config load.
      - The openai 3.x and anthropic 2.x SDKs both use httpx2 underneath, not httpx, so the timeout must use the `Timeout` that the SDK itself exports (connect timeout 5 s by default, read timeout 120 s by default, `max_retries` 1 by default).
    - Ollama puts the thinking process of qwen3 in the `message.reasoning` field, and it does not get mixed into `content`; `reasoning_effort: none` turns thinking off, and a one-sentence answer drops from about 350 tokens to 7.
    - The `messages.create` signature of the anthropic 2.6 SDK no longer has a `temperature` parameter, so it is not passed.
    - When rerank is on, the reranker is created already in `from_settings`, so a config error is found at start-up, though the model is loaded only on first use. The LLM is created only on first use, so an engine that only does retrieval needs no LLM key.
    - "Where the chunk is" is produced by one function only: `chunking.chunk_context(title, heading_path)`, used by the reranker, the prompt template and the CLI. The embedding text adds only the heading path and not the document title, which is intentional: the title would repeat in every chunk of the same document and pull these chunks too close together in vector space.
    - `Answer.status` takes the value `answered` or `no_sources`. With no hit, `text` is empty, and the hint is rendered by the caller in its own language; the CLI shows English. When citations are removed, only the citation marks themselves are handled, and other text written by the model is not touched.
    - Codex review fixes (2026-11-08):
      - **Sources cannot be forged**: each source is put in `<source-NONCE id="n" from="title > section">…</source-NONCE>`. The NONCE is a 12-digit hexadecimal number generated at random every time the prompt is built (suggested by the security review), which a document cannot know in advance, so it cannot open or close a source block. As a second line of defence, `<source…` and the current tag in the document body are both escaped; quotes and angle brackets in the `from` attribute are also escaped. The system prompt states that the text in a source is only material, not an instruction.
      - **Record the fused level**: in hybrid mode `Hit.scores` and `Hit.ranks` also contain `fused` (without rerank, `score` equals it).
      - **A chunk deleted in the middle of a search**: the chunk contents are fetched for the whole ranking list, those already deleted are removed first, and then the needed number is cut off, so the places left empty are filled by the candidates behind.
      - **An error when the model returns an empty reply**: when the content is empty or only whitespace (for example because the token budget was all used up by thinking), `LLMError` is raised, and it is not returned as a normal answer.
      - **Valid grouped citations are kept as they are**: when all the numbers of a group are valid (such as `[1, 3]`, `[1，3]`), the model's original way of writing is kept; only a group that contains an invalid number is rewritten.
    - Not implemented for now (evaluated in the S4 simplify): `reasoning_effort=none` by default for ollama (so far verified only on qwen3, and what happens with a model that does not support thinking is not yet known, so it is only a recommendation); rerank text truncation, and merging overlapping chunks in the prompt (to be decided with the S5 eval data); warming up the rerank model at service start-up (S6).
  - **Answering**: the prompt is in the two templates `prompts/answer_system.j2` and `answer_user.j2`, using StrictUndefined, so a missing field raises an error directly. The numbers cited in the model's answer are checked one by one by the code: a number outside the range 1..k is removed from the text and recorded in `Answer.dropped`. When there is no hit, the model is not called. `Answer` records `provider/model` and the token usage.
- S3 implementation notes (2026-11-08):
  - **BM25 has two implementations**: `keyword.py` (Python, easy to read) and SQL (faster), chosen by `store.bm25_impl`. Both share the same statistics CTE: N and avgdl come from `kbs`, and n(t) is counted with a window function **before** the metadata filter, so filtering only narrows the candidates and does not change scores, with tests checking each of the two implementations. The postings table stores the chunk's `length` and `doc_id` directly (migration 0003), so scoring reads only the postings table and does not need to look back into `chunks` row by row.
  - The transaction of the **vector query** sets `hnsw.iterative_scan = strict_order` (pgvector 0.8). Several kbs share the same `embeddings_<dim>` table and the same HNSW index, and the kb filter happens after the index scan; without this parameter, once the planner chooses HNSW the candidates (`ef_search`=40) may all belong to other kbs, and the result would fall short of top_k. When the kb is very small, the planner usually filters by kb with the primary key first and then sorts exactly, which is correct as it is. Measured: with 2000 other vectors in the same table, turning the parameter off returns only 1 row, and `strict_order` returns 5.
  - **Read operations do not create a kb**: searching in a kb that does not exist raises `KbNotFound`, because a misspelled name should raise an error and not return an empty result; a delete operation returns False, which keeps it idempotent. A write operation creates the kb when it does not exist.
  - The `options` that the user wrote in the connection string are kept, and `search_path` is appended after it.
  - The pgvector version is checked both when the store is opened and when `literag migrate` runs, and it must be at least 0.8 (`iterative_scan` needs it).
  - Only hybrid mode takes candidates from the two paths by `top_k × candidate_factor`; a single mode takes only `top_k`. `Hit.scores` holds only the scores of each stage itself, and the final score is `Hit.score`.
  - **Ties and floating-point error** (Codex review, 2026-11-08): the terms of each chunk's score are added in a fixed order of words; the Python version sorts first and then sums, and the SQL version uses `sum(... ORDER BY term)`. This way the score of a chunk does not differ in the last digit because the database returns rows in a different order. When sorting, scores are first rounded to 9 decimal places (`ranking.PRECISION`, SQL uses `round(score::numeric, 9)`) and then compared, two scores that differ only by floating-point noise are treated as a tie, and ties are ordered by chunk_id.
  - **Read operations check the kb every time**: whether the kb exists and whether the identity matches is confirmed again by every read operation, at the cost of one more light query. This way, after another process deletes the kb, the next search raises `KbNotFound` and does not return an empty result. A write operation checks only once per engine; if the kb is deleted afterwards, the locked write raises `KbNotFound`.
  - `stats()` does no identity check, which is intentional: its very use is to see with which model and config a kb was built, and it must be visible even when the config does not match; when the kb does not exist it returns None (the CLI shows "does not exist"). Codex suggested that it also go through the check of read operations, which was not accepted, for the same reason as above.
  - Not implemented for now (evaluated in the S3 simplify, 2026-11-08): running the vector and BM25 paths in parallel, left to the S6 service layer; making read operations use autocommit and moving cache writes off the request path, to be decided when there is measured data; computing the vector distance only once, which needs EXPLAIN first to confirm that it does not affect HNSW.
  - In tests, data where HNSW recall is incomplete must be built with evenly distributed random vectors. The 8-dimension vectors generated by the bag-of-words fake embedder contain many duplicates, which make some nodes in the HNSW graph never reachable by search; a real model does not have this.
- S12 implementation notes (2026-11-14, eval moves to lite-eval):
  - The adapter `LiteRagTarget` implements lite-eval's Target protocol: `config()` is `effective_settings` and `setup_identity()` is `index_identity`, so configs of the same index are grouped by lite-eval, the first builds the kb, and the others take over its kb and embedder; the kb is deleted only after the whole group has run. When setup fails (building the kb or computing the query vectors), the adapter itself deletes the kb (lite-eval tears down only targets whose setup succeeded), and if the delete fails again it is attached to the original error as a note.
  - In setup all the query vectors are computed without timing (stored in the embedding cache, so this works only when `embedding.cache` is on, as before S12); lite-eval does one more warm-up search before timing. Each run records the basis of the latency in `target.info.latency_scope`, the table footnote is written accordingly, and configs with a different basis are stated separately. Concurrency is 1 (retrieval is synchronous), and the output cache is off (latency is a measurement of this run, and a cached output would carry the latency of an old run).
  - The corpus is parsed only once per PDF loader, and the embedder and reranker are built only once per config; the `python:` entry `built_in_code` of the generated suite only raises an error: the target is constructed in code (sharing one store), and this suite is only for the record and cannot be re-run alone. A lite-eval error (such as a setup failure) is one `error:` line in the CLI, with exit code 1.
  - Dependencies: lite-eval declares a version range (its AGENTS rule 13), lite-rag's docling extra requires typer < 0.28, and it resolves to typer 0.27.3.
  - After the migration, the real local run (bge-m3) of the three rows mode-vector, mode-bm25 and chunk-markdown is the same item by item as the S5 baseline (2026-11-08) in hit@1, hit@5 and MRR@10 (run once more after the close-out changes, still the same).
  - Verification: pytest 480 passed (both installs, `--all-extras` and CI's `--extra anthropic`, were run); the bge-m3 equivalence test of `-m ollama` passes; 19/19 broken scripts turn red.
  - **/simplify changed**: when a config name is turned into a lite-eval name, leading non-alphanumeric characters are removed (`_x` and `.hidden` used to be rejected by lite-eval); a kb is also deleted when computing the query vectors in setup fails; the corpus is parsed only once per loader (before, once per setup group: with two groups, 8 documents were parsed 24 times); golden normalises each document only once; `make()`, the nullable store and `_owns_store` are removed, and the factory becomes a required parameter; a failure to delete a kb raises `StoreError`; `--out` writes lite-eval's run record plus the per-question ranks; under the table, the status and warnings of each run are printed one by one; lite-eval's errors become a one-line error in the CLI. The first four each have a red-then-green test.
  - **/simplify not changed**: lite-eval publishing `run_suite(targets=…)` (waiting for a second user; lite-rag pins lite-eval by commit and runs the equivalence tests at every upgrade); lite-eval also calling teardown after a setup failure (it would change the Target protocol, so it is handled in the adapter instead); merged ranx calls and duplicate copies of the dataset (the time is negligible).
  - **Codex review**: one held up and was fixed (red then green): the table footnote always said "the query vectors have been computed", but when `embedding.cache` is off the latency in fact includes the query embedding, and now it is written by the basis recorded for each run. One not accepted: Codex thought the parse-count test also counted the `load_corpus` of the check, but in fact `len(parsed)` is evaluated first and the test is right (a broken script turns it red); but the way it is written depends on the evaluation order, and it was changed to compute the expected value before patching.
  - **Two things found by CI after the push** (both red then green): `cli.py` imported lite_eval at the top level, while the service image does not install the `eval` extra, so the whole `literag` could not start (the image smoke test failed). Now lite-eval is imported only in `literag eval`, and when it is not installed a one-line "install the eval extra" error is given; `_errors` recognises lite-eval's errors only when `lite_eval.errors` is already loaded. The other: the CLI error test was refused ahead of time in CI by the billing guard of `CI=true`, and the test now clears `CI`.
- S13 implementation notes (2026-11-18, the grounded judge of answers, lite-eval E5b):
  - Depends on lite-eval 0.3.0 (release wheel); after the upgrade all of lite-rag's tests (including the equivalence tests of S12) pass without changing a line.
  - **Calibration set** (`eval_grounded.py`, generated by `eval/make_judge_grounded.py`, committed; a test checks that the committed one agrees with the generator):
    - The qualified answer = the sentence where the evidence is (sentence splitting recognises only `.!?` followed by a space at the end of a sentence, and `。！？`, which is narrower than the sentence splitting of chunking: the base answer must be a sentence that a reader would quote whole), with the paragraph that cites it as source [1]; a question whose evidence spans sentences (q07) has no clean base answer and is not used.
    - The mechanical corruptions use lite-eval's `judges/construct.py`: change a number, add a sentence from another document, replace with the answer of another question, remove the citation. Insertion and swapping take only other documents of the **same language**: the language is taken from the golden set (the language of a document = the language of its questions) and is not guessed from characters (found by Codex: an English sentence quoting one Chinese character would be treated as Chinese).
    - Hand-written corruptions (`hand-corruptions.jsonl`, two per question): stating the opposite of something the source says (`contradict_source`), and adding a sentence that sounds reasonable but that no source has said (`plausible_elaboration`). A test checks that the hand-written and inserted content is not in the whole document of that question.
    - Real answers (`real-answers.jsonl`): the 32 answers of the first real-answer check, with all the sources they received, annotated after reading one by one (28 grounded, 2 unsupported, 2 unclear kept on file and not used); the annotation was read by a model and awaits a spot check by the owner.
  - **Why there are hand-written corruptions and real answers**: with only mechanical corruptions, qwen3:8b (thinking off) was 262/262 correct; but reading real answers, it let through two problematic answers (q04 added an explanation that the source does not have, q20 contradicted a source it cited), and judged one faithful paraphrase (q27) as unsupported. The labels of a construction inherit the shape of the construction (methodology §2, lite-eval design §3 L2): the errors of a real model are reasonable-looking additions and inconspicuous contradictions, not whole sentences replaced.
  - **Results** (223 cases × 2 runs, temperature 0, the two runs exactly the same): thinking off overall 0.955; mechanical corruptions, qualified answers and reversed sources are all 1.000; plausible additions 0.774 (7 of 31 missed: all attributions, mechanisms and rules of thumb, such as "SCA standard" and "national regulation"); real answers 0.900, but neither of the two problematic real answers was caught. Conclusion: it can only be a coarse screen (finding wrong answers, contradictions, fabrication) and **cannot be a gate**. Thinking on: overall 0.982; plausible additions 0.968 (1 of 31 missed); real answers 0.933, no longer misjudging that faithful paraphrase, but the two problematic real answers are still both not caught (they cite several sources, and the judge considered the added claim "explicitly or implicitly" supported); one new misjudgment (q21 with the citation removed, where the judge read the spring equinox half-sentence as the awakening of insects); p50 11.8 seconds, 3.7 times that of thinking off. The conclusion is unchanged: if used, use thinking on, but an added claim across several sources of the kind in the real answers is caught by neither setting, and two samples are not enough to measure the miss rate.
  - **Switching to another model family** (decided by the owner 2026-11-18, through llm-gateway's gpt-sub, ChatGPT subscription quota; suite `eval/judge-grounded/suites/calibrate-gateway.yaml`, 1 repetition): overall 0.996; plausible additions 31/31, and both problematic real answers were caught (q20 also pointed out that the added words contradict the source [5] it cites). The only mistake is the qualified baseline of q28: the answer is the original sentence of the source, but the judge ruled it unsupported because it "does not directly answer the question", mixing relevance into whether there is grounding. Next step: write into the rules that only whether the source supports it is looked at (changing the rules means changing the instrument, so everything must be judged again); before using it as a gate, run once more on `gpt-api` (a subscription channel is not the API). The labels of the real slices and the hand-written corruptions were still written by a model, awaiting the owner's check.
  - **An attempt to tighten the rules** (same day, not accepted): "do not judge whether the question was answered" was added to lite-eval's grounded prompt, and both wordings fixed q28, but each let through one answer that was replaced with another question's answer (q01, q08; the judge's own reasoning said it was not in the sources, yet the conclusion was grounded). On the 119 cases judged by all three rules: the original rule 119/119, the two tightened versions 118/119 each. A gate fears most letting an error through, so the original rule is kept, and q28 stays as a known false positive. The third round ran out of ChatGPT subscription quota at about 120 calls (upstream 429), and the 0.529 of that round does not represent the judge's level.
  - **gpt-api did not get run** (same day): `calibrate-gateway.yaml` gained a `gpt-api` config (through llm-gateway, gpt-5.6-luna, billed per token). On trying to judge 1 case, OpenAI returned 401, the local API key was not accepted, and no cost was incurred. After the owner puts in a valid key, judge 1 case first, estimate the cost of 223 cases from the real token usage, and then the owner decides whether to run the full set.
  - **To be measured (do it after quota or key is restored; development is finished, noted 2026-11-18)**:
    1. **A second repetition of gpt-sub**: after the ChatGPT subscription quota is restored, run `calibrate-gateway.yaml --config gpt-sub` once more with the original rule (prompt version `26e4dc07801c`), to see whether the subscription channel is stable on the same cases (it does not guarantee to obey temperature 0). About 223 subscription calls.
    2. **gpt-api**: after the owner puts in a valid OpenAI key, judge 1 case first, estimate the full cost from the real token usage, and with the owner's agreement run `--config gpt-api` in full. This must be done before taking gpt-sub as a gate (a subscription channel is not the API).
    3. **New real answers**: run another batch of `literag ask`, annotate one by one, and use them as a new real slice. When the grounded rule is changed later, compare with this batch, and stop tuning the prompt on these 223 cases.
    4. **The owner checks the labels**: the labels of `real-answers.jsonl` (q04 and q20 unsupported, q27 grounded, q03 and q28 unclear) and the 62 sentences of `hand-corruptions.jsonl`. If anything changes, regenerate the calibration set, and all the items above must be run again.
  - **`literag eval-answers`**: `LiteRagAnswerTarget` (a subclass of `LiteRagTarget`, `query_key = "question"`, given `llm_for` at construction) runs `ask` on the golden questions and outputs `{question, answer, sources, status, dropped}`, and the latency includes one answering call; billing uses `billing(settings, answers=True)`, which counts the answering model. The judge is taken from that one config in the calibration suite (including the key variable, the price and the URL as written; the record does not have these), lite-eval checks it item by item before any call and checks the agreement rate of the calibration; the CLI checks the calibration run before opening the store. The output cache is on: the same batch of answers can be judged again with another judge, without regenerating.
  - The generated dataset and suite share `write_generated_dataset`, `golden_cases` and `write_suite` (the `dataset.yaml` of the retrieval eval is byte for byte unchanged, so the dataset hash is still comparable with earlier runs; Codex found that it was once almost changed).
  - **Self-evaluation bias**: the answer and the judge are both qwen3:8b. A judge of another model family (through llm-gateway, as the owner instructed) is the next control.
- S14 implementation notes (2026-11-18, whether the answer is correct, and the effect of auto-routing; that item of lite-model design §15):
  - **Why**: lite-model's routing (the `smart-router` of the LiteLLM Proxy) needs to measure "is auto-routing worth it". The grounded judge only judges "whether there is
    grounding", and it misses the most common errors of real answers; so "did it answer correctly" (`key_fact`) was added first, and then the model that actually answered was recorded (`response_model`).
  - **`key_fact`**: one regular expression for each of the 32 questions. The first version was too wide (found by an independent review: numbers had no boundary, `16` matched "7 gram … 16 grams",
    `不予受理` (not accepted) carried no number of days, and the `not|no` of q08 matched almost any English answer); after tightening, counterexample tests were added (a changed number or another question's answer must not match;
    for q12 the changed number is the tilt angle while the question asks which planet, the answer is still correct, listed as an exception). An answer that states the opposite direction cannot be seen by the regex, and is left to the judge.
  - **The cache defect of `eval-answers` (since S13)**: the suite did not write `cache`, and lite-eval's default `auto` reused the last output, so the second
    run hit all 32 and the latency was the last one's too. The second run of this experiment was therefore voided and re-run; fixed to `cache: off`, with the regression test red first.
  - **`response_model`**: it was first called `answered`; an independent review pointed out that it collided with lite-model's `Provenance.answered` and `status:
    "answered"`, and that with a plain LiteLLM alias it is only the alias, so it was renamed and written down as a transitional field (§6).
  - **Results** (32 questions × 2 real runs, temperature 0; "correct" = `key_fact` and grounded; judge qwen3:8b thinking on, calibration agreement 0.982):

    | Config | Correct (two runs / mean) | grounded mean | Latency p50 |
    | --- | --- | --- | --- |
    | Direct Ollama qwen3:8b | 0.938 / 0.938 / 0.938 | 0.969 | 12.0 s |
    | Proxy workhorse (qwen3:8b) | 0.938 / 0.938 / 0.938 | 0.969 | 11.9 s |
    | Proxy small (qwen3:1.7b) | 0.719 / 0.781 / 0.750 | 0.781 | 3.8 s |
    | Proxy smart-router | 0.875 / 0.812 / 0.844 | 0.891 | 5.2 s |

  - **The route splits by language, not by difficulty**: in both runs all 16 Chinese questions went to 1.7b, and the 10 questions that went to 8b were all English (the heuristic classifier looks at English technical words,
    code and reasoning words, and in its eyes Chinese prompts are all "simple"). The 10 questions that picked 8b were all correct (2 of which 1.7b alone would get wrong); the questions that picked 1.7b got 2–3 fewer correct than all-8b,
    and the losses are all in Chinese questions. In the second run, on q31, 1.7b used all 1024 output tokens on thinking, and the answer was empty.
  - **Every answer that did not pass was read one by one**: those of 8b are all slight additions (such as "磨细会提高萃取、降低酸度" (grinding finer raises extraction and lowers acidity)); those of 1.7b have real errors (saying the awakening of insects is the "second" solar term,
    one extra `*` in a command, saying used napkins are recyclable, saying there is none while the source has it, and in the second run saying "磨粗" (grind coarser) for "磨细" (grind finer)). `key_fact` alone would let these through,
    so only "correct" is reported. On q12 every config answers "资料没有" (not in the materials): the retrieval did not find it, and it is not a problem of the model.
  - **Going through the Proxy does not change quality**: direct and workhorse are the same in both runs.
  - **Limits**: 32 single-document fact questions; the judge is also qwen3:8b; the route and the judge share one Ollama, and the model switching load time is counted into the route's latency.
  - During the experiment another session raised lite-eval to 0.4.1 (`3eec810`): it only added a "warn when target errors are the majority" and one line in the report, judging and the judge are unchanged, and the two runs are comparable.
  - **Close-out review**: `/simplify` (4 angles): switched to lite-eval's `parse_scorer_id` / `spec_name`, the legacy `Question` carries
    `key_fact` directly, the golden test goes through `load_golden`, `key_fact` is all or nothing, and the report says so when not checked; skipped: `_send` returning `Completion`, and
    merging the four configs (`load_settings` has no include). Codex was unavailable (quota used up, until 2026-12-03), and an independent Claude agent did the same
    read-only review: `key_fact` too wide (high, fixed, see above), `answered` collides and was not written into the design (medium, renamed and written into §6), the two paths differ in the
    `max_tokens` parameter name and in the thinking default (medium, **not changed**: lite-model S0 measured that LiteLLM 1.108.9 through `ollama_chat/` honours both parameter names,
    and when `reasoning_effort` is not set both sides think and the thinking does not enter the answer body, and the output tokens of the two experiments are close too), route latency includes model switching (low, noted as a limit).

**S18 (2026-11-19) Connect lite-cache L1.**
- The `cache` extra (lite-cache v0.1.0 release wheel) and the `cache` config (`enabled` off by default, `roles`, `ttl_s`, `schema`, `question_log`,
  `prompt_secret_env`); PG only, using lite-rag's own `database_url`; `cache.schema` cannot be the same as `store.schema`. The cache does not enter the index identity.
- The runtime's lite-model registers lite-cache's `CacheHook` (`RagRuntime(model=…)` can be injected, tests use a recording transport); the calls of `contextual`, `graph` and
  `answer` are each inside `cache_context` (scope `literag/<namespace>`, and a role not in `roles` is still not cached); lite-cache's errors are turned into
  lite-rag's `StoreError` (503), and when a read fails no request is sent.
- **Finding**: the source block nonce of §6 is random for every prompt, so requests never repeat and L1 cannot hit even once (writes succeed, hits stay 0). The owner chose plan A: cached calls are
  rendered inside `keyed_nonces`, with nonce = the first 12 hexadecimal digits of `HMAC-SHA256(secret, the prompt rendered with a placeholder tag)`; the secret comes from `LITERAG_PROMPT_SECRET`
  (≥32 characters and at least 12 distinct characters, otherwise `ConfigError`; Codex pointed out that length does not prove randomness, so obviously repetitive values are blocked, and the README says to generate it with `openssl rand -hex 32`). Without the secret the nonce cannot be computed, and the strength is the same as a random nonce (48 bits); when the cache is off or the role is not cached it is still a random nonce.
- **Finding**: lite-model 0.2.1's `Bounded.map` uses a plain thread pool, whose worker threads do not inherit the contextvar, so `call_context` does not reach the hook, and when `contextual` concurrency is > 1
  the cache fails silently (with concurrency 4 the scopes the hook sees are all None). `BoundedLLM.map` works around it with `copy_context()` per item; lite-model 0.3.0 plans to fix the root.
  2026-11-20: raised to lite-model 0.3.0 / lite-cache 0.2.0, lite-model's own `Bounded.map` now carries the caller's context, and the workaround was deleted;
  `test_contextual_concurrency_four_still_hits` passes unchanged and serves as the regression test.
- `Completion` / `Answer` carry `cached`; the CLI marks `(cached answer …)`, the HTTP answer gets one more `cached` field (additive only); `ask()` has an optional question log.
- Checks: 556 unit + integration tests, 9 under `-m ollama`; mutations: removing the keyed nonce turns 4 cache tests red, removing `copy_context` turns the concurrency test red (the test text at first was cut into only 1 chunk,
  `Bounded.map` ran serially, and this mutation was not caught; after changing it to 4 sections it was caught), removing `Answer.cached` turns 3 tests red. The real round trip of `-m ollama` was not run at first: the local
  Ollama was not up at the time, and was then started with `brew services run ollama` and run afterwards: all 9 of `-m ollama` pass, including the newly added real round trip (production path
  `RagEngine.from_settings`, qwen3:8b), where on the second ask `cached` is true and hits 1 stored 1; it turns red when the hook registration is removed.
- **Close-out review**: `/simplify` (4 angles) fixed: `cache.schema` rejects `public` / `information_schema` / `pg_*` (shares
  `sql_identifier` with the store, red first); when the cache is on, `from_settings` already checks the extra and the secret (like the reranker, red first); `close()` removes the hook from the injected model
  (red first); with a secret, `render_pair` renders only once; the extra import and the error conversion of `cache.py` are each in one place only; the tests share `tests/chat.py`. Not changed: storing the secret in
  `CacheBinding` (on the first entry into `cache_context` the model and the binding have not been built yet), and `prompt_key` using `extras.api_key` (the error says API key).
  Codex 1 finding (see above, the strength of the secret). The independent whole-branch review ("mergeable once fixed"): 2 important — **the question text is not escaped the way documents are, and with a secret the fixed
  placeholder could be impersonated as a real tag by the same string in the question**, changed to use an unpredictable random placeholder at every render and normalise it into a fixed marker before computing the HMAC (red first: the placeholder
  in the question produced one more real tag); when the hook name is duplicated on an injected model, lite-model's error is turned into lite-rag's `ConfigError` (red first). Minor: the reindex test also compares
  the graph, a test is added for a cache read failure during ingest, the tests close the runtime, use the public `hooks[name]`, the CLI writes `saved` and not `tokens` for a cached answer (red first),
  and the question log gets a lock (untested: the interleaving is hard to reproduce stably). Changing `LITERAG_PROMPT_SECRET` is equivalent to invalidating the whole cache (the README says so).
