> **Superseded (2026-11-25):** the decided design is `docs/superpowers/specs/2026-11-24-lineage-design.md`, built in S25 (design §12). This guide is kept as the original intent.

# Lite Lineage implementation guide (for Claude Code)

> Audience: Claude Code working in the lite-rag repository.
> Usage: **first analyze the existing code and submit an implementation plan for my confirmation, then start work.** This document gives design intent and constraints; the actual implementation must follow the current state of the repository.

---

## 0. Background and goals

lite-rag is to become a general RAG module reused across projects. It currently lacks two abilities:

1. **Incremental reprocessing**: when a source document, the chunk strategy or the embedding model changes, recompute only the affected parts, with no full rebuild.
2. **Traceability (Lineage / Provenance)**: for any artifact (chunk, embedding, summary, generated page), the upstream source can be found; for any source change, the downstream impact can be found.

In addition, the query side must be able to answer: which chunks and which document versions this answer used, and whether this data was already stale at that time.

**Non-goals**: no enterprise-grade lineage platform like OpenLineage/Marquez; no visualization UI; no distributed scheduling.

---

## 1. Layering (hard constraint)

```
lite-lineage  (general layer, knows nothing about RAG)
    ▲
    │ depends on (one-way)
lite-rag      (RAG layer: chunk/embed steps, KB, query trace)
```

- Put the general layer in a separate package inside the lite-rag repository, for example `packages/lineage/` or `lite_rag/lineage/`, following the existing repository structure.
- **The general layer must not import any RAG code.** `artifact.kind` is just a string (`"chunk"`, `"embedding"`, `"summary"`, ...), and the general layer does not interpret it.
- It will later be reused by other projects (TenderTED, Cairn visa-document checking, the AI learning system, AdForge), so design the interface as a "general derived-data pipeline". Split it into a separate repo only after a second project has adopted it.

---

## 2. Core concepts

| Concept | Meaning |
|---|---|
| document | The raw input (file, URL, DB record); versions are told apart by `content_hash` |
| step | The **definition** of a processing step: name + version + config_hash (such as `chunk@v2`, `embed:bge-m3`) |
| processing_run | One **execution** of a step: status, time, error |
| artifact | The **immutable** product of one execution, with a fingerprint |
| lineage_edge | An artifact-level "input → output" derivation relation |

### Key principles

1. **An artifact is immutable.** When the content changes, generate a new artifact and mark the old one superseded.
2. **The fingerprint drives incremental work**:
   ```
   fingerprint = hash(step.name + step.version + step.config_hash
                      + sorted(content_hash of all input artifacts/documents))
   ```
   Before executing, query by fingerprint: if one exists and is active, reuse it directly (skip); compute only when it does not exist. This is the same idea as the Bazel and Nix caches.
3. **run_id manages the batch, lineage_edges manage individual artifacts.** When one run handles 1000 documents and produces 50,000 chunks, run_id alone cannot tell which chunk came from which document, so edges are required.
4. **lineage_edges record only derivation relations.** The following two relations are **not put into** lineage_edges:
   - KB membership → `kb_members`
   - The consumption relation of an Agent or a query → `rag_query_sources`
5. **Validity period**: an artifact carries `valid_from` and `superseded_at`, which is what makes it possible to judge "whether the data used at the moment of the query was already stale".

---

## 3. Table structure (general layer)

> Postgres is the reference syntax. If lite-rag currently uses SQLite or needs a local mode, keep both working (recursive CTEs are supported by both; JSONB degrades to TEXT JSON in SQLite). First confirm how the repository accesses the DB (ORM or raw SQL) and follow it.

```sql
CREATE TABLE lineage_documents (
  id            UUID PRIMARY KEY,
  namespace     TEXT NOT NULL,          -- tenant/project isolation
  source_uri    TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  version       INT  NOT NULL,
  status        TEXT NOT NULL DEFAULT 'active',  -- active/superseded/deleted
  metadata      JSONB,
  valid_from    TIMESTAMPTZ NOT NULL DEFAULT now(),
  superseded_at TIMESTAMPTZ,
  UNIQUE (namespace, source_uri, version)
);

CREATE TABLE lineage_steps (
  id          UUID PRIMARY KEY,
  name        TEXT NOT NULL,
  version     TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  config      JSONB,
  UNIQUE (name, version, config_hash)
);

CREATE TABLE lineage_runs (
  id          UUID PRIMARY KEY,
  step_id     UUID NOT NULL REFERENCES lineage_steps,
  status      TEXT NOT NULL,            -- running/succeeded/failed
  started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  error       TEXT,
  stats       JSONB                     -- added/skipped/failed counts
);

CREATE TABLE lineage_artifacts (
  id            UUID PRIMARY KEY,
  namespace     TEXT NOT NULL,
  kind          TEXT NOT NULL,
  run_id        UUID REFERENCES lineage_runs,
  content_hash  TEXT NOT NULL,
  fingerprint   TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'active',  -- active/superseded/deleted
  payload_ref   TEXT,                   -- points to a vector ID / object store / text table
  metadata      JSONB,                  -- page, offset, etc.
  valid_from    TIMESTAMPTZ NOT NULL DEFAULT now(),
  superseded_at TIMESTAMPTZ,
  UNIQUE (namespace, fingerprint)
);

CREATE TABLE lineage_edges (
  parent_id   UUID NOT NULL,
  parent_type TEXT NOT NULL,            -- 'document' | 'artifact'
  child_id    UUID NOT NULL REFERENCES lineage_artifacts,
  run_id      UUID REFERENCES lineage_runs,
  PRIMARY KEY (parent_id, child_id)
);
CREATE INDEX ON lineage_edges (child_id);
CREATE INDEX ON lineage_edges (parent_id);
```

**Design trade-offs (evaluate, then decide):**
- Should documents be merged into artifacts (as `kind='source'`)? After merging, edges only need to connect artifacts, which makes recursive queries simpler. Keeping a separate table makes the semantics clearer. Choose one based on the existing code and explain why.
- A one-to-one relation such as chunk → embedding would bloat the edges table. Consider storing `parent_id` directly on the artifact for one-to-one relations, and writing edges only for many-to-many. Estimate the data volume first, then decide.

---

## 4. General layer API (suggested signatures; adjust to the repository's language and style)

```python
register_document(namespace, source_uri, content_hash, metadata) -> Document
    # content_hash unchanged → return the existing version; changed → create a new version, mark the old one superseded

get_or_create_step(name, version, config) -> Step   # computes config_hash internally

compute_fingerprint(step, inputs: list[Document|Artifact]) -> str

should_recompute(namespace, step, inputs) -> tuple[bool, Artifact | None]

run_step(namespace, step, inputs, fn) -> list[Artifact]
    # 1. compute the fingerprint → on a hit, return directly without calling fn
    # 2. on a miss: create a run → call fn(inputs) → write artifacts + edges → end the run
    # 3. the whole process must be idempotent; if fn fails, mark the run failed and leave no half-finished active artifact

upstream_of(artifact_id) -> list[Node]       # recursive CTE upwards
downstream_of(node_id) -> list[Artifact]     # recursive CTE downwards (impact analysis)

mark_deleted(document_id) -> list[Artifact]  # cascade-mark downstream artifacts as deleted, return the list so the upper layer can clean up vectors
supersede(...)                               # mark the old artifacts when a new version takes effect
gc(older_than, keep_metadata=True)           # delete the payload of superseded/deleted artifacts, keep the metadata rows
```

An example of a recursive query for impact analysis:

```sql
WITH RECURSIVE downstream AS (
  SELECT child_id FROM lineage_edges WHERE parent_id = :node_id
  UNION
  SELECT e.child_id FROM lineage_edges e
  JOIN downstream d ON e.parent_id = d.child_id
)
SELECT a.* FROM lineage_artifacts a JOIN downstream d ON a.id = d.child_id;
```

---

## 5. RAG layer (implemented in lite-rag)

### 5.1 Turn the existing ingestion into steps

- Wrap each of load → parse → chunk → embed → write to the vector store in the existing flow as a `run_step` call.
- Every step needs an explicit version and config (chunk_size, overlap, embedding model name, dimension, ...).
- When writing to the vector store, the vector's metadata must carry `artifact_id`. On deletion, delete the vectors according to the artifact list returned by lineage.

### 5.2 RAG-specific tables

```sql
CREATE TABLE kb_members (
  kb_id       UUID NOT NULL,
  document_id UUID NOT NULL,
  added_at    TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (kb_id, document_id)
);

CREATE TABLE rag_query_traces (
  id               UUID PRIMARY KEY,
  namespace        TEXT NOT NULL,
  user_id          TEXT,
  query_text       TEXT,              -- can be configured to store only a hash
  model_name       TEXT,
  prompt_version   TEXT,
  embed_model      TEXT,
  index_version    TEXT,
  retrieval_config JSONB,             -- top_k、filters、reranker
  answer_ref       TEXT,
  external_trace_id TEXT,             -- external trace ID such as Langfuse (optional)
  latency_ms INT, tokens_in INT, tokens_out INT,
  created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE rag_query_sources (
  trace_id    UUID NOT NULL REFERENCES rag_query_traces,
  artifact_id UUID NOT NULL REFERENCES lineage_artifacts,
  stage       TEXT NOT NULL,          -- retrieved/reranked/in_context/cited
  rank        INT,
  score       FLOAT,
  PRIMARY KEY (trace_id, artifact_id, stage)
);
```

- Do **not** store the document version redundantly in sources; look it up through `upstream_of(artifact_id)`.
- To judge whether stale data was used at query time: `artifact.superseded_at IS NOT NULL AND artifact.superseded_at < trace.created_at`.
- Trace records must be **switchable off and have a configurable retention period** (for example, after 90 days keep only aggregate data).
- **The answer of an ordinary Q&A is not written into artifacts.** Only outputs that are persisted and reused (cached answers, generated reports or HTML learning pages) become artifacts and get edges.

---

## 6. Implementation stages

1. **Analysis (do this first, and produce a plan for my confirmation)**
   - The existing ingestion flow, storage (vector store, metadata DB), configuration method, language and dependencies.
   - Whether there is any hash, deduplication or versioning logic now.
   - Give: where the package goes, whether documents are merged into artifacts, how one-to-one edges are handled, and which existing files need to change.
2. **General layer**: tables + migration + API + unit tests.
3. **Hook into ingestion**: turn chunk/embed into steps, and give vectors an artifact_id.
4. **Deletion and GC**: deleting a source cascades to clean up vectors; GC of superseded artifacts.
5. **Query trace**: the two trace tables + instrumentation in the retrieval flow (four stages).
6. **Documentation**: the README states the usage and how third-party projects adopt the general layer.

After each stage, make sure existing features and tests are not affected.

---

## 7. Acceptance criteria (must be covered by automated tests)

- [ ] Ingest the same batch of documents again: on the second run, 0 embedding calls, everything skipped.
- [ ] Modify 1 document: only the chunks/embeddings of that document are recomputed, everything else is skipped.
- [ ] Modify the chunk config (version or config changes): all chunks and their downstream are recomputed; the embedding step is also recomputed even though its own config is unchanged (because the input hash changed).
- [ ] Switch the embedding model: all chunks are reused, only the embeddings are recomputed; the old and new indexes can coexist.
- [ ] Delete 1 document: all downstream artifacts are marked deleted, the matching vectors are removed from the vector store, and retrieval no longer returns them.
- [ ] `upstream_of(embedding_id)` can return the source document and its version.
- [ ] `downstream_of(document_id)` returns the complete impact list.
- [ ] A step execution fails: the run is marked failed, no half-finished active artifact remains, and a rerun can recover.
- [ ] The query trace can tell retrieved/in_context/cited apart, and can judge whether a stale chunk was used at that time.
- [ ] No RAG module appears in the general layer's imports (add a test or lint rule to guarantee this).

---

## 8. References (borrow ideas only, add no dependencies)

- LangChain Indexing API / RecordManager: hash deduplication + the cleanup modes. Note that its incremental mode does not clean up sources that have been deleted from the corpus; we must do that.
- LlamaIndex IngestionPipeline + docstore: hash deduplication and upsert.
- CocoIndex: a declarative incremental engine + end-to-end lineage. If data sources multiply later and real-time sync is needed, consider migrating to it.
- OpenLineage: the conceptual model of dataset / job / run; naming can be borrowed, but do not add it as a dependency.

---

## 9. Questions you (Claude Code) must answer explicitly in the plan

1. Which parts of the existing code can be turned into steps directly, and which need a rewrite?
2. DB choice: Postgres only, or both SQLite and Postgres?
3. Are documents merged into artifacts? How are one-to-one edges stored?
4. Does the vector store support batch deletion by metadata? If not, how is deletion implemented?
5. In the existing retrieval interface, where can the four trace stages be instrumented with the smallest change?
6. Estimate the change scope and risk of each stage.
