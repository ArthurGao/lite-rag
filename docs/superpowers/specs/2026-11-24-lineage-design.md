# lite-rag S25: Lineage (the lineage ledger) design

> Status: draft, awaiting owner confirmation (2026-11-24). Replaces the parts of `docs/LINEAGE_GUIDE.md` that conflict with this document: the guide is the initial intent, this document is the final direction after checking the existing code.
> When implementing, also update `docs/lite-rag-design.md` §4 (new tables), §9 (configuration) and §12 (S25 stage and acceptance).

## 1. Intent

The owner wants lite-rag to be a general RAG module reused across projects, with the **answer lineage** that a mature RAG should have:

- For any indexed chunk, we can find which **version** of which document it came from.
- For any document, we can find all the artifacts derived from it (impact analysis).
- For any question and answer, we can find which chunks were used (retrieved / put into the prompt / cited), whether those chunks **were already stale at that time**, and the original text of the stale version as it was then.

The typical use: a user complains "答案说签证费 100 元，实际是 120 元" (the answer says the visa fee is 100 yuan, but it is actually 120). The trace shows whether the document said 100 at the time and was changed later, or whether retrieval or the model is at fault.

The layer that records "who derived from whom" is **general** (it knows nothing about RAG) and will be reused later by TenderTED, Cairn, AdForge and other projects. The RAG-specific parts (how chunks are recorded, query traces) live in lite-rag.

## 2. Goals / Non-goals / Anti-goals

**Goals**
1. A general ledger package `lite_rag.lineage`: it records artifacts, versions and derivation edges; it looks up upstream, downstream and the version that was valid at a given time; it cleans up stale original text.
2. lite-rag's write paths (upsert / delete / delete_kb / reindex) record in the ledger **in the same transaction**.
3. Query trace: `search` / `ask` can optionally record the chunk versions used in four stages (retrieved / reranked / in_context / cited).
4. Old chunk versions keep their original text; by default the text is removed after 90 days, while the hash and the metadata row are kept.

**Non-goals**
- The ledger does not drive computation (no `run_step(fn)` from guide §4, no fingerprint-based skipping). lite-rag already has: skip when the document hash is unchanged (`engine.py` `upsert_text`), `embedding_cache_<dim>` reuses vectors by `(model_id, key)`, lite-cache L1 reuses the contextual / graph LLM calls, and PG foreign keys cascade-delete vectors.
- It does not change rule 3 (a new embedding model = a new kb + `reindex`) or rule 4 (chunk id `"{doc_id}#{n}"` is stable, upsert replaces the whole document). The retrieval tables hold only current data; the ledger is a side, append-only record.
- No run / step tables (`lineage_steps` / `lineage_runs` in guide §3). How an artifact was produced (chunker config, model) is recorded in the artifact's `producer` and `metadata`.
- No visualization UI, no OpenLineage integration, no trace aggregate statistics.
- No split into a separate repo (trigger conditions in §8).

**Anti-goals** (it counts as a failure even if it runs technically)
- Ledger writes change any existing retrieval or ingestion result in lite-rag (eval numbers change, upsert skip logic changes).
- The ledger and the index disagree: a chunk in the index has no currently valid artifact in the ledger, or the reverse.
- `lite_rag.lineage` imports any other `lite_rag.*` module.
- A failed trace write is silently swallowed (violates rule 5, "no silent fallbacks").

## 3. Constraints (from existing code and rules)

- Both storage backends must be supported: PostgreSQL (default, `stores/pg.py`, one transaction per upsert, `_lock_kb` serializes per kb) and Chroma + SQLite (`stores/chroma.py`, SQLite transaction + `pending/ready` status, single process). The ledger tables live in the **same database** (PG: the same schema; Chroma backend: the same `lite_rag.sqlite`), so they can join the same transaction.
- PG migrations are run by `db.migrate(conn, schema)` (`migrations/000N_*.sql`); SQLite migrations are run by `migrate_sqlite` (`migrations_sqlite/`).
- Tests do not call real models; PG integration tests use `LITERAG_TEST_DATABASE_URL` and schema `lite_rag_test`; every new test must be seen red first.
- Every stage goes through the close-out (simplify → Codex read-only review → fix → re-run all checks).

## 4. Option comparison (owner chose A)

| | A. Ledger (chosen) | B. Driver (original guide) | C. RAG built-in lineage only |
|---|---|---|---|
| Approach | The general layer only records and queries; lite-rag's flow is unchanged and records in existing transactions | The general layer's `run_step(fn)` computes a fingerprint to decide whether to compute; ingestion is rebuilt as steps | `documents` gets versions, chunks record their source version, two trace tables |
| Reuse | Yes (the ledger knows nothing about RAG) | Yes, with an incremental engine included | No |
| Cost | Other projects must build their own "skip recompute" | Overlaps with the existing skip / vector cache / lite-cache; ingestion rewrite, high risk | Goes against the "general module" goal |

Why A: it satisfies both "general" and "RAG lineage", does not duplicate existing abilities, and does not touch the behavior built up over 24 stages. Artifacts carry `content_hash`, so when a project without its own skip logic is added later, `run_step` can be built on top of the ledger without redoing the ledger.

## 5. The general layer `lite_rag.lineage`

### 5.1 Concepts

| Concept | Meaning |
|---|---|
| artifact | One **immutable** version of a product: a version of a source document, a version of a chunk, and so on, are all artifacts. `(namespace, kind, key)` identifies the logical identity; `content_hash` identifies the content |
| version | Several artifacts with the same `(namespace, kind, key)`, increasing by `version`; at most one is `active` at any time |
| edge | A `parent → child` derivation relation. Only derivation is recorded (kb membership and query consumption do not go here) |
| validity | `valid_from` / `ended_at`; `status` is `active` / `superseded` / `deleted` |

**Decision: source documents get no table of their own; they are one `kind` of artifact.** Reason: edges connect only artifacts, so the recursive upstream / downstream query has one kind of node; versions, validity and deletion follow the same rules for documents and chunks. The fields of guide §3's `lineage_documents` (`source_uri`, `metadata`) go into the artifact's `key` / `metadata`.

**Decision: every derivation is written as an edge; no `parent_id` is stored on the artifact.** Reason: in lite-rag an embedding is not an artifact by itself (see 6.1), and the remaining document → chunk relation is one-to-many with one edge per chunk, so the one-to-one bloat the guide worried about does not occur.

### 5.2 Tables (PG syntax; SQLite stores JSON and times as TEXT, same semantics)

```sql
CREATE TABLE lineage_artifacts (
  id           TEXT PRIMARY KEY,           -- uuid4 string
  namespace    TEXT NOT NULL,              -- isolation key given by the caller; lite-rag uses the kb physical key "<ns>:<kb>"
  kind         TEXT NOT NULL,              -- not interpreted by the general layer: "document", "chunk", ...
  key          TEXT NOT NULL,              -- logical identity: doc_id, chunk_id, ...
  version      INT  NOT NULL,              -- counts up from 1 within the same (namespace, kind, key)
  content_hash TEXT NOT NULL,
  producer     TEXT,                       -- how it was produced, e.g. "chunker:markdown@<digest>"
  payload      TEXT,                       -- original text, may be NULL; set to NULL after gc
  metadata     JSONB NOT NULL DEFAULT '{}',
  status       TEXT NOT NULL DEFAULT 'active',   -- active / superseded / deleted
  valid_from   TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at     TIMESTAMPTZ,                -- the time it became superseded or deleted
  UNIQUE (namespace, kind, key, version)
);
CREATE UNIQUE INDEX ON lineage_artifacts (namespace, kind, key) WHERE status = 'active';
CREATE INDEX ON lineage_artifacts (namespace, status);

CREATE TABLE lineage_edges (
  parent_id TEXT NOT NULL REFERENCES lineage_artifacts (id) ON DELETE CASCADE,
  child_id  TEXT NOT NULL REFERENCES lineage_artifacts (id) ON DELETE CASCADE,
  metadata  JSONB NOT NULL DEFAULT '{}',   -- position of the child inside this parent, etc. (e.g. chunk_id, offset)
  PRIMARY KEY (parent_id, child_id)
);
CREATE INDEX ON lineage_edges (child_id);
```

- `ended_at` replaces the guide's `superseded_at`, because deletion uses it too.
- Edges carry `metadata`: the same artifact can appear in different versions and at different positions of a parent (see 6.1). Position information belongs to "this derivation relation", not to the immutable artifact itself.
- The migration files live inside the package, in `lite_rag/lineage/migrations/` (PG) and `migrations_sqlite/`, with their own version table `lineage_schema_migrations`, run by the general layer's `migrate(conn)`. **lite-rag side (the implementer's decision, fixed after review)**: `db.migrate`, `migrate_store` and the automatic migration when a Chroma store opens all call it after lite-rag's own migrations; the check in `PgStore` that "refuses to open when migrations are pending" also covers the ledger migrations. Reason: llm-steward's contract tests call `lite_rag.db.migrate` directly and its in-process provider builds its own store, so attaching it only to `literag migrate` would make them fail only at the first upsert. This way `lineage.enabled: true` is safe as a default for library users too, and llm-steward needs no code change when its pin is upgraded.

### 5.3 API

The general layer takes the caller's connection, works inside the caller's transaction, and never opens or commits a transaction itself. There are two implementations, `PgLedger(conn)` (psycopg) and `SqliteLedger(conn)` (sqlite3), both satisfying the same `Ledger` Protocol.

```python
record(namespace, kind, key, content_hash, *, parents=(), producer=None,
       payload=None, metadata=None, edge_metadata=None) -> Artifact
    # an active version with the same key and the same content_hash exists → reuse it, add the edges to parents, return it
    # otherwise create version+1, mark the old active version superseded (ended_at=now), write the edges, return the new one

retire(artifact_id, *, status="superseded"|"deleted") -> list[Artifact]
    # end this artifact, and cascade to end those of its descendants that "no longer have any active parent";
    # return the list of ended artifacts (including itself)

active(namespace, kind, key) -> Artifact | None
as_of(namespace, kind, key, at: datetime) -> Artifact | None   # the version valid at a given time
get(artifact_ids) -> dict[str, Artifact]
upstream(artifact_id) -> list[Artifact]      # recursive upwards; the edge information is enough to rebuild the path
downstream(artifact_id) -> list[Artifact]    # recursive downwards (impact analysis)
retire_namespace(namespace, status="deleted") -> int
gc(ended_before: datetime) -> int            # artifacts that have ended and whose ended_at is earlier than that time: set payload to NULL
migrate(conn) -> list[str]
```

**Core rule: "an orphan ends".** An artifact ends only after all of its parents have ended. This rule makes "when a document gets a new version, unchanged chunks stay valid" hold naturally (see 6.2), and it also makes deletion cascade.

### 5.4 Boundary

- `lite_rag/lineage/` depends only on the standard library, psycopg (the PG implementation) and sqlite3. A test walks the imports in its source and fails on any `lite_rag.` import (its own package excepted).
- It does not read lite-rag's settings; the table name prefix is fixed to `lineage_`, and the PG schema is decided by the `search_path` of the caller's connection.

## 6. RAG layer (inside lite-rag)

### 6.1 What is recorded

| kind | key | content_hash | producer | payload | Parents |
|---|---|---|---|---|---|
| `document` | `doc_id` | lite-rag's existing `content_hash(title, text, metadata, pages)` | — | Not stored (the full text is already in `documents.text`; the original text of old versions is covered by the chunk text) | None |
| `chunk` | `"{doc_id}/{text_hash}/{k}"` | `text_hash` = sha256(chunk original text `Chunk.text`) | The digest of the kb's chunker identity | The chunk original text | Every document version that contains it |

- **A chunk's identity is by content, not by position (the implementer's decision, fixed after review).** The position `doc#n` and the offsets move as a whole when earlier paragraphs are added or removed; when contextual is on, each chunk's context is generated from the whole document, so any edit changes it. If these were part of the identity, changing one character would give every later chunk a new version, and "only that paragraph is stale" would not hold. So the identity depends only on the chunk original text:
  - `k` is the index of this occurrence among identical original texts in the same document (counting from 0), used to tell repeated boilerplate paragraphs apart;
  - Position information (`chunk_id`, `n`, `start`, `end`, `page`, `parent_n`) and the contextual context of this version are written in the **edge's metadata**: the same chunk is `doc#4` in v1 and `doc#5` in v2, and the two edges each record their own;
  - Meaning: "stale" = this original text no longer exists in the current document version. If the original text is unchanged and only its position or context changed, it is not stale.
  - Cost: a chunk's embedding input contains the context and the title, so the ledger cannot restore "the complete text that was indexed at that time"; it can only restore the original text and the context of that time (on the edge). This fits the purpose of lineage.
- namespace = the kb physical key (`<namespace>:<kb>`), so same-named documents in different kbs do not interfere; a `reindex` into a new kb is a separate, independent set of artifacts.
- **An embedding is not an artifact by itself**: a kb is bound to one model only (rule 3), and vectors correspond one-to-one to chunks. The chunk's `metadata` records `embedding` (`model_id`) and `dim`; the contextual / graph models are also recorded in metadata (if enabled). `upstream(chunk)` then answers "which version of which document did this vector come from".
- The parent chunk of parent_child: not recorded separately in this phase (parent chunks are only used for display and do not take part in indexing); the chunk metadata records `parent_n`.

### 6.2 Write paths (all inside the store's existing transaction)

- **upsert (document changed)**: `record(document)` → new version v(n+1), the old version is superseded; for each newly cut chunk, `record(chunk, parents=[new document version], edge_metadata=position and context)`. A chunk with the same original text reuses the old artifact and gets one more edge, so **when only paragraph 5 is changed, only the chunks whose original text changed get a new version**, and the shift of numbers and offsets in the paragraphs before and after has no effect. Finally `retire(old document version)`: by "an orphan ends", only the old chunks that the new version no longer references are superseded.
- Note: when a chunk's original text spans the modified spot (the chunk boundary moves with the edit), it also gets a new version. This is a real content change, not a false alarm.
- **upsert (document unchanged)**: lite-rag already skips it before entering the transaction; the ledger is untouched.
- **delete**: `retire(document active version, status="deleted")`, cascading to its chunks.
- **delete_kb**: `retire_namespace(kb physical key)`.
- **Chroma backend**: the ledger is written in the SQLite transaction (the `pending` stage); when writing to Chroma fails, the existing logic deletes the document, and the ledger records it as `deleted` accordingly. Principle: the ledger reflects the store's final state.
- **The Store Protocol signature does not change**: the store implementations can already get the kb information (chunker, model), and the store assembles the producer / metadata needed for recording inside the transaction.

### 6.3 Query trace

```sql
CREATE TABLE rag_query_traces (
  id            TEXT PRIMARY KEY,
  namespace     TEXT NOT NULL,           -- the engine's namespace
  kbs           JSONB NOT NULL,          -- kb names of this query
  operation     TEXT NOT NULL,           -- 'search' | 'ask'
  query         TEXT,                    -- by configuration: original text / sha256 / NULL
  retrieval     JSONB NOT NULL,          -- mode, top_k, rerank, where, route choice
  model_id      TEXT,                    -- answering model (ask), from lite-model provenance
  usage         JSONB,
  cached        BOOLEAN,
  status        TEXT,                    -- Answer.status
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE rag_query_sources (
  trace_id     TEXT NOT NULL REFERENCES rag_query_traces (id) ON DELETE CASCADE,
  artifact_ns  TEXT NOT NULL,            -- kb physical key
  artifact_key TEXT NOT NULL,            -- (implementation: may be NULL, see §12) the chunk's ledger key "{doc_id}/{text_hash}/{k}"
  chunk_id     TEXT NOT NULL,            -- position at retrieval time, for readability
  stage        TEXT NOT NULL,            -- retrieved / reranked / in_context / cited
  rank         INT,
  score        DOUBLE PRECISION,
  PRIMARY KEY (trace_id, artifact_ns, chunk_id, stage)   -- chunk_id is unique only within a kb
);
```

- Stage definitions: `retrieved` = the chunks that enter the candidate pool after fusion (`pool` of them); `reranked` = the order after rerank (only when rerank is enabled); `in_context` = the hits finally returned / put into the prompt; `cited` = the citations kept after `check_citations`. Under parent_child, what is recorded is the **matched child chunk**.
- **Pin the version at the moment of retrieval (the implementer's decision, fixed after review)**: the trace does not look up "the currently active artifact" at write time. It computes the ledger key directly from the chunk original text read at retrieval time (`StoredChunk.text`). So if the document is changed between retrieval and the trace write (during the model call) in `ask`, the trace still records the text that was actually read at that time. For this, the `chunks` table gets a column `lineage_key`, computed and written at upsert time (independent of the ledger switch, since the key depends only on the document id and the original text); retrieval reads it together with the chunk, and the trace uses it directly, with no need to work out `k` again.
- Therefore `rag_query_sources` has no foreign key to `lineage_artifacts`: a trace can be written normally even when the key cannot be found in the ledger (the kb has not been backfilled yet, or it was written while recording was off), and `trace show` displays it as "未记录" (not recorded).
- Stale judgment (computed at `trace show` time): take the artifact that was valid for that key at `trace.created_at` (`as_of`; the implementation uses `started_by` instead, see §12). If it ended after that → "当时有效、之后过期" (valid then, stale since), and show its original text (if not removed by gc); if still active → "仍然有效" (still valid); if not found → "未记录" (not recorded). This is a mark and raises no error.
- `Answer` gets a `trace_id` (None when tracing is not enabled); the HTTP `ask` response carries it.
- **Trace of search (owner decision, 2026-11-24)**: the existing `search` / `search_kbs` signatures and behavior do not change and record no trace. New additions:
  - `search_traced(...)` / `search_kbs_traced(...)`: same parameters as the original methods, returning `Traced(hits: list[Hit], trace_id: str | None)`; they record the three stages `retrieved` / `reranked` / `in_context`, with `operation='search'`.
  - `record_cited(trace_id, chunk_ids)` (implementation: takes Hits, see §12): a host that generates its own answers (llm-steward only calls `search_kbs`) uses it to add the finally cited chunks as the `cited` stage; the chunk_id must have appeared in that trace's `in_context`, otherwise it raises `InvalidRequest`.
  - Reason: llm-steward consumes `list[Hit]` and the return type cannot change; `ask` is changed internally to call the traced version.
  - HTTP gets no traced-search endpoint in this phase; llm-steward uses the in-process call.
- A failed trace write raises an error (rule 5). The trace is written in a separate small transaction after the retrieval read.
- Latency is not recorded: lite-rag's existing code does not measure time, and adding it is a separate matter.

### 6.4 Configuration

```yaml
lineage:
  enabled: true            # record in the ledger; when off, write paths do not record and the query API raises an error
  keep_text_days: 90       # how long gc waits before clearing the original text of ended artifacts
trace:
  enabled: false           # off by default: it stores the question text
  query: text              # text / hash / none
  retention_days: 90       # gc deletes older traces
```

`literag lineage gc` handles two things at once: it clears original text (`keep_text_days`) and deletes old traces (`retention_days`). It does not run on a schedule by itself; the user schedules it. An artifact row referenced by a trace is never deleted; only its `payload` is cleared.

### 6.5 Backfill of existing data

The documents of existing kbs have no record in the ledger, and an upsert is skipped when a document is unchanged, so they are never added later (this includes all existing kbs of llm-steward). A missing backfill does not make any feature raise an error: traces are written as usual and the related sources show "未记录" (not recorded); recording starts naturally the next time a document changes. To fill in the history, `literag lineage backfill <kb>` is provided: from the existing rows of `documents` and `chunks` it generates v1 document and chunk artifacts (a chunk's context is already in the `chunks.context` column, so the indexed text can be restored). For documents that already have a ledger, it is a no-op. Old documents with `documents.text` NULL can still be backfilled (the chunk original text is in the `chunks` table).

### 6.6 CLI / HTTP

- CLI: `literag lineage show <kb> <doc_id>` (the versions and their chunks), `literag lineage upstream <artifact_id>`, `literag lineage downstream <artifact_id>`, `literag trace show <trace_id>` (the chunks of each stage, whether they are stale, the original text of stale versions), `literag lineage backfill <kb>`, `literag lineage gc`.
- HTTP: in this phase only the `ask` response returns `trace_id`; the search endpoints do not change; an HTTP interface for querying the ledger is not built (non-goal, to be added later as needed).

## 7. Stages

Every stage ends with the close-out, all existing tests stay green, and the numbers of `literag eval` do not change (anti-goal 1).

1. **S25a General ledger**: `lite_rag/lineage/`, the two sets of migrations, `PgLedger` / `SqliteLedger`, the import boundary test. Contract tests run the same cases against both implementations.
2. **S25b Hook into the write paths**: ledger recording for upsert / delete / delete_kb in the PG and Chroma stores; `literag migrate` calls the ledger migrations; the backfill command.
3. **S25c Query trace**: the two tables, `search_traced` / `search_kbs_traced` / `record_cited`, instrumentation of `ask`, `trace_id`, `trace show`.
4. **S25d gc and wrap-up**: `lineage gc`, `lineage show/upstream/downstream`, the README says how a third party uses `lite_rag.lineage` on its own, update design §4 / §9 / §12.

## 8. Conditions for splitting into a separate repo

Split when any one of these appears: TenderTED / Cairn / AdForge or similar want to use the ledger; lite-mem or llm-steward want to record their own derivation relations. The precondition for splitting is already guaranteed by 5.4, and splitting = move the directory + lite-rag changes its imports + publish a wheel + change the pin.

## 9. Acceptance (covered by automated tests)

- Ledger contract (run once each on PG and SQLite): `record` reuses on identical content, and gives a new version and supersedes the old one on different content; "an orphan ends"; `as_of` returns the version valid at the given time; `upstream` / `downstream` are correct; `gc` clears only the payload of ended and expired rows, and the rows stay.
- One paragraph of a document is changed (including a change that shifts the numbers and offsets of later chunks, and the case with contextual on): only the chunks whose original text changed get a new version, and the other chunks are still the same artifact, with two parent edges (v1, v2); the `chunk_id` / offset on the two edges are each the value of that time.
- A document has two paragraphs with identical original text: they are two artifacts (`k`=0, 1) and do not overwrite each other.
- Delete a document: the document and all its chunk artifacts are `deleted`; after delete_kb there is no active artifact under that kb namespace.
- Consistency: for a document that has been recorded (recording was on at write time, or it has been backfilled), after any sequence of upserts / deletes, every chunk in the store corresponds to exactly one edge from the document's active version, pointing to one active chunk artifact, and there is no extra active artifact (tested on both the PG and Chroma backends).
- The document is modified during ask (changed after retrieval and before the trace write): the trace records the key of the old original text read at retrieval time, and `trace show` marks it "当时有效、之后过期" (valid then, stale since).
- Trace enabled on a kb that has not been backfilled: `ask` succeeds, and the sources show "未记录" (not recorded).
- `search` / `search_kbs` write no trace rows even when trace is enabled, and their return values do not change; `search_kbs_traced` + `record_cited` record the four stages; `record_cited` raises an error and writes no rows when given a chunk_id that is not in `in_context`.
- llm-steward's way (calling `lite_rag.db.migrate` directly and then building a store) also creates the ledger tables.
- Chroma write failure: the document is `deleted` in the ledger, consistent with the store.
- Trace: the four stages can be told apart; when a document is modified after ask, `trace show` marks the cited chunk as "当时有效、之后过期" (valid then, stale since) and shows the old original text; with `trace.enabled=false` no rows are written.
- Backfill: after backfilling an existing kb, the consistency check holds; backfilling again is a no-op.
- Boundary: `lite_rag.lineage` does not import other lite-rag modules.
- Behavior unchanged: all existing tests are green; `-m ollama` is not required, but the numbers of the eval config run with `FakeEmbedder` match those before the change.

## 10. Left to the implementer

- How `PgLedger` / `SqliteLedger` share SQL (parameter-style adaptation or two copies of the SQL).
- How the recursive queries are written, and the depth guard.
- CLI output format.
- Whether the consistency test uses random operation sequences (property-based) or fixed scenarios.
- Switching llm-steward to `search_kbs_traced` + `record_cited` is llm-steward's own follow-up work (after the pin upgrade) and is outside S25.

## 12. Decisions made during implementation (2026-11-25, S25a–d)

These changed or added to the text above; this section takes precedence:

- **Ledger API**: the entry points are `migrate(conn)` / `pending(conn)` / `ledger_for(conn)`, which choose PG or SQLite by connection type (not the four functions such as `migrate_pg`). Also added: `record_many` (checks the parents once for a group of chunks), `count_active` (consistency check), `history` (all versions of one key), `started_by` (the latest version that had started at a given time, whether or not it has ended). The PG / SQLite differences are concentrated in `lineage.dialect`, shared by the ledger and lite-rag's trace tables.
- **Concurrency**: the caller must serialize writes to the same namespace (lite-rag: PG holds the kb row lock, Chroma uses a process lock). The ledger itself takes no lock.
- **The ledger does not record the contextual context**: it is text the model wrote from the content, and storing it on an edge would outlive its deletion (final review #1). Only positions are recorded on the edge (`chunk_id`, `n`, `start`, `end`, `page`, `parent_n`).
- **Deleting clears the original text**: a `deleted` artifact has its `payload` and `metadata` (the document's title / source) cleared at once (security review, final review #1); a `superseded` one is kept until `keep_text_days` and then cleared by gc. `trace show` displays the chunks of a deleted document as "过期" (stale) but without original text.
- **The document artifact's metadata** records `title`, `source`, `embedding`, `dim`; the chunk artifact's metadata is empty (chunks of the same kb are all the same, so it is not repeated).
- **backfill repairs deletions made while the ledger was off**: a document that is still active in the ledger but no longer in the store is recorded as `deleted` (final review #2). A kb that was dropped entirely while the ledger was off cannot be repaired (the kb is gone from the store, and backfill reports `KbNotFound`): its artifacts stay as they are.
- **`chunks.lineage_key`**: computed at write time (independent of the ledger switch) and read at retrieval time; the trace uses it directly and does not work out `k`.
- **Trace storage**: `artifact_key` may be NULL (not recorded); `in_context` is not stored, and at read time it is taken as the first `top_k` of the last ordering stage (reranked if there is a rerank, otherwise retrieved). The Store protocol has only one `traces()`, returning a `TraceLog` inside one transaction; when reading traces, the SQL filters by namespace.
- **Trace time**: taken right after the chunks are read (not after the model answers). `trace_report` uses `started_by(key, time)` to find the version read at that time: ended means "过期" (stale), still valid means "有效" (valid), not found means "未记录" (not recorded).
- **`record_cited(trace_id, cited: Sequence[Hit])`**: takes Hits rather than chunk_ids, because chunk_id is unique only within a kb, and a cross-kb trace needs the kb name.
- **gc**: `literag lineage gc` does not split by namespace and acts on the whole store; clearing original text and deleting traces both work in batches of 1000 rows, one transaction per batch (the partial index `lineage_artifacts_gc` contains only ended rows that still have original text). With `lineage.enabled: false` it only deletes traces and does not touch the ledger. It does not prune the edges "from a stable chunk to an old document version": those are the lineage itself, and the growth is one edge per chunk per edit.
- **Storage estimate**: each trace is about 2×`top_k` rows (rerank off, the default) or 2×`top_n`+`top_k` rows (rerank on). At 10,000 queries a day, kept for 90 days, with rerank on, PG needs about 5–10 GB.
- **Additions before merge (2026-11-25)**: `record_cited` accepts each trace only once: it conditionally updates the trace's status from empty (an unfinished search trace) to `answered` and writes the cited rows in the same transaction, so of concurrent calls only one succeeds; the trace of an `ask` (including those with no citations, `no_sources`, `failed`) is always rejected, and passing an empty list also uses up that one time. `engine.purge_traces()` / `literag trace purge --yes` deletes all traces of one namespace in batches (traces only; delete a kb with `drop-kb`), with the index `ix_rag_query_traces_namespace (namespace, created_at, id)` (migration 0009 / sqlite 0004); when the model call fails, `ask` also writes a trace (`status: failed`, including the retrieval stages), then raises the model's error as usual (if writing the trace itself fails, it is attached as a note to that error and does not replace it); `record()` also goes through `record_many` (so there is only one write path): one query fetches the latest versions of all keys, superseded ones are ended in batches, and new versions and edges each use one `executemany` (300 chunks drop from about 900 calls to about 7); edges whose content did not change are not rewritten.
- **Clock**: the version time comes from the writer's clock and the trace time from the reader's. Hosts that share one store need synchronized clocks (NTP), otherwise a valid chunk may be judged stale. The database time is not used instead, because that would cost one more query per recording.
- **Not handled (unreachable)**: when the document hash is unchanged but the chunk texts cut from it differ, the old chunks do not end. This path cannot be reached now (the kb's chunker is fixed and the hash covers the full text); handling it would need a "cut the edge" interface in the ledger.
