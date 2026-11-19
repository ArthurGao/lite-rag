# lite-rag S18 — lite-cache L1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** lite-rag can answer repeated model calls from lite-cache's L1: the index roles (`contextual`, `graph`) when a namespace rebuilds the same chunks, and `ask()` for a repeated question over the same retrieved chunks. Off by default; when off, nothing changes.

**Architecture:** An optional `cache` extra (lite-cache v0.1.0). When `cache.enabled`, the runtime opens a lite-cache store on lite-rag's own PostgreSQL (schema `lite_cache`) and registers a `CacheHook` on its one lite-model; the engine wraps each role's calls in `call_context(scope="literag/<namespace>", cache=Policy(...))` for the roles listed in `cache.roles`. `BoundedLLM.map` runs each item in a copy of the caller's context so the hook sees it in worker threads.

**Tech Stack:** Python 3.12, uv, lite-model 0.2.1, lite-cache 0.1.0 (release wheel), PostgreSQL, pytest, ruff, mypy.

**Spec:** lite-rag `docs/lite-rag-design.md` §12 S18 (acceptance); lite-cache `docs/lite-cache-design.md` §0 (2026-11-19 rows), §3, §9 (adapter contract). Read both.

## Global Constraints

- lite-rag AGENTS.md rules apply: Protocols, namespaces mandatory, index identity, no silent fallbacks, prompts in `.j2`, tests never call a real model, pinned deps, code-facing text English.
- Dependency: extra `cache = ["lite-cache"]`; `[tool.uv.sources] lite-cache = { url = "https://github.com/ArthurGao/lite-cache/releases/download/v0.1.0/lite_cache-0.1.0-py3-none-any.whl" }`; dev group gains `"lite-rag[cache]"` (as it has `lite-rag[eval]`). lite-cache pins lite-model 0.2.1 by URL — the same lite-model lite-rag pins; keep them in step.
- `lite_cache` is imported only in `src/lite_rag/cache.py`, through `lite_rag.extras` (a missing extra is a `ConfigError` naming `lite-rag[cache]`).
- Scope for every cached call: `literag/<namespace>` (lite-cache §0 2026-11-19, owner's choice A). Never down to the kb or the user.
- `cache.enabled` defaults to `false`; with it false, no hook is registered, no lite-cache schema is created, request bodies and index identity are unchanged. The cache is **not** part of index identity.
- Tests use `LITERAG_TEST_DATABASE_URL`; the store keeps schema `lite_rag_test`; the cache uses schema `lite_rag_cache_test` (dropped before and after). Model calls go to a recording transport (lite-model `build(..., http_client=...)`), never a real model.
- Stage close-out per lite-rag CLAUDE.md (`/simplify`, Codex `exec -s read-only … < /dev/null`, fix, re-run); commits only with the owner's go-ahead.

## Review Focus

1. **`contextual.concurrency > 1`** → the worker threads must see `call_context`; otherwise caching is silently off for ingest. (Task 2)
2. **`cache.roles` without one role** (e.g. only `answer`) → that role's calls go uncached, the others still cached; nothing breaks. (Task 4)
3. **Cache enabled with the Chroma store or no `database_url`** → `ConfigError` at settings validation, before any ingest. (Task 3)
4. **A cache read failure during `ask()` or ingest** (database down) → lite-cache raises `StoreError` before sending; lite-rag must surface it as a `LiteRagError` (HTTP 503-ish), not a raw traceback, and not fall back silently. (Task 4)
5. **The question log when the directory does not exist or the question contains newlines/Unicode** → one valid JSON object per line, UTF-8, directory created. (Task 5)

---

## File structure

```
src/lite_rag/config.py     + CacheConfig, Settings.cache (+ validation)
src/lite_rag/cache.py      new: open_cache(settings) -> CacheBinding (store + hook), cache_context(...)
src/lite_rag/llm.py        BoundedLLM.map copies context; Completion.cached; build_model returns LiteModel
src/lite_rag/engine.py     RagRuntime registers the hook / closes the store; RagEngine wraps roles' calls; ask() logs
src/lite_rag/answer.py     Answer.cached
src/lite_rag/cli.py        `literag ask` marks a cached answer
src/lite_rag/server/app.py Answer JSON carries `cached` (additive)
pyproject.toml, uv.lock    extra + source + dev group
tests/unit/test_cache_config.py, tests/unit/test_bounded_context.py
tests/integration/test_cache.py   (PG + recording transport)
```

---

### Task 1: Dependency and `Completion.cached`

**Files:** Modify `pyproject.toml`, `uv.lock`, `src/lite_rag/llm.py`, `src/lite_rag/answer.py`; Test `tests/unit/test_llm.py`

**Interfaces — Produces:** `Completion(text, model_id, usage, cached: bool = False)`; `Answer.cached: bool = False`; `build_model(...) -> lite_model.LiteModel`.

- [ ] **Step 1: Failing test** — `test_a_cached_reply_is_marked`: a `RoleLLM` over a lite-model `LiteModel` (recording transport) with a hook whose `before` returns a stored `TextReply` → `complete()` returns `Completion.cached is True` and the stored `model_id`; without the hook `cached is False`.
- [ ] **Step 2:** run → FAIL (no attribute `cached`).
- [ ] **Step 3:** implement; `uv add --optional cache lite-cache` with the source URL; `lite-rag[cache]` in dev; `uv lock`.
- [ ] **Step 4:** `uv run pytest tests/unit/test_llm.py tests/unit/test_llm_requests.py -q` → PASS (request bodies unchanged).

### Task 2: `BoundedLLM.map` carries the caller's context

**Files:** Modify `src/lite_rag/llm.py`; Test `tests/unit/test_bounded_context.py`

**Interfaces — Produces:** `BoundedLLM.map(one, items)` — same signature; each item runs in `contextvars.copy_context()` taken at the call.

- [ ] **Step 1: Failing test** — `test_call_context_reaches_hooks_in_map_workers`: a `LiteModel` on a recording transport with a recording hook; `RoleLLM(model, "contextual").bounded(4).map(complete_one, [8 items])` inside `call_context(scope="literag/ns", cache=True)` → every one of the 8 `Call.context` the hook saw has `scope == "literag/ns"`. With `bounded(1)` too.
- [ ] **Step 2:** run → FAIL with concurrency 4 (contexts empty in workers).
- [ ] **Step 3:** implement: capture `ctx = contextvars.copy_context()` in `map`, run `lambda i: ctx.copy().run(one, i)` (a fresh copy per item: `Context.run` is not re-entrant across threads).
- [ ] **Step 4:** PASS; full unit suite green.

### Task 3: `CacheConfig` and validation

**Files:** Modify `src/lite_rag/config.py`; Test `tests/unit/test_cache_config.py`

**Interfaces — Produces:** `CacheConfig(enabled: bool = False, roles: list[Role] = ["answer", "contextual", "graph"], ttl_s: int | None = None, schema_name: str = Field("lite_cache", alias="schema"), question_log: Path | None = None)`; `Settings.cache: CacheConfig`. Env: `LITERAG_CACHE__ENABLED` etc.

- [ ] **Step 1: Failing tests** — defaults (`enabled False`, all three roles); `LITERAG_CACHE__ENABLED=true` with `store.backend=chroma` → `ConfigError`/`ValidationError` naming PostgreSQL; enabled without `database_url` → same; `roles=["nope"]` refused; `ttl_s=0` refused; `schema` validated like `store.schema` (identifier) and not equal to `store.schema`; the cache settings are not in `index_identity`.
- [ ] **Step 2–4:** fail → implement → pass.

### Task 4: Runtime registers the hook; the engine scopes each role

**Files:** Create `src/lite_rag/cache.py`; Modify `src/lite_rag/engine.py`; Test `tests/integration/test_cache.py`

**Interfaces:**
- `cache.open_cache(settings: Settings) -> CacheBinding` — `CacheBinding(store, hook)`, `close()`; imports `lite_cache` via `lite_rag.extras`; store = `lite_cache.open_store(lite_cache.Settings(database_url=settings.database_url, schema_name=settings.cache.schema_name))`.
- `cache.cache_context(settings: Settings, namespace: str, role: Role, question: str | None = None) -> ContextManager` — `call_context(scope=f"literag/{namespace}", cache=Policy(ttl_s=...), question=...)` when `enabled` and `role in roles`, else a null context.
- `RagRuntime`: when `settings.cache.enabled`, `_role()` builds the model, then `open_cache` and `model.hooks.register(binding.hook)`; `close()` closes the binding too. A `StoreError` from lite-cache is re-raised as a lite-rag `StoreError`/`LiteRagError` (HTTP 503) at the engine boundary.
- `RagEngine.upsert_text`: the contextualizer and graph-extractor calls each run inside `cache_context(..., role)`; `ask()` runs the answer call inside `cache_context(..., "answer", question)` and sets `Answer.cached`.

- [ ] **Step 1: Failing tests** (`test_cache.py`, real PG, recording transport, FakeEmbedder; a second embedding model id simulates "change the embedding"):
  - `test_off_is_unchanged` — `enabled=false`: ingest + ask twice → requests counted as before, no `lite_rag_cache_test` schema exists, `Answer.cached is False`.
  - `test_reindex_into_a_new_kb_reuses_contexts_and_graph` — contextual + graph on; ingest doc into kb `a`; `reindex --into b` with another embedding model → zero new model requests; stored contexts and graph rows of `b` equal `a`'s.
  - `test_a_repeated_question_is_answered_from_the_cache` — second `ask` sends nothing, `cached is True`, same text, citations and `model_id`.
  - `test_two_namespaces_never_share` — same doc and question in `t1` and `t2` → both call the model.
  - `test_contextual_concurrency_four_still_hits` — `chunking.contextual.concurrency=4`.
  - `test_roles_not_listed_are_not_cached` — `roles=["answer"]` → contextual calls repeat on reindex, answers hit.
  - `test_a_cache_read_failure_is_a_lite_rag_error` — cache schema dropped mid-run → `ask` raises lite-rag's error type, no request sent.
- [ ] **Step 2–4:** fail → implement → pass; `uv run pytest -q` all green.

### Task 5: Question log, CLI and HTTP

**Files:** Modify `src/lite_rag/engine.py` (or `cache.py`), `src/lite_rag/cli.py`, `src/lite_rag/server/app.py`; Tests in `tests/integration/test_cache.py`, `tests/unit/test_cli.py`, `tests/integration/test_server.py`

**Interfaces:** `cache.question_log(path: Path, scope: str, question: str, now: datetime) -> None` — appends `{"ts": ISO-8601 UTC, "scope": ..., "question": ...}` as one JSON line (UTF-8, `ensure_ascii=False`), creating the directory.

- [ ] **Step 1: Failing tests** — `question_log` unset → no file; set → two asks give two lines, each valid JSON, a question with `\n` and Chinese round-trips; `literag ask` output marks a cached answer (`(cached)`); `POST /v1/ask` JSON has `"cached": true` on the second call (additive field; existing fields unchanged).
- [ ] **Step 2–4:** fail → implement → pass.

### Task 6: Close-out

- [ ] lite-rag design: S18 record in §12 (what was built, numbers), status in AGENTS.md "Current stage"; README "cache" section; workspace `AGENTS.md` pin table gains lite-cache.
- [ ] `uv run pytest -q`, `uv run ruff check && uv run ruff format --check && uv run mypy`; `-m ollama` round trip with cache on (two asks; second cached) when Ollama is up.
- [ ] `/simplify`, Codex (`codex exec -s read-only -o <out.md> "Review git diff b0d63bc …" < /dev/null`), fix, re-run; propose commits (lite-rag S18; lite-cache docs scope change; lite-model docs plan line).
