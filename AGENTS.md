# lite-rag — Agent Guide

lite-rag is a **small, fully local RAG engine built for learning**: own chunking, own BM25, vector search
in PostgreSQL + pgvector (Chroma as an optional extra), RRF fusion, optional cross-encoder rerank, graph
retrieval, and answer generation through lite-model roles (Ollama / OpenAI-compatible / Anthropic). It is also a KB engine for
[llm-steward](https://github.com/ArthurGao/llm-steward), which imports it in-process.

This file is the shared guide for every coding agent. `CLAUDE.md` imports it.

Naming: repo `lite-rag`, Python package `lite_rag`, CLI `literag`, env prefix `LITERAG_`.

## Sources of truth (highest first)

1. `docs/lite-rag-design.md` — the design. §3 interfaces (§3.1 chunking, §3.2 embedding), §4 storage and ids, §5 retrieval, §6 generation,
   §7 HTTP service, §8 evaluation, §10 llm-steward integration, §12 stages and acceptance.
2. Project KB: `/Users/arthurgao/Docs/Knowledgebase/projects/lite-rag/` — decisions, findings, lessons.
   Engine and library facts shared across projects (Ollama, Chroma, …) go to `KB/global/`. Write with the
   `kb-maintain` skill.

If a change must contradict the design, update the design in the same change and say why in the commit.

## Current stage

**S0–S25 done** (2026-11-08/25; S25: lineage — a generic provenance ledger `lite_rag.lineage` recorded by every write in its transaction, chunks keyed by text, `literag lineage backfill|show|upstream|downstream|gc`, opt-in query traces `search_traced` / `ask` / `record_cited` / `literag trace show` that say whether an answer used text that has since changed — design §12 S25 record, spec `docs/superpowers/specs/2026-11-24-lineage-design.md`; S24: a harder cross-language set (near-duplicate variants): the best single multilingual embedder (Qwen3-0.6B, 0.97 hit@1) beats per-language embedders (0.94 with rerank) — design §12 S24 record; S23: the Cohere and Gemini embedders call through lite-model 0.8.0, request bodies and index identity unchanged — every HTTP embedder is lite-model's now — design §12 S23 record; S22: each kb picks a named embedder (`embeddings.<name>`, `create_kb(embedding=)`, `--embedding`), found again from the identity it recorded; `search_kbs` across embedders fuses by RRF — enable rerank there (measured on `golden-xlang`) — design §12 S22 record; S21: `mode: route` — the `route` role (local qwen3:1.7b) picks vector or hybrid per question; measured first with `eval/route_ceiling.py`; opt-in, not the default — design §12 S21 record; S20: Voyage embeddings and the new Voyage reranker (`rerank.provider: voyage`) call through lite-model 0.7.0, index identity unchanged, `-m voyage` opt-in and paid — design §12 S20 record; S19: the ollama / openai embedders call through lite-model 0.6.0's `embed()`, config and index identity unchanged — design §12 S19 record; S18: lite-cache L1 for the `answer` / `contextual` / `graph` calls, `cache` extra, off by default, scope `literag/<namespace>`, keyed source-block nonces for cached calls (`LITERAG_PROMPT_SECRET`) — design §12 S18 record; S17: the index roles `contextual` / `graph` default to `qwen3:1.7b` without thinking — answers over their indexes unchanged, design §12 S17 record; S16: graph extraction asks for a reply held to a JSON schema (`GraphReply`), once more if still off it — design §12 S16 record; S15: every LLM call goes through lite-model 0.2.0 — roles `answer` / `contextual` / `graph`, request bodies and index identity unchanged, design §12 S15 record; S11: graph retrieval, local mode, PostgreSQL only; S12: `literag eval` runs on
lite-eval, numbers identical to before; S13: the grounded judge from lite-eval E5 calibrated on a
constructed set and run on real `literag ask` answers, `literag eval-answers`; S14: answer correctness (`key_fact`), `response_model`, and the measured effect of
LiteLLM auto-routing on answers — design §12 S14 record). The owner's direction
(2026-11-12): lite-rag covers the main RAG techniques side by side, each a stage measured with `literag eval`
(candidates: graph global mode S11b, query rewriting — HyDE / multi-query, RAPTOR, self-checking retrieval);
llm-steward's needs still come first.

Decided (2026-11-08): separate public repo; library + optional HTTP service (`literag serve`, Docker image) over one
`RagEngine`; storage is PostgreSQL + pgvector by default (vectors, own BM25 postings and the document registry in one
transaction), Chroma + SQLite optional; own BM25 with jieba;
Ollama for local embedding (`bge-m3`) and chat (`qwen3:8b` answers, `qwen3:1.7b` contexts and graphs); LLM switchable between `ollama`, `openai`
(any OpenAI-compatible endpoint, including llm-gateway) and `anthropic` — through lite-model since S15 (also
`gateway`, `litellm`).

## Rules

1. **Every component is a Protocol** (`Chunker`, `Embedder`, `Store`, `Reranker`, `LLM`);
   the engine depends on the Protocols, never on a concrete class. Swapping an implementation is config only.
2. **Namespaces are mandatory**: an engine is bound to one namespace (`project` or `project/tenant`) and
   builds every physical key (`<namespace>:<kb>`) itself; no API takes a raw physical key.
3. **An index is bound to its model**: `kbs` records model, dim, both prefixes and the chunker; any mismatch
   raises. Switching model = a new kb + re-embedding, never in place. Model-specific prefixes live in
   `model_profiles.py`, applied by the embedder — callers never add them.
4. **Ids are stable**: chunk id = `"{doc_id}#{n}"`; `upsert_text` replaces a document's chunks in both indexes;
   `delete` removes them from both. On PostgreSQL a document write is one transaction. An unchanged document (same hash and chunker config) is a no-op.
5. **No silent fallbacks**: a failed model call raises. No fallback chain between LLMs or embedders. Every answer
   records `backend/model` of the model that answered; citations are validated against the retrieved chunks.
   LLM calls go through lite-model (S15; `llm.py` is lite-rag's side): its rules apply — retries on the same model
   only, errors without upstream text, keys only through `api_key_env`. `contextual` and `graph` are index
   identity: never routable, and the top-level `llm` settings tune answers only.
6. **Prompts live in `src/lite_rag/prompts/*.j2`**, never inline strings.
7. **Secrets only from environment variables.** Never commit `.env*` except `.env.example`.
8. **Tests never call a real model or a paid API.** Unit and integration tests use `FakeEmbedder` / `FakeLLM`.
   Real Ollama runs only under the opt-in marker `-m ollama`. Integration tests use their own PG schema
   (`LITERAG_TEST_DATABASE_URL`, schema `lite_rag_test`), never the app schema. Never call llm-gateway subscription profiles from
   tests or on your own initiative.
9. **Pinned dependencies** for the engines (`psycopg`, `fastembed`, later `chromadb` and the rerank package): exact versions; an upgrade
   re-runs `-m ollama` and the eval.
10. **Measure, don't guess**: a change meant to improve retrieval comes with an eval run (`literag eval`) before and
   after; record the numbers in the commit or the KB.

## Tech stack

Python 3.12, uv, pydantic / pydantic-settings, Typer, PostgreSQL + pgvector, chromadb (extra), FastAPI (extra), jieba, lite-model (every LLM call;
it pins the OpenAI and Anthropic SDKs), Jinja2, pytest, ruff, mypy. Ollama on the host (`127.0.0.1:11434`).

## Commands

```
uv run pytest                 # unit + integration, fakes only
uv run pytest -m ollama       # opt-in: real Ollama (bge-m3 / qwen3:8b / qwen3:1.7b must be pulled)
uv run --all-extras pytest -m model   # opt-in: real Hugging Face models (the reranker, ~0.6 GB once)
uv run --all-extras literag eval -c eval/configs/chunk-markdown.yaml -c eval/configs/mode-vector.yaml   # compare configs
uv run --with ranx==0.3.21 --all-extras literag eval -c … --crosscheck   # also recompute the metrics with ranx
uv run --all-extras lite-eval run eval/judge-grounded/suites/calibrate.yaml --config qwen3-8b-nothink   # calibrate the grounded judge
uv run --all-extras literag eval-answers -c eval/configs/chunk-markdown.yaml --calibration <run dir> --min-agreement <x>
uv run python eval/make_judge_grounded.py      # regenerate the judge's calibration set (committed)
uv run ruff check && uv run ruff format --check && uv run mypy
uv run literag ingest <kb> <path> | search <kb> "<q>" | ask <kb> "<q>" | eval --config a.yaml --config b.yaml
uv run literag reindex <kb> --into <new-kb>      # rebuild under the current settings (model / chunker)
uv run literag key create --name <who> --namespace '<glob>' [--kb '<glob>'] --mode read|write
uv run --extra server literag serve              # HTTP API on 127.0.0.1:8700 (design §7)
docker compose -f deploy/compose.yaml --env-file deploy/.env up -d --build   # service + PG (Ollama on host)
```

## Workflow

1. Work stage by stage (design §12). Each stage ends with every check green; every new test seen red once.
2. Definition of done: ruff, mypy, tests pass; acceptance items covered by tests; design updated if behaviour
   changed.
3. Conventional Commits (`feat(retrieval): …`). All code-facing text and docs (design doc, README, plans, comments) in English; Chinese only as examples or test data.
4. **Stage close-out — after every stage's implementation passes its checks, before its commit and before the
   next stage starts:**
   1. Simplify the stage's full diff (reuse, simplification, efficiency, altitude) and apply the fixes.
   2. Independent read-only Codex review of the same diff (see `CLAUDE.md` for the command).
   3. Fix every finding that holds up — a bug gets a failing regression test first. A finding you reject is
      written down with the reason; never silently dropped.
   4. Re-run all checks (step 2) so they cover the final code, then report the review outcome with the stage.
   S8 is the final close-out over the whole engine before S9 (llm-steward adapter).
