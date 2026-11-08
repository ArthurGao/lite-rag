# lite-rag — Agent Guide

lite-rag is a **small, fully local RAG engine built for learning**: own chunking, own BM25, vector search
in Chroma, RRF fusion, optional cross-encoder rerank, and answer generation through a switchable LLM
(Ollama / OpenAI-compatible / Anthropic). It is also a KB engine for
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

**S5 — evaluation** (design §12; S0–S4 done 2026-11-08). Next: S5b advanced chunking and more embedders. llm-steward's P3 (WeKnora) is paused until S9 is done.

Decided (2026-11-08): separate public repo; library + optional HTTP service (`literag serve`, Docker image) over one
`RagEngine`; storage is PostgreSQL + pgvector by default (vectors, own BM25 postings and the document registry in one
transaction), Chroma + SQLite optional; own BM25 with jieba;
Ollama for local embedding (`bge-m3`) and chat (`qwen3:8b`); LLM switchable between `ollama`, `openai`
(any OpenAI-compatible endpoint, including llm-gateway) and `anthropic`.

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
   records `provider:model`; citations are validated against the retrieved chunks.
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

Python 3.12, uv, pydantic / pydantic-settings, Typer, PostgreSQL + pgvector, chromadb (extra), FastAPI (extra), jieba, OpenAI SDK (OpenAI-compatible LLMs and
Ollama `/v1`), Anthropic SDK (extra), Jinja2, pytest, ruff, mypy. Ollama on the host (`127.0.0.1:11434`).

## Commands

```
uv run pytest                 # unit + integration, fakes only
uv run pytest -m ollama       # opt-in: real Ollama (bge-m3 / qwen3:8b must be pulled)
uv run --all-extras pytest -m model   # opt-in: real Hugging Face models (the reranker, ~0.6 GB once)
uv run --all-extras literag eval -c eval/configs/chunk-markdown.yaml -c eval/configs/mode-vector.yaml   # compare configs
uv run ruff check && uv run ruff format --check && uv run mypy
uv run literag ingest <kb> <path> | search <kb> "<q>" | ask <kb> "<q>" | eval --config a.yaml --config b.yaml
```

## Workflow

1. Work stage by stage (design §12). Each stage ends with every check green; every new test seen red once.
2. Definition of done: ruff, mypy, tests pass; acceptance items covered by tests; design updated if behaviour
   changed.
3. Conventional Commits (`feat(retrieval): …`). All code-facing text in English.
4. **Stage close-out — after every stage's implementation passes its checks, before its commit and before the
   next stage starts:**
   1. Simplify the stage's full diff (reuse, simplification, efficiency, altitude) and apply the fixes.
   2. Independent read-only Codex review of the same diff (see `CLAUDE.md` for the command).
   3. Fix every finding that holds up — a bug gets a failing regression test first. A finding you reject is
      written down with the reason; never silently dropped.
   4. Re-run all checks (step 2) so they cover the final code, then report the review outcome with the stage.
   S8 is the final close-out over the whole engine before S9 (llm-steward adapter).
