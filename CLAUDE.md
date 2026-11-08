@AGENTS.md

# Claude Code notes

- At the start of a session: check §12 of `docs/lite-rag-design.md` for the current stage and its acceptance,
  and skim the project KB at `/Users/arthurgao/Docs/Knowledgebase/projects/lite-rag/`.
- Never answer an Ollama / Chroma / jieba / rerank-library API question from memory: check context7 or the
  pinned version's docs, and record surprises in the KB (`KB/global/<library>/` for cross-project facts).
- Relevant user-level skills: `rag-design` (chunking, hybrid search, rerank, evaluation), `llm-app-architecture`
  (switchable LLM, no fallback chain, provenance), `llm-model-eval` (comparing models), `testing-and-ci`,
  `kb-maintain`.
- Ollama runs on the host; check it for free with `curl -s http://127.0.0.1:11434/api/tags` before an
  `-m ollama` run.
