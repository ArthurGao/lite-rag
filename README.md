# lite-rag

A small, fully local RAG engine built for learning — every step is plain code you can read and swap:

- **Chunking**, configurable per knowledge base: fixed, recursive, sentence, paragraph, Markdown, semantic,
  parent-child, plus optional LLM-written chunk context (contextual retrieval)
- **Embedding**: local (Ollama, fastembed, sentence-transformers) or API (OpenAI-compatible, Voyage, Cohere, Gemini),
  one model per knowledge base, chosen by measuring on your own data
- **Storage**: PostgreSQL + pgvector (one transaction per document), or Chroma + SQLite for a dependency-free setup
- **Retrieval**: vector search + an own BM25 (jieba), fused with RRF; optional cross-encoder rerank
- **Generation**: answers with validated citations, through a switchable LLM — Ollama, any OpenAI-compatible
  endpoint, or Anthropic
- **Evaluation**: `literag eval` compares configurations side by side (hit@k, MRR, latency)

Use it as a Python library or run it as a service (`literag serve`, Docker image, HTTP API). The default stack
(Ollama + PostgreSQL/pgvector + jieba + bge models) runs locally and costs nothing.

Status: design stage. See [docs/lite-rag-design.md](docs/lite-rag-design.md).
