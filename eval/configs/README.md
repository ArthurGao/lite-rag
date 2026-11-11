# Evaluation configs

Each file is a lite-rag config overlay (environment variables still win). Compare any set:

```
uv run --all-extras literag eval -c eval/configs/chunk-fixed.yaml -c eval/configs/chunk-markdown.yaml
```

- `chunk-*.yaml` — the chunkers, all hybrid retrieval, 120-token chunks (small, so the choice matters)
- `mode-vector.yaml`, `mode-bm25.yaml` — one retrieval path only (markdown chunks)
- `hybrid-rerank.yaml` — hybrid plus the cross-encoder (needs the fastembed extra)
- `chunk-semantic.yaml`, `chunk-semantic-chonkie.yaml` — semantic chunking, own and chonkie (`[chonkie]` extra)
- `chunk-parent-child.yaml` (120/480) and `chunk-parent-child-small.yaml` (40/120, against
  `chunk-markdown-small.yaml`) — small chunks indexed, parents returned
- `chunk-contextual.yaml` — markdown chunks with an LLM context header each (local qwen3; one
  call per chunk, a few minutes)
- `embed-*.yaml` — embedding models, vector retrieval only, against `mode-vector.yaml` (bge-m3):
  fastembed MiniLM and multilingual-e5-large, sentence-transformers Qwen3-Embedding-0.6B (first
  run downloads ~4 GB in all)
- `pdf-*.yaml` — PDF loaders, run on the PDF corpus:

```
uv run --all-extras literag eval -c eval/configs/pdf-pypdf-recursive.yaml \
  -c eval/configs/pdf-docling-markdown.yaml --corpus eval/corpus-pdf --golden eval/golden-pdf.jsonl
```

Identifier questions (`exact`: error codes, config keys, versions, model numbers, one section
per identifier), to compare retrieval paths where BM25 should help — e.g. against the mode configs:

```
uv run --all-extras literag eval -c eval/configs/mode-vector.yaml -c eval/configs/mode-bm25.yaml \
  -c eval/configs/chunk-markdown.yaml --corpus eval/corpus-exact --golden eval/golden-exact.jsonl
```

Multi-hop questions (S11: the answer sits two or three documents away from what the question
names), against graph retrieval — `graph-*.yaml` turn on entity extraction (local qwen3, one call
per chunk, a few minutes per build):

```
uv run --all-extras literag eval -c eval/configs/chunk-markdown.yaml -c eval/configs/graph-hybrid.yaml \
  --corpus eval/corpus-multihop --golden eval/golden-multihop.jsonl
```

The PDFs in `eval/corpus-pdf/` are generated from `eval/pdf-src/*.md` by `eval/make_pdf.py`.
