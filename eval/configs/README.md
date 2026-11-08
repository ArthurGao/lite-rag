# Evaluation configs

Each file is a lite-rag config overlay (environment variables still win). Compare any set:

```
uv run --all-extras literag eval -c eval/configs/chunk-fixed.yaml -c eval/configs/chunk-markdown.yaml
```

- `chunk-*.yaml` — the chunkers, all hybrid retrieval, 120-token chunks (small, so the choice matters)
- `mode-vector.yaml`, `mode-bm25.yaml` — one retrieval path only (markdown chunks)
- `hybrid-rerank.yaml` — hybrid plus the cross-encoder (needs the fastembed extra)
