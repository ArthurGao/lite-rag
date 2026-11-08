"""What known embedding models need (design §3.2): how documents and queries must be written, and
which tokenizer measures their input. One table, one lookup — every embedder goes through it.

Getting the prefixes wrong raises no error — retrieval quality just drops quietly — so the
embedder applies them itself and callers never see them. Matched on the model name (lowercase,
regex search), first match wins; `embedding.query_prefix` / `document_prefix` / `tokenizer` in the
config override them (`""` = no prefix).

Sources (checked 2026-11-08, from each model card on Hugging Face):
- intfloat/multilingual-e5-*: "query: " / "passage: ", required, even for non-English text
- nomic-ai/nomic-embed-text-*: "search_query: " / "search_document: ", required
- BAAI/bge-*-v1.5: a query instruction for retrieval; "no instruction needs to be added to passages"
- BAAI/bge-m3: no instruction
- Qwen/Qwen3-Embedding-*: f"Instruct: {task}\\nQuery:{query}"; no instruction for documents
- gemini-embedding-2 (ai.google.dev embeddings guide): "task: search result | query: " /
  "title: {title} | text: " ("title: none" without one); gemini-embedding-001 takes a taskType
  parameter instead, as Voyage and Cohere take an input_type: no prefix for those
"""

import re
from dataclasses import dataclass

QWEN3_TASK = "Given a web search query, retrieve relevant passages that answer the query"
BGE_ZH = "为这个句子生成表示以用于检索相关文章："
BGE_EN = "Represent this sentence for searching relevant passages:"
HF_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")  # "org/name": a Hugging Face repo id


@dataclass(frozen=True)
class Prefixes:
    query: str = ""
    document: str = ""


@dataclass(frozen=True)
class ModelProfile:
    prefixes: Prefixes
    tokenizer: str | None  # Hugging Face tokenizer id; None = estimate token counts


# (regex searched in the lowercased model name, prefixes, tokenizer for non-HF names such as
# Ollama tags). Order matters: specific first.
KNOWN: tuple[tuple[str, Prefixes, str | None], ...] = (
    (r"bge-m3", Prefixes(), "BAAI/bge-m3"),
    (r"bge-(large|base|small)-zh-v1\.5", Prefixes(query=BGE_ZH), None),
    (r"bge-(large|base|small)-en-v1\.5", Prefixes(query=BGE_EN), None),
    (
        r"nomic-embed-text",
        Prefixes(query="search_query: ", document="search_document: "),
        "nomic-ai/nomic-embed-text-v1.5",
    ),
    (r"qwen3-embedding", Prefixes(query=f"Instruct: {QWEN3_TASK}\nQuery:"), None),
    (
        r"(^|[/:_-])multilingual-e5([-_]|$)",
        Prefixes(query="query: ", document="passage: "),
        "intfloat/multilingual-e5-large",  # all sizes share the XLM-R tokenizer
    ),
    (r"(^|[/:_-])e5([-_]|$)", Prefixes(query="query: ", document="passage: "), None),
    (
        r"^gemini-embedding-2",
        Prefixes(query="task: search result | query: ", document="title: none | text: "),
        None,
    ),
)


def profile_for(
    model: str,
    query: str | None = None,
    document: str | None = None,
    tokenizer: str | None = None,
) -> ModelProfile:
    """The profile for `model`; explicit `query` / `document` / `tokenizer` (even "") win.

    A model named like a Hugging Face repo (`org/name`) is its own tokenizer."""
    name = model.lower()
    known = next(
        ((p, t) for pattern, p, t in KNOWN if re.search(pattern, name)), (Prefixes(), None)
    )
    prefixes, known_tokenizer = known
    if known_tokenizer is None and HF_REPO.match(model):
        known_tokenizer = model
    return ModelProfile(
        Prefixes(
            query=prefixes.query if query is None else query,
            document=prefixes.document if document is None else document,
        ),
        known_tokenizer if tokenizer is None else (tokenizer or None),
    )


def prefixes_for(model: str, query: str | None = None, document: str | None = None) -> Prefixes:
    return profile_for(model, query, document).prefixes
