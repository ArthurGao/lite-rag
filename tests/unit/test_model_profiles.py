import json

import httpx2
import pytest

from lite_rag.config import EmbeddingConfig
from lite_rag.embedding import OllamaEmbedder, model_id
from lite_rag.model_profiles import Prefixes, prefixes_for, profile_for


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("bge-m3:567m", Prefixes()),
        ("jeffh/intfloat-multilingual-e5-large:f16", Prefixes("query: ", "passage: ")),
        ("intfloat/e5-base-v2", Prefixes("query: ", "passage: ")),
        ("nomic-embed-text:v1.5", Prefixes("search_query: ", "search_document: ")),
        ("BAAI/bge-small-zh-v1.5", Prefixes("为这个句子生成表示以用于检索相关文章：", "")),
        ("qwen3-embedding:0.6b", Prefixes(query=prefixes_for("qwen3-embedding").query)),
        ("text-embedding-3-small", Prefixes()),  # unknown: nothing added
        ("acme/encoder-xe5-base", Prefixes()),  # "e5" inside another token is not e5
    ],
)
def test_known_models_get_their_prefixes(model: str, expected: Prefixes) -> None:
    assert prefixes_for(model) == expected


def test_qwen3_queries_use_the_instruct_template() -> None:
    query = prefixes_for("Qwen/Qwen3-Embedding-0.6B").query
    assert query.startswith("Instruct: ") and query.endswith("\nQuery:")


def test_configured_prefixes_win_and_empty_means_none() -> None:
    assert prefixes_for("nomic-embed-text", query="", document="doc: ") == Prefixes("", "doc: ")


def test_the_embedder_applies_the_model_prefixes_itself() -> None:
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        inputs = json.loads(request.content)["input"]
        seen.extend(inputs)
        return httpx2.Response(200, json={"embeddings": [[1.0] for _ in inputs]})

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    e = OllamaEmbedder(EmbeddingConfig(model="nomic-embed-text"), client)
    e.embed_documents(["refunds take 7 days"])
    e.embed_query("how long do refunds take")
    assert seen == [
        "search_document: refunds take 7 days",
        "search_query: how long do refunds take",
    ]
    assert model_id(e.provider, e.model) == "ollama/nomic-embed-text"


@pytest.mark.parametrize(
    ("model", "tokenizer"),
    [
        ("bge-m3:567m", "BAAI/bge-m3"),
        ("nomic-embed-text", "nomic-ai/nomic-embed-text-v1.5"),
        ("BAAI/bge-small-zh-v1.5", "BAAI/bge-small-zh-v1.5"),  # an HF repo is its own tokenizer
        ("text-embedding-3-small", None),  # unknown: estimate
    ],
)
def test_the_tokenizer_follows_the_model(model: str, tokenizer: str | None) -> None:
    assert profile_for(model).tokenizer == tokenizer


def test_a_configured_tokenizer_wins_and_empty_means_estimate() -> None:
    assert profile_for("bge-m3", tokenizer="my/tok").tokenizer == "my/tok"
    assert profile_for("bge-m3", tokenizer="").tokenizer is None
