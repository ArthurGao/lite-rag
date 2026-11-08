import json
from typing import Any

import httpx2
import pytest

from lite_rag.config import EmbeddingConfig
from lite_rag.embedding import EmbeddingError, OllamaEmbedder, OpenAICompatEmbedder
from lite_rag.errors import ConfigError

Handler = Any


def _client(handler: Handler) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def _ollama_reply(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    return httpx2.Response(200, json={"embeddings": [[float(len(t)), 1.0] for t in body["input"]]})


def test_ollama_batches_and_keeps_order() -> None:
    seen: list[list[str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/embed"
        seen.append(json.loads(request.content)["input"])
        return _ollama_reply(request)

    cfg = EmbeddingConfig(batch_size=2, max_concurrency=1, document_prefix="passage: ")
    e = OllamaEmbedder(cfg, _client(handler))
    vectors = e.embed_documents(["a", "bb", "ccc"])
    assert seen == [["passage: a", "passage: bb"], ["passage: ccc"]]
    assert [v[0] for v in vectors] == [10.0, 11.0, 12.0]  # len("passage: x…"), in input order


def test_the_query_gets_its_own_prefix() -> None:
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.extend(json.loads(request.content)["input"])
        return _ollama_reply(request)

    e = OllamaEmbedder(EmbeddingConfig(query_prefix="query: "), _client(handler))
    e.embed_query("refund")
    assert seen == ["query: refund"]


def test_parallel_batches_still_come_back_in_order() -> None:
    e = OllamaEmbedder(EmbeddingConfig(batch_size=1, max_concurrency=4), _client(_ollama_reply))
    texts = ["x" * n for n in range(1, 9)]
    assert [v[0] for v in e.embed_documents(texts)] == [float(n) for n in range(1, 9)]


def test_overload_is_retried_with_backoff_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(503) if calls < 3 else _ollama_reply(request)

    sleeps: list[float] = []
    e = OllamaEmbedder(EmbeddingConfig(max_retries=4), _client(handler), sleep=sleeps.append)
    assert e.embed_query("q") == [1.0, 1.0]
    assert calls == 3 and sleeps == [0.5, 1.0]


def test_exhausted_retries_raise_instead_of_returning_nothing() -> None:
    e = OllamaEmbedder(
        EmbeddingConfig(max_retries=2),
        _client(lambda r: httpx2.Response(500)),
        sleep=lambda s: None,
    )
    with pytest.raises(EmbeddingError, match="failed after 3 attempt"):
        e.embed_documents(["a"])


def test_a_client_error_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(400, json={"error": "model not found"})

    e = OllamaEmbedder(EmbeddingConfig(max_retries=4), _client(handler), sleep=lambda s: None)
    with pytest.raises(EmbeddingError, match="failed after 1 attempt"):
        e.embed_query("q")
    assert calls == 1


def test_a_wrong_vector_count_is_an_error() -> None:
    e = OllamaEmbedder(
        EmbeddingConfig(), _client(lambda r: httpx2.Response(200, json={"embeddings": [[1.0]]}))
    )
    with pytest.raises(EmbeddingError, match="asked for 2 vectors, got 1"):
        e.embed_documents(["a", "b"])


def test_dim_is_learned_once_from_the_model() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return _ollama_reply(request)

    e = OllamaEmbedder(EmbeddingConfig(), _client(handler))
    assert (e.dim, e.dim, calls) == (2, 2, 1)


def test_openai_compatible_sends_key_dimensions_and_reorders_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_EMBED_KEY", "sk-test")

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url == "https://api.example.com/v1/embeddings"
        assert request.headers["authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        assert body == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 2}
        data = [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
        return httpx2.Response(200, json={"data": data})

    cfg = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        base_url="https://api.example.com/v1",
        api_key_env="MY_EMBED_KEY",
        dimensions=2,
    )
    e = OpenAICompatEmbedder(cfg, _client(handler))
    assert e.embed_documents(["a", "b"]) == [[1.0, 0.0], [0.0, 2.0]]


def test_openai_compatible_refuses_a_missing_key() -> None:
    cfg = EmbeddingConfig(provider="openai", base_url="https://x/v1", api_key_env="NOPE_KEY")
    with pytest.raises(ConfigError, match="NOPE_KEY"):
        OpenAICompatEmbedder(cfg)
