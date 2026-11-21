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


def _vectors(request: httpx2.Request, vector: Any = None) -> httpx2.Response:
    """An OpenAI-compatible `/embeddings` reply: `[len(text), 1.0]` per input (S19: Ollama is
    reached at its `/v1/embeddings` through lite-model, not `/api/embed`)."""
    body = json.loads(request.content)
    data = [
        {"object": "embedding", "index": i, "embedding": vector or [float(len(t)), 1.0]}
        for i, t in enumerate(body["input"])
    ]
    return httpx2.Response(200, json={"object": "list", "data": data, "model": body["model"]})


def _status(code: int) -> httpx2.Response:
    # "retry in 1 ms": the SDK's own backoff would otherwise sleep in every retry test
    return httpx2.Response(code, headers={"retry-after-ms": "1"}, json={"error": {"message": "x"}})


def test_ollama_batches_and_keeps_order() -> None:
    seen: list[list[str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == "http://127.0.0.1:11434/v1/embeddings"
        seen.append(json.loads(request.content)["input"])
        return _vectors(request)

    cfg = EmbeddingConfig(batch_size=2, max_concurrency=1, document_prefix="passage: ")
    e = OllamaEmbedder(cfg, _client(handler))
    vectors = e.embed_documents(["a", "bb", "ccc"])
    assert seen == [["passage: a", "passage: bb"], ["passage: ccc"]]
    assert [v[0] for v in vectors] == [10.0, 11.0, 12.0]  # len("passage: x…"), in input order


def test_the_ollama_request_carries_no_dimensions() -> None:
    # it did not before S19 either, and the dim is part of the index identity
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return _vectors(request, [1.0, 2.0])

    OllamaEmbedder(EmbeddingConfig(dimensions=2), _client(handler)).embed_query("q")
    assert bodies == [{"model": "bge-m3:567m", "input": ["q"], "encoding_format": "float"}]


def test_the_query_gets_its_own_prefix() -> None:
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.extend(json.loads(request.content)["input"])
        return _vectors(request)

    e = OllamaEmbedder(EmbeddingConfig(query_prefix="query: "), _client(handler))
    e.embed_query("refund")
    assert seen == ["query: refund"]


def test_parallel_batches_still_come_back_in_order() -> None:
    e = OllamaEmbedder(EmbeddingConfig(batch_size=1, max_concurrency=4), _client(_vectors))
    texts = ["x" * n for n in range(1, 9)]
    assert [v[0] for v in e.embed_documents(texts)] == [float(n) for n in range(1, 9)]


def test_overload_is_retried_on_the_same_model_then_succeeds() -> None:
    models: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        models.append(json.loads(request.content)["model"])
        return _status(503) if len(models) < 3 else _vectors(request)

    e = OllamaEmbedder(EmbeddingConfig(max_retries=4), _client(handler))
    assert e.embed_query("q") == [1.0, 1.0]
    assert models == ["bge-m3:567m"] * 3


def test_exhausted_retries_raise_instead_of_returning_nothing() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return _status(500)

    e = OllamaEmbedder(EmbeddingConfig(max_retries=2), _client(handler))
    with pytest.raises(EmbeddingError, match=r"role embed \(ollama:bge-m3:567m\) failed"):
        e.embed_documents(["a"])
    assert calls == 3


def test_a_client_error_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return _status(404)

    e = OllamaEmbedder(EmbeddingConfig(max_retries=4), _client(handler))
    with pytest.raises(EmbeddingError, match="failed"):
        e.embed_query("q")
    assert calls == 1


def test_a_batch_too_large_is_halved() -> None:
    sizes: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        n = len(json.loads(request.content)["input"])
        sizes.append(n)
        return _status(413) if n > 1 else _vectors(request)

    e = OllamaEmbedder(EmbeddingConfig(batch_size=4), _client(handler))
    assert [v[0] for v in e.embed_documents(["a", "bb", "ccc"])] == [1.0, 2.0, 3.0]
    assert sizes == [3, 1, 2, 1, 1]


def test_a_wrong_vector_count_is_an_error() -> None:
    def one(request: httpx2.Request) -> httpx2.Response:
        data = [{"object": "embedding", "index": 0, "embedding": [1.0]}]
        return httpx2.Response(200, json={"object": "list", "data": data, "model": "bge-m3:567m"})

    e = OllamaEmbedder(EmbeddingConfig(), _client(one))
    with pytest.raises(EmbeddingError, match="asked for 2 vectors, got 1"):
        e.embed_documents(["a", "b"])


def test_dim_is_learned_once_from_the_model() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return _vectors(request)

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
        assert body == {
            "model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 2,
            "encoding_format": "float",
        }  # fmt: skip
        data = [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
        return httpx2.Response(200, json={"object": "list", "data": data, "model": body["model"]})

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


def test_openai_compatible_without_a_key_sends_only_a_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # S19: through lite-model the OpenAI SDK always sends a bearer; a keyless endpoint (TEI,
    # vLLM) gets lite-model's placeholder and ignores it. Never a key from the environment.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == "Bearer unused"
        return _vectors(request, [1.0])

    cfg = EmbeddingConfig(provider="openai", model="m", base_url="http://tei:8080/v1")
    assert OpenAICompatEmbedder(cfg, _client(handler)).embed_query("q") == [1.0]


def test_the_index_identity_is_what_it_was_before_s19() -> None:
    # kbs records provider, model, dim and both prefixes (rule 3): moving the calls onto
    # lite-model must not change any of them, or every existing kb would refuse to open
    from lite_rag.embedding import build_embedder, model_id

    e = build_embedder(EmbeddingConfig())
    assert (e.provider, e.model, model_id(e.provider, e.model)) == (
        "ollama", "bge-m3:567m", "ollama/bge-m3:567m",
    )  # fmt: skip
    nomic = build_embedder(EmbeddingConfig(model="nomic-embed-text"))
    assert (nomic.prefixes.query, nomic.prefixes.document) == (
        "search_query: ",
        "search_document: ",
    )
    keyed = build_embedder(EmbeddingConfig(provider="openai", model="text-embedding-3-small"))
    assert model_id(keyed.provider, keyed.model) == "openai/text-embedding-3-small"
