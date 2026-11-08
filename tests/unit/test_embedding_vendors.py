"""Hosted embedders (S5c). Request and response shapes follow each vendor's API reference
(checked 2026-11-08): Voyage `POST /v1/embeddings`, Cohere `POST /v2/embed`, Gemini
`POST /v1beta/models/{model}:batchEmbedContents`. No test reaches a real service."""

import json
from collections.abc import Callable

import httpx2
import pytest

from lite_rag.config import EmbeddingConfig
from lite_rag.embedding import CohereEmbedder, GeminiEmbedder, VoyageEmbedder, build_embedder
from lite_rag.errors import ConfigError

Handler = Callable[[httpx2.Request], httpx2.Response]


def _client(handler: Handler) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    monkeypatch.setenv("CO_API_KEY", "co-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")


def test_voyage_sends_the_input_type_per_role_and_reorders_by_index() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url == "https://api.voyageai.com/v1/embeddings"
        assert request.headers["authorization"] == "Bearer pa-test"
        body = json.loads(request.content)
        seen.append(body)
        data = [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(body["input"]))]
        return httpx2.Response(200, json={"data": data[::-1], "usage": {"total_tokens": 3}})

    cfg = EmbeddingConfig(provider="voyage", model="voyage-4", dimensions=2)
    e = VoyageEmbedder(cfg, _client(handler))
    assert e.embed_documents(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    e.embed_query("q")
    assert seen == [
        {"model": "voyage-4", "input": ["a", "b"], "input_type": "document", "output_dimension": 2},
        {"model": "voyage-4", "input": ["q"], "input_type": "query", "output_dimension": 2},
    ]
    assert (e.prefixes.query, e.prefixes.document) == ("", "")  # the API adds its own


def test_cohere_v2_asks_for_float_embeddings_by_role() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url == "https://api.cohere.com/v2/embed"
        assert request.headers["authorization"] == "Bearer co-test"
        body = json.loads(request.content)
        seen.append(body)
        return httpx2.Response(
            200, json={"embeddings": {"float": [[1.0, 0.0]] * len(body["texts"])}}
        )

    e = CohereEmbedder(EmbeddingConfig(provider="cohere", model="embed-v4.0"), _client(handler))
    assert e.embed_documents(["a"]) == [[1.0, 0.0]]
    e.embed_query("q")
    assert [b["input_type"] for b in seen] == ["search_document", "search_query"]
    assert all(b["embedding_types"] == ["float"] and b["model"] == "embed-v4.0" for b in seen)


def test_cohere_batches_stay_within_its_96_texts() -> None:
    sizes: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["texts"]
        sizes.append(len(texts))
        return httpx2.Response(200, json={"embeddings": {"float": [[1.0]] * len(texts)}})

    cfg = EmbeddingConfig(provider="cohere", model="embed-v4.0", batch_size=500, max_concurrency=1)
    CohereEmbedder(cfg, _client(handler)).embed_documents(["t"] * 200)
    assert sizes == [96, 96, 8]


def test_gemini_001_sends_a_task_type_per_text() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001"
        assert request.url == f"{url}:batchEmbedContents"
        assert request.headers["x-goog-api-key"] == "gm-test"
        body = json.loads(request.content)
        seen.append(body)
        return httpx2.Response(
            200, json={"embeddings": [{"values": [1.0, 2.0]} for _ in body["requests"]]}
        )

    cfg = EmbeddingConfig(provider="gemini", model="gemini-embedding-001", dimensions=2)
    e = GeminiEmbedder(cfg, _client(handler))
    assert e.embed_documents(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
    e.embed_query("q")
    first = seen[0]["requests"][0]  # type: ignore[index]
    assert first == {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": "a"}]},
        "taskType": "RETRIEVAL_DOCUMENT",
        "outputDimensionality": 2,
    }
    assert seen[1]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"  # type: ignore[index]


def test_gemini_2_takes_its_task_as_a_text_prefix_one_content_per_text() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx2.Response(
            200, json={"embeddings": [{"values": [1.0]} for _ in body["requests"]]}
        )

    e = GeminiEmbedder(
        EmbeddingConfig(provider="gemini", model="gemini-embedding-2"), _client(handler)
    )
    e.embed_documents(["a", "b"])
    e.embed_query("q")
    docs, query = seen
    # one Content per text: several parts in one Content would be merged into one vector
    assert [r["content"] for r in docs["requests"]] == [  # type: ignore[index]
        {"parts": [{"text": "title: none | text: a"}]},
        {"parts": [{"text": "title: none | text: b"}]},
    ]
    assert query["requests"][0]["content"] == {  # type: ignore[index]
        "parts": [{"text": "task: search result | query: q"}]
    }
    assert all("taskType" not in r for b in seen for r in b["requests"])  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("provider", "env"),
    [("voyage", "VOYAGE_API_KEY"), ("cohere", "CO_API_KEY"), ("gemini", "GEMINI_API_KEY")],
)
def test_a_hosted_embedder_needs_its_key(
    provider: str, env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(env)
    with pytest.raises(ConfigError, match=env):
        build_embedder(EmbeddingConfig(provider=provider, model="m"))  # type: ignore[arg-type]


def test_each_provider_defaults_to_its_own_endpoint() -> None:
    from lite_rag.embedding import OllamaEmbedder, OpenAICompatEmbedder

    assert OllamaEmbedder(EmbeddingConfig())._base_url == "http://127.0.0.1:11434"
    assert OpenAICompatEmbedder(EmbeddingConfig(provider="openai"))._base_url == (
        "https://api.openai.com/v1"
    )
    assert VoyageEmbedder(EmbeddingConfig(provider="voyage"))._base_url == (
        "https://api.voyageai.com/v1"
    )
    proxy = EmbeddingConfig(provider="voyage", base_url="https://proxy/v1/")
    assert VoyageEmbedder(proxy)._base_url == "https://proxy/v1"


def test_a_gemini_model_without_a_prefix_profile_gets_a_task_type() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx2.Response(
            200, json={"embeddings": [{"values": [1.0]} for _ in body["requests"]]}
        )

    cfg = EmbeddingConfig(provider="gemini", model="text-embedding-004")
    GeminiEmbedder(cfg, _client(handler)).embed_query("q")
    assert seen[0]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"  # type: ignore[index]


def test_every_configurable_provider_has_an_embedder() -> None:
    from typing import get_args

    from lite_rag.embedding import EMBEDDERS

    provider = EmbeddingConfig.model_fields["provider"].annotation
    assert set(get_args(provider)) == set(EMBEDDERS)


class _FakeST:
    """Stands in for a SentenceTransformer model: records what it was asked to encode."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> list[object]:
        import numpy as np

        self.calls.append((list(texts), kwargs))
        return [np.array([float(len(t)), 0.0]) for t in texts]

    def get_embedding_dimension(self) -> int:
        return 2


def test_sentence_transformers_applies_the_model_prefixes_and_normalises() -> None:
    from lite_rag.embedding import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder(
        EmbeddingConfig(provider="sentence_transformers", model="intfloat/multilingual-e5-base")
    )
    fake = _FakeST()
    e.__dict__["_impl"] = fake  # the model boundary: no download in unit tests
    assert e.embed_documents(["a"]) == [[len("passage: a"), 0.0]]
    e.embed_query("q")
    assert [texts for texts, _ in fake.calls] == [["passage: a"], ["query: q"]]
    assert all(kw["normalize_embeddings"] is True for _, kw in fake.calls)
    assert e.dim == 2 and e.embed_documents([]) == []


def test_a_batch_over_the_token_limit_is_split_until_it_fits() -> None:
    sizes: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["input"]
        sizes.append(len(texts))
        if len(texts) > 2:  # Voyage also caps total tokens per request (4xx, not retryable)
            return httpx2.Response(400, json={"detail": "max allowed tokens per batch exceeded"})
        data = [{"index": i, "embedding": [float(t)]} for i, t in enumerate(texts)]
        return httpx2.Response(200, json={"data": data})

    cfg = EmbeddingConfig(provider="voyage", model="voyage-4", batch_size=5, max_concurrency=1)
    vectors = VoyageEmbedder(cfg, _client(handler)).embed_documents(["1", "2", "3", "4", "5"])
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]  # every text once, in order
    assert sizes == [5, 2, 3, 1, 2]  # 5 refused → 2 + 3; 3 refused → 1 + 2


def test_a_single_text_that_is_refused_is_an_error_not_a_loop() -> None:
    from lite_rag.embedding import EmbeddingError

    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(len(json.loads(request.content)["input"]))
        return httpx2.Response(400, json={"detail": "bad input"})

    cfg = EmbeddingConfig(provider="voyage", model="voyage-4", batch_size=4, max_concurrency=1)
    with pytest.raises(EmbeddingError, match="400"):
        VoyageEmbedder(cfg, _client(handler)).embed_documents(["a", "b", "c", "d"])
    assert len(calls) <= 7 and calls[-1] == 1  # halved down to one text, then given up


def test_other_client_errors_are_not_split() -> None:
    from lite_rag.embedding import EmbeddingError

    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(len(json.loads(request.content)["input"]))
        return httpx2.Response(401, json={"detail": "invalid key"})

    cfg = EmbeddingConfig(provider="voyage", model="voyage-4", batch_size=4, max_concurrency=1)
    with pytest.raises(EmbeddingError, match="401"):
        VoyageEmbedder(cfg, _client(handler)).embed_documents(["a", "b", "c", "d"])
    assert calls == [4]  # a bad key is not a size problem: one call, then the error
