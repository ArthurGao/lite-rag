"""The Voyage reranker (S20): Voyage AI's `/v1/rerank` through lite-model's `rerank()` (its 0.7.0).
Request and reply shapes follow Voyage's API reference (2025-12-10); no test reaches the service."""

import json

import httpx2
import pytest

from lite_rag.config import RerankConfig, Settings
from lite_rag.errors import ConfigError
from lite_rag.rerank import RerankError, VoyageReranker, build_reranker


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")


def _client(seen: list[dict[str, object]], status: int = 200) -> httpx2.Client:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == "https://api.voyageai.com/v1/rerank"
        assert request.headers["authorization"] == "Bearer pa-test"
        body = json.loads(request.content)
        seen.append(body)
        n = len(body["documents"])
        data = sorted(
            ({"index": i, "relevance_score": i / 10} for i in range(n)),
            key=lambda d: -d["relevance_score"],
        )  # Voyage sorts by relevance; the scores must come back in input order
        return httpx2.Response(
            status, json={"object": "list", "data": data, "model": "rerank-2.5", "usage": {}}
        )

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_voyage_scores_every_text_in_input_order() -> None:
    seen: list[dict[str, object]] = []
    reranker = VoyageReranker(RerankConfig(provider="voyage"), _client(seen))
    assert reranker.score("refund?", ["a", "b", "c"]) == [0.0, 0.1, 0.2]
    assert seen == [
        {"query": "refund?", "documents": ["a", "b", "c"], "model": "rerank-2.5",
         "return_documents": False}
    ]  # fmt: skip
    assert reranker.model_id == "voyage/rerank-2.5"


def test_the_model_and_key_variable_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEAM_VOYAGE", "pa-test")
    config = RerankConfig(provider="voyage", model="rerank-2.5-lite", api_key_env="TEAM_VOYAGE")
    assert build_reranker(config).model_id == "voyage/rerank-2.5-lite"


def test_a_missing_key_fails_at_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY")
    with pytest.raises(ConfigError, match="VOYAGE_API_KEY"):
        build_reranker(RerankConfig(provider="voyage"))


def test_a_failed_call_raises_without_the_apis_text() -> None:
    reranker = VoyageReranker(RerankConfig(provider="voyage", max_retries=0), _client([], 401))
    with pytest.raises(RerankError, match="AuthenticationError"):
        reranker.score("q", ["a"])


def test_the_fastembed_model_is_not_sent_to_voyage() -> None:
    # the default model is fastembed's; voyage gets its own unless one is named
    assert RerankConfig(provider="voyage").model == "rerank-2.5"
    assert RerankConfig().model == "BAAI/bge-reranker-v2-m3-int8"
    with pytest.raises(ValueError, match="voyage"):
        RerankConfig(provider="fastembed", api_key_env="X")
    with pytest.raises(ValueError, match="cache_dir"):  # Codex S20: fastembed's own field
        RerankConfig(provider="voyage", cache_dir="/tmp/models")


def test_settings_take_the_voyage_reranker() -> None:
    settings = Settings.model_validate({"rerank": {"enabled": True, "provider": "voyage"}})
    assert settings.rerank.provider == "voyage"
