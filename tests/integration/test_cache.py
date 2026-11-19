"""lite-cache's L1 in lite-rag (S18; lite-cache design §9): the runtime's own lite-model on a
recording transport, the cache in schema `lite_rag_cache_test` of the test database."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import httpx2
import psycopg
import pytest
from psycopg import sql

from lite_rag.config import Settings, load_settings
from lite_rag.embedding import CachingEmbedder
from lite_rag.engine import RagEngine, RagRuntime
from lite_rag.errors import ConfigError, LiteRagError
from lite_rag.llm import build_model
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeEmbedder
from tests.chat import chat_completion
from tests.conftest import TEST_SCHEMA
from tests.integration.support import CHUNKING, NS

CACHE_SCHEMA = "lite_rag_cache_test"
# Four paragraphs, each too long to share a 40-character chunk: four index calls per role, so
# `contextual.concurrency` 4 really runs them in worker threads.
TEXT = "\n\n".join(
    [
        "手冲咖啡的粉水比通常是一比十六，偏浓可以调到一比十五左右。",
        "水温在九十二度左右，浅烘豆可以略高，深烘豆可以略低一些。",
        "研磨度接近细砂糖，萃取时间控制在两分半到三分钟之间为宜。",
        "闷蒸三十秒让咖啡粉充分排气，之后分三次缓慢注水完成冲煮。",
    ]
)
QUESTION = "手冲咖啡的粉水比是多少？"
SECRET = "k7Qp2Lx9Vd4Rm8Tz1Wc6Yb3Nh5Gf0Jas"  # a test value, as `openssl rand` would give


class Endpoint:
    """An OpenAI-compatible endpoint that answers by role: graph (a JSON schema), contexts
    (the index model) and answers. Every request is recorded with its kind."""

    REPLIES: ClassVar[dict[str, str]] = {
        "graph": '{"entities": [{"name": "咖啡", "type": "drink"}], "relations": []}',
        "answer": "粉水比是 1:16 [1]。",
        "context": "这段讲咖啡。",
    }

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.kinds: list[str] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            kind = "graph"
        else:
            kind = "answer" if body["model"] == "qwen3:8b" else "context"
        self.requests.append(body)
        self.kinds.append(kind)
        return httpx2.Response(
            200, json=chat_completion(body["model"], self.REPLIES[kind], usage=(10, 5))
        )

    def count(self, kind: str) -> int:
        """Requests of one kind: graph, context or answer."""
        return self.kinds.count(kind)


@pytest.fixture(autouse=True)
def prompt_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITERAG_PROMPT_SECRET", SECRET)


@pytest.fixture
def cache_schema(pg: psycopg.Connection) -> Iterator[str]:
    drop = sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(CACHE_SCHEMA))
    pg.execute(drop)
    yield CACHE_SCHEMA
    pg.execute(drop)


def settings(pg_url: str, *, enabled: bool = True, **cache: Any) -> Settings:
    return load_settings(
        database_url=pg_url,
        chunking={**CHUNKING, "contextual": {"enabled": True, **cache.pop("contextual", {})}},
        graph={"enabled": True},
        cache={"enabled": enabled, "schema": CACHE_SCHEMA, **cache},
    )


RUNTIMES: list[RagRuntime] = []


@pytest.fixture(autouse=True)
def _close_runtimes() -> Iterator[None]:
    yield
    while RUNTIMES:
        RUNTIMES.pop().close()  # its cache store and hook; closing the store twice is harmless


def engine(
    store: PgStore, cfg: Settings, endpoint: Endpoint, *, model: str = "fake-a", ns: str = NS
) -> RagEngine:
    """An engine on the runtime's own lite-model (recording transport); its cache is closed after
    the test (the store fixture closes the store)."""
    lite = build_model(cfg.llm, http_client=httpx2.Client(transport=httpx2.MockTransport(endpoint)))
    embedder = CachingEmbedder(FakeEmbedder(dim=8, model=model), store)
    runtime = RagRuntime(cfg, store, embedder, model=lite)
    RUNTIMES.append(runtime)
    return runtime.engine(ns)


def contexts(pg: psycopg.Connection, kb: str) -> list[str]:
    query = sql.SQL("SELECT context FROM {}.chunks WHERE kb = %s ORDER BY n").format(
        sql.Identifier(TEST_SCHEMA)
    )
    return [row[0] for row in pg.execute(query, (kb,))]


def mentions(pg: psycopg.Connection, kb: str) -> list[tuple[str, str, str]]:
    query = sql.SQL(
        "SELECT chunk_id, name, type FROM {}.graph_mentions WHERE kb = %s ORDER BY chunk_id, entity"
    ).format(sql.Identifier(TEST_SCHEMA))
    return [(c, n, t) for c, n, t in pg.execute(query, (kb,))]


def schema_exists(pg: psycopg.Connection, name: str) -> bool:
    row = pg.execute("SELECT to_regnamespace(%s)", (name,)).fetchone()
    return row is not None and row[0] is not None


def test_off_is_unchanged(
    store: PgStore, pg: psycopg.Connection, pg_url: str, cache_schema: str
) -> None:
    endpoint = Endpoint()
    off = engine(store, settings(pg_url, enabled=False), endpoint)
    off.upsert_text("kb", "d", "Coffee", TEXT)
    first, second = off.ask("kb", QUESTION), off.ask("kb", QUESTION)
    assert endpoint.count("answer") == 2
    assert (first.cached, second.cached) == (False, False)
    assert not schema_exists(pg, cache_schema)
    on = engine(store, settings(pg_url), Endpoint())
    assert on.kb_info("kb") == off.kb_info("kb")  # the cache is not index identity


def test_reindex_into_a_new_kb_reuses_contexts_and_graph(
    store: PgStore, pg: psycopg.Connection, pg_url: str, cache_schema: str
) -> None:
    endpoint = Endpoint()
    cfg = settings(pg_url)
    engine(store, cfg, endpoint, model="fake-a").upsert_text("a", "d", "Coffee", TEXT)
    built = (endpoint.count("context"), endpoint.count("graph"))
    assert built[0] > 0 and built[1] > 0
    # Another embedding model: a new kb, the same chunks and so the same index-role requests.
    engine(store, cfg, endpoint, model="fake-b").reindex("a", "b")
    assert (endpoint.count("context"), endpoint.count("graph")) == built
    a, b = (contexts(pg, f"{NS}:{kb}") for kb in ("a", "b"))
    assert a == b and all(a)
    graph_a, graph_b = (mentions(pg, f"{NS}:{kb}") for kb in ("a", "b"))
    assert graph_a == graph_b and graph_a  # the graph, word for word


def test_a_repeated_question_is_answered_from_the_cache(
    store: PgStore, pg_url: str, cache_schema: str
) -> None:
    endpoint = Endpoint()
    rag = engine(store, settings(pg_url), endpoint)
    rag.upsert_text("kb", "d", "Coffee", TEXT)
    first, second = rag.ask("kb", QUESTION), rag.ask("kb", QUESTION)
    assert endpoint.count("answer") == 1
    assert (first.cached, second.cached) == (False, True)
    assert (second.text, second.citations, second.model_id) == (
        first.text,
        first.citations,
        first.model_id,
    )


def test_two_namespaces_never_share(store: PgStore, pg_url: str, cache_schema: str) -> None:
    endpoint = Endpoint()
    for ns in ("proj/t1", "proj/t2"):
        rag = engine(store, settings(pg_url), endpoint, ns=ns)
        rag.upsert_text("kb", "d", "Coffee", TEXT)
        assert rag.ask("kb", QUESTION).cached is False
    assert endpoint.count("answer") == 2


def test_contextual_concurrency_four_still_hits(
    store: PgStore, pg_url: str, cache_schema: str
) -> None:
    endpoint = Endpoint()
    cfg = settings(pg_url, contextual={"concurrency": 4})
    engine(store, cfg, endpoint, model="fake-a").upsert_text("a", "d", "Coffee", TEXT)
    built = endpoint.count("context")
    assert built >= 4  # enough items for the worker threads to be used
    engine(store, cfg, endpoint, model="fake-b").reindex("a", "b")
    assert endpoint.count("context") == built


def test_roles_not_listed_are_not_cached(store: PgStore, pg_url: str, cache_schema: str) -> None:
    endpoint = Endpoint()
    cfg = settings(pg_url, roles=["answer"])
    engine(store, cfg, endpoint, model="fake-a").upsert_text("a", "d", "Coffee", TEXT)
    built = endpoint.count("context")
    rag = engine(store, cfg, endpoint, model="fake-b")
    rag.reindex("a", "b")
    assert endpoint.count("context") == 2 * built  # contexts called again
    rag.ask("b", QUESTION)
    assert rag.ask("b", QUESTION).cached is True  # answers cached


def test_a_cache_read_failure_is_a_lite_rag_error(
    store: PgStore, pg: psycopg.Connection, pg_url: str, cache_schema: str
) -> None:
    endpoint = Endpoint()
    rag = engine(store, settings(pg_url), endpoint)
    rag.upsert_text("kb", "d", "Coffee", TEXT)
    pg.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cache_schema)))
    sent = len(endpoint.requests)
    with pytest.raises(LiteRagError):
        rag.ask("kb", QUESTION)
    assert len(endpoint.requests) == sent  # nothing sent: no silent fallback


def test_the_cache_needs_a_prompt_secret(
    store: PgStore, pg_url: str, cache_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without it the source-block nonce stays random: no request would ever repeat (design §6).
    monkeypatch.delenv("LITERAG_PROMPT_SECRET")
    endpoint = Endpoint()
    with pytest.raises(ConfigError, match="LITERAG_PROMPT_SECRET"):
        engine(store, settings(pg_url), endpoint).upsert_text("kb", "d", "Coffee", TEXT)
    assert endpoint.requests == []


def test_a_short_prompt_secret_is_refused(
    store: PgStore, pg_url: str, cache_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITERAG_PROMPT_SECRET", "short")
    with pytest.raises(ConfigError, match="32"):
        engine(store, settings(pg_url), Endpoint()).upsert_text("kb", "d", "Coffee", TEXT)


def test_the_question_log_is_off_unless_set(
    store: PgStore, pg_url: str, cache_schema: str, tmp_path: Path
) -> None:
    rag = engine(store, settings(pg_url), Endpoint())
    rag.upsert_text("kb", "d", "Coffee", TEXT)
    rag.ask("kb", QUESTION)
    assert list(tmp_path.iterdir()) == []


def test_the_question_log_has_one_json_line_per_question(
    store: PgStore, pg_url: str, cache_schema: str, tmp_path: Path
) -> None:
    log = tmp_path / "new-dir" / "questions.jsonl"
    rag = engine(store, settings(pg_url, question_log=str(log)), Endpoint())
    rag.upsert_text("kb", "d", "Coffee", TEXT)
    odd = "粉水比\n是多少？"
    rag.ask("kb", QUESTION)
    rag.ask("kb", odd)
    lines = log.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["question"] for r in records] == [QUESTION, odd]
    assert {r["scope"] for r in records} == {f"literag/{NS}"}
    assert all(r["ts"].endswith("+00:00") for r in records)


def test_a_missing_secret_fails_at_startup(
    pg_url: str, cache_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Like the reranker: `literag serve` must not start healthy and fail on the first request.
    monkeypatch.delenv("LITERAG_PROMPT_SECRET")
    cfg = settings(pg_url).model_copy(
        update={"store": settings(pg_url).store.model_copy(update={"schema_name": TEST_SCHEMA})}
    )
    with pytest.raises(ConfigError, match="LITERAG_PROMPT_SECRET"):
        RagRuntime.from_settings(cfg)


def test_closing_the_runtime_unregisters_the_hook(
    store: PgStore, pg_url: str, cache_schema: str
) -> None:
    endpoint = Endpoint()
    lite = build_model(
        settings(pg_url).llm, http_client=httpx2.Client(transport=httpx2.MockTransport(endpoint))
    )
    runtime = RagRuntime(
        settings(pg_url), store, CachingEmbedder(FakeEmbedder(dim=8), store), model=lite
    )
    runtime.llm()  # builds the role, opens the cache, registers the hook
    assert lite.hooks["lite-cache"] is not None
    runtime.close()
    with pytest.raises(KeyError):  # the model may outlive the runtime
        lite.hooks["lite-cache"]


@pytest.mark.parametrize("weak", ["a" * 32, "ab" * 20, "0123456789" * 4])
def test_an_obviously_weak_prompt_secret_is_refused(
    store: PgStore, pg_url: str, cache_schema: str, monkeypatch: pytest.MonkeyPatch, weak: str
) -> None:
    # Length alone does not make a secret unpredictable (Codex S18): refuse the obvious ones.
    monkeypatch.setenv("LITERAG_PROMPT_SECRET", weak)
    with pytest.raises(ConfigError, match="random"):
        engine(store, settings(pg_url), Endpoint()).upsert_text("kb", "d", "Coffee", TEXT)


def test_a_hook_name_already_taken_is_a_lite_rag_config_error(
    store: PgStore, pg_url: str, cache_schema: str
) -> None:
    class Other:
        name = "lite-cache"

        def before(self, call: Any) -> None: ...

        def after(self, call: Any, reply: Any) -> None: ...

    lite = build_model(
        settings(pg_url).llm, http_client=httpx2.Client(transport=httpx2.MockTransport(Endpoint()))
    )
    lite.hooks.register(Other())  # e.g. the host registered its own CacheHook
    runtime = RagRuntime(
        settings(pg_url), store, CachingEmbedder(FakeEmbedder(dim=8), store), model=lite
    )
    try:
        with pytest.raises(ConfigError, match="lite-cache"):
            runtime.llm()
    finally:
        runtime.close()


def test_a_cache_read_failure_during_ingest_is_a_lite_rag_error(
    store: PgStore, pg: psycopg.Connection, pg_url: str, cache_schema: str
) -> None:
    endpoint = Endpoint()
    rag = engine(store, settings(pg_url, contextual={"concurrency": 4}), endpoint)
    rag.upsert_text("warm", "d0", "Coffee", "预热：让缓存的 schema 建好。")  # opens the cache
    pg.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cache_schema)))
    sent = len(endpoint.requests)
    with pytest.raises(LiteRagError):
        rag.upsert_text("kb", "d", "Coffee", TEXT)  # contexts in map workers read the cache
    assert len(endpoint.requests) == sent
