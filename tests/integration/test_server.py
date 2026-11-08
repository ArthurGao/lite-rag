"""The HTTP service over a real store with a fake embedder (design §7, S6 acceptance)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lite_rag.config import ApiKeyConfig, load_settings
from lite_rag.server import create_app
from lite_rag.server.auth import new_key
from lite_rag.stores.pg import PgStore
from tests.integration.support import CHUNKING, make_engine, make_runtime

NS = "steward/1"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def keys() -> dict[str, str]:
    return {name: new_key()[0] for name in ("writer", "reader", "other")}


@pytest.fixture
def client(store: PgStore, keys: dict[str, str]) -> Iterator[TestClient]:
    from lite_rag.server.auth import key_hash

    scopes = {
        "writer": (["steward/*"], ["*"], "write"),
        "reader": (["steward/*"], ["c-*"], "read"),
        "other": (["tenderted/*"], ["*"], "write"),
    }
    entries = [
        ApiKeyConfig(name=n, sha256=key_hash(k), namespaces=ns, kbs=kbs, mode=mode)  # type: ignore[arg-type]
        for n, k in keys.items()
        for ns, kbs, mode in [scopes[n]]
    ]
    settings = load_settings(chunking=CHUNKING, server={"keys": [e.model_dump() for e in entries]})
    engine = make_runtime(store)
    with TestClient(create_app(settings, engine)) as c:
        yield c


def _h(key: str, namespace: str = NS) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "X-LiteRag-Namespace": namespace}


def _fill(client: TestClient, key: str, kb: str = "c-1") -> None:
    for doc_id, text in {"refund": "Refunds take seven days.", "ship": "Shipping is free."}.items():
        r = client.put(
            f"/v1/kbs/{kb}/documents/{doc_id}",
            headers=_h(key),
            json={"title": doc_id, "text": text},
        )
        assert r.status_code == 200, r.text


def test_health_needs_no_key(client: TestClient) -> None:
    r = client.get("/health")
    assert (r.status_code, r.json()) == (200, {"status": "ok"})


def test_401_without_a_valid_key_as_problem_details(client: TestClient) -> None:
    for headers in ({"X-LiteRag-Namespace": NS}, _h("lrk_nope_nope")):
        r = client.post("/v1/kbs/c-1/search", headers=headers, json={"query": "refunds"})
        assert r.status_code == 401
        assert r.headers["content-type"] == "application/problem+json"
        assert r.headers["www-authenticate"] == "Bearer"
        assert r.json()["status"] == 401 and r.json()["title"] == "Unauthorized"


def test_403_outside_the_keys_scope(client: TestClient, keys: dict[str, str]) -> None:
    _fill(client, keys["writer"])
    cases = [
        (keys["other"], NS, "c-1"),  # another project's key: namespace
        (keys["reader"], NS, "faq"),  # kb outside its globs
    ]
    for key, ns, kb in cases:
        r = client.post(f"/v1/kbs/{kb}/search", headers=_h(key, ns), json={"query": "refunds"})
        assert r.status_code == 403, (key, kb)
    r = client.put(
        "/v1/kbs/c-1/documents/x", headers=_h(keys["reader"]), json={"title": "x", "text": "y"}
    )
    assert r.status_code == 403 and "read-only" in r.json()["detail"]
    r = client.get("/v1/kbs", headers=_h(keys["other"]))  # listing is checked too
    assert r.status_code == 403


def test_the_namespace_header_is_required(client: TestClient, keys: dict[str, str]) -> None:
    r = client.post(
        "/v1/kbs/c-1/search",
        headers={"Authorization": f"Bearer {keys['writer']}"},
        json={"query": "q"},
    )
    assert r.status_code == 400 and "X-LiteRag-Namespace" in r.json()["detail"]


def test_api_and_library_return_the_same_hits(
    client: TestClient, keys: dict[str, str], store: PgStore
) -> None:
    _fill(client, keys["writer"])
    body = {"query": "refunds seven days", "top_k": 2, "mode": "hybrid"}
    api = client.post("/v1/kbs/c-1/search", headers=_h(keys["reader"]), json=body).json()["hits"]
    lib = make_engine(store, namespace=NS).search("c-1", body["query"], top_k=2, mode="hybrid")
    assert [(h["chunk_id"], h["score"], h["ranks"]) for h in api] == [
        (h.chunk_id, h.score, h.ranks) for h in lib
    ]


def test_kbs_are_created_listed_inspected_and_deleted(
    client: TestClient, keys: dict[str, str]
) -> None:
    w = _h(keys["writer"])
    created = client.put("/v1/kbs/c-2", headers=w)
    assert created.status_code == 200 and created.json()["kb"] == "c-2"
    _fill(client, keys["writer"], "c-1")
    assert client.get("/v1/kbs", headers=w).json()["kbs"] == ["c-1", "c-2"]
    assert client.get("/v1/kbs", headers=_h(keys["writer"], "steward/2")).json()["kbs"] == []
    stats = client.get("/v1/kbs/c-1", headers=w).json()
    assert (stats["documents"], stats["namespace"]) == (2, NS)
    assert client.delete("/v1/kbs/c-1/documents/ship", headers=w).status_code == 204
    assert client.delete("/v1/kbs/c-1/documents/ship", headers=w).status_code == 404
    assert client.delete("/v1/kbs/c-1", headers=w).status_code == 204
    r = client.post("/v1/kbs/c-1/search", headers=w, json={"query": "refunds"})
    assert r.status_code == 404 and r.json()["title"] == "KbNotFound"


def test_a_kb_built_differently_is_409(
    client: TestClient, keys: dict[str, str], store: PgStore
) -> None:
    make_engine(store, namespace=NS, size=17).upsert_text("c-9", "d", "D", "Text here.")
    r = client.post("/v1/kbs/c-9/search", headers=_h(keys["writer"]), json={"query": "text"})
    assert r.status_code == 409 and "built differently" in r.json()["detail"]


def test_a_file_upload_is_ingested_with_its_pages(client: TestClient, keys: dict[str, str]) -> None:
    w = _h(keys["writer"])
    with (FIXTURES / "pages.pdf").open("rb") as f:
        r = client.post(
            "/v1/kbs/c-1/files", headers=w, files={"file": ("handbook.pdf", f, "application/pdf")}
        )
    assert r.status_code == 200 and r.json()["doc_id"].startswith("file:")
    hits = client.post(
        "/v1/kbs/c-1/search", headers=w, json={"query": "fifty dollars", "mode": "bm25"}
    ).json()["hits"]
    assert hits[0]["page"] == 3 and hits[0]["title"] == "Refund Handbook"


def test_an_invalid_body_is_422_problem(client: TestClient, keys: dict[str, str]) -> None:
    r = client.post(
        "/v1/kbs/c-1/search", headers=_h(keys["writer"]), json={"query": "", "top_k": 0}
    )
    assert r.status_code == 422 and r.headers["content-type"] == "application/problem+json"
    assert "query" in r.json()["detail"] and "top_k" in r.json()["detail"]


def test_ask_answers_through_the_api(store: PgStore, keys: dict[str, str]) -> None:
    from lite_rag.testing import FakeLLM

    settings = load_settings(chunking=CHUNKING)  # no keys: open (loopback only, see serve)
    engine = make_runtime(store, llm=FakeLLM(reply="Seven days [1]."))
    with TestClient(create_app(settings, engine), base_url="http://localhost") as c:  # open mode
        h = {"X-LiteRag-Namespace": NS}
        c.put(
            "/v1/kbs/c-1/documents/refund",
            headers=h,
            json={"title": "R", "text": "Refunds take seven days."},
        )
        answer = c.post(
            "/v1/kbs/c-1/ask", headers=h, json={"question": "How long do refunds take?"}
        ).json()
    assert answer["status"] == "answered" and answer["citations"][0]["doc_id"] == "refund"


def test_an_upstream_failure_is_reported_without_its_detail(
    client: TestClient, keys: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from lite_rag.engine import RagEngine
    from lite_rag.errors import StoreError

    def broken(*args: object, **kwargs: object) -> None:
        raise StoreError("connection to postgresql://u:s3cret@db failed")

    monkeypatch.setattr(RagEngine, "search", broken)
    r = client.post("/v1/kbs/c-1/search", headers=_h(keys["writer"]), json={"query": "q"})
    assert r.status_code == 503 and r.json() == {
        "type": "about:blank",
        "title": "StoreError",
        "status": 503,
    }
    assert "s3cret" not in r.text  # logged on the server, never returned


def test_an_upload_over_the_limit_is_refused_before_it_is_parsed(
    store: PgStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lite_rag.engine import RagEngine

    reached: list[str] = []
    monkeypatch.setattr(RagEngine, "ingest", lambda *a, **k: reached.append("ingest"))
    settings = load_settings(chunking=CHUNKING, server={"max_upload_mb": 0.001})  # ~1 KB
    with TestClient(create_app(settings, make_runtime(store)), base_url="http://localhost") as c:
        big = ("big.md", b"# Big\n\n" + b"x " * 2000, "text/markdown")
        r = c.post("/v1/kbs/c-1/files", headers={"X-LiteRag-Namespace": NS}, files={"file": big})
    assert r.status_code == 413 and r.headers["content-type"] == "application/problem+json"
    assert reached == []  # the body was cut off before the route (or the parser) saw all of it


def test_an_upload_error_never_echoes_the_file_name_or_a_server_path(
    client: TestClient, keys: dict[str, str]
) -> None:
    evil = ("<script>alert(1)</script>.exe", b"MZ...", "application/octet-stream")
    r = client.post("/v1/kbs/c-1/files", headers=_h(keys["writer"]), files={"file": evil})
    assert r.status_code == 400 and "unsupported file type" in r.json()["detail"]
    assert "<script>" not in r.text and "/tmp" not in r.text and "upload" not in r.json()["detail"]


def test_the_same_file_name_replaces_the_same_document(
    client: TestClient, keys: dict[str, str]
) -> None:
    from lite_rag.loaders import doc_id_for

    w = _h(keys["writer"])
    ids = [
        client.post(
            "/v1/kbs/c-1/files", headers=w, files={"file": ("faq.md", body, "text/markdown")}
        ).json()["doc_id"]
        for body in (b"# FAQ\n\nOld answer.", b"# FAQ\n\nNew answer.")
    ]
    assert ids == [doc_id_for("faq.md")] * 2  # from the name, not the temporary path
    assert client.get("/v1/kbs/c-1", headers=w).json()["documents"] == 1


def test_the_body_limit_acts_before_any_route_code(store: PgStore) -> None:
    settings = load_settings(chunking=CHUNKING, server={"max_upload_mb": 0.001})  # ~1 KB
    with TestClient(create_app(settings, make_runtime(store)), base_url="http://localhost") as c:
        big = ("big.md", b"x " * 20_000, "text/markdown")  # 40 KB: over limit + overhead
        r = c.post("/v1/kbs/c-1/files", files={"file": big})  # no namespace header at all
    # 413, not the 400 the route's dependency would give: nothing ran, nothing was parsed
    assert r.status_code == 413 and r.json()["title"] == "TooLarge"


def test_an_open_server_only_answers_to_localhost_host_names(store: PgStore) -> None:
    """Open mode (no keys) relies on binding loopback; a DNS-rebinding page reaches loopback
    with its own host name, so any other Host is refused."""
    settings = load_settings(chunking=CHUNKING)  # no keys
    app = create_app(settings, make_runtime(store))
    with TestClient(app, base_url="http://localhost:8700") as c:
        assert c.get("/health").status_code == 200
        assert c.get("/health", headers={"Host": "127.0.0.1:8700"}).status_code == 200
        assert c.get("/health", headers={"Host": "[::1]:8700"}).status_code == 200
        assert c.get("/health", headers={"Host": "evil.example:8700"}).status_code == 400


def test_a_json_body_over_the_limit_is_refused_before_authentication(
    store: PgStore,
) -> None:
    settings = load_settings(chunking=CHUNKING, server={"max_body_mb": 0.001, "keys": []})
    with TestClient(create_app(settings, make_runtime(store)), base_url="http://localhost") as c:
        r = c.put("/v1/kbs/c-1/documents/big", json={"title": "t", "text": "x" * 50_000})
    assert r.status_code == 413 and r.json()["title"] == "TooLarge"


def test_listing_shows_only_the_kbs_a_key_may_see(client: TestClient, keys: dict[str, str]) -> None:
    for kb in ("c-1", "faq"):
        client.put(f"/v1/kbs/{kb}", headers=_h(keys["writer"]))
    listed = client.get("/v1/kbs", headers=_h(keys["reader"])).json()["kbs"]  # kbs: ["c-*"]
    assert listed == ["c-1"]
    assert client.get("/v1/kbs", headers=_h(keys["writer"])).json()["kbs"] == ["c-1", "faq"]


def test_a_server_misconfiguration_is_a_500_without_its_detail(
    store: PgStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator forgot the LLM key: not the caller's fault (not a 400), and the env var name
    stays in the log. A caller's own mistake still gets a 400 that says what is wrong."""
    from lite_rag.embedding import CachingEmbedder
    from lite_rag.engine import RagRuntime
    from lite_rag.testing import FakeEmbedder

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load_settings(chunking=CHUNKING, llm={"provider": "openai", "model": "gpt-x"})
    runtime = RagRuntime(settings, store, CachingEmbedder(FakeEmbedder(dim=8), store))
    with TestClient(create_app(settings, runtime), base_url="http://localhost") as c:
        h = {"X-LiteRag-Namespace": NS}
        c.put("/v1/kbs/c-1/documents/r", headers=h, json={"title": "R", "text": "Refunds."})
        r = c.post("/v1/kbs/c-1/ask", headers=h, json={"question": "refunds?"})
        assert r.status_code == 500 and "OPENAI_API_KEY" not in r.text
        bad = c.post("/v1/kbs/BAD/search", headers=h, json={"query": "x"})
        assert bad.status_code == 400 and "invalid knowledge base name" in bad.json()["detail"]


def test_a_write_key_cannot_create_or_write_kbs_outside_its_globs(store: PgStore) -> None:
    from lite_rag.server.auth import key_hash

    token = new_key()[0]
    scoped = ApiKeyConfig(
        name="scoped", sha256=key_hash(token), namespaces=["steward/*"], kbs=["c-*"], mode="write"
    )
    settings = load_settings(chunking=CHUNKING, server={"keys": [scoped.model_dump()]})
    with TestClient(create_app(settings, make_runtime(store))) as c:
        h = _h(token)
        assert c.put("/v1/kbs/c-2", headers=h).status_code == 200
        assert c.put("/v1/kbs/faq", headers=h).status_code == 403
        doc = {"title": "T", "text": "x"}
        assert c.put("/v1/kbs/faq/documents/d", headers=h, json=doc).status_code == 403
        assert c.put("/v1/kbs/c-2/documents/d", headers=h, json=doc).status_code == 200


def test_only_the_upload_route_gets_the_upload_limit(store: PgStore) -> None:
    settings = load_settings(
        chunking=CHUNKING, server={"max_body_mb": 0.001, "max_upload_mb": 1, "keys": []}
    )
    with TestClient(create_app(settings, make_runtime(store)), base_url="http://localhost") as c:
        body = b"x" * 5000  # over max_body_mb, under the upload limit
        for path in ("/anything/files", "/v1/kbs/a/b/files", "/v1/kbs/c-1/files/x"):
            assert c.post(path, content=body).status_code == 413, path


def test_a_missing_document_is_a_404_that_names_the_document(store: PgStore) -> None:
    settings = load_settings(chunking=CHUNKING)
    with TestClient(create_app(settings, make_runtime(store)), base_url="http://localhost") as c:
        h = {"X-LiteRag-Namespace": NS}
        c.put("/v1/kbs/c-1", headers=h)
        r = c.delete("/v1/kbs/c-1/documents/nope", headers=h)
    assert r.status_code == 404
    assert r.json()["title"] == "DocumentNotFound" and "'nope'" in r.json()["detail"]
