"""What only the Chroma backend has (design §4): its write order, crash recovery, filter limits."""

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lite_rag.cli import app
from lite_rag.config import load_settings
from lite_rag.engine import RagEngine
from lite_rag.errors import StoreError
from lite_rag.stores.base import KbInfo, KbRef
from lite_rag.stores.chroma import SQLITE_FILE, ChromaStore, collection_name, migrate_sqlite
from lite_rag.stores.factory import evaluation_store, migrate_store
from tests.contract.test_store import KB, NS, info, put


class Crash(BaseException):
    """The process dying mid-operation: no `except Exception` cleanup runs."""


Reopen = Callable[[], ChromaStore]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    migrate_sqlite(tmp_path / "data")
    return tmp_path / "data"


class _Opened:
    """The store a test is using; `reopen` closes it and opens the data directory again."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir, self.current = data_dir, ChromaStore(data_dir)

    def reopen(self) -> ChromaStore:
        self.current.close()
        self.current = ChromaStore(self.data_dir)
        return self.current


@pytest.fixture
def opened(data_dir: Path) -> Iterator[_Opened]:
    o = _Opened(data_dir)
    yield o
    o.current.close()


@pytest.fixture
def store(opened: _Opened) -> ChromaStore:
    opened.current.create_kb(info())
    return opened.current


@pytest.fixture
def reopen(opened: _Opened) -> Reopen:
    return opened.reopen


def _raise(exc: BaseException) -> Callable[..., None]:
    def fail(*args: Any) -> None:
        raise exc

    return fail


def _rows(data_dir: Path, table: str) -> int:
    with sqlite3.connect(data_dir / SQLITE_FILE) as conn:
        return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _vectors(store: ChromaStore, doc_id: str, kb: KbRef = KB) -> list[str]:
    return list(store._collection(kb).get(where={"doc_id": doc_id}, include=[])["ids"])


def _hidden(store: ChromaStore, doc_id: str) -> bool:
    return (
        store.document_hash(KB, doc_id) is None
        and store.chunks(KB, [f"{doc_id}#0"]) == {}
        and all(not c.startswith(doc_id) for c, _ in store.vector_query(KB, [1, 0, 0, 0], 10))
        and store.bm25_query(KB, ["alpha"], 10, 1.2, 0.75) == []
    )


def _two_docs(store: ChromaStore) -> None:
    put(store, "d1", [("alpha one", [1, 0, 0, 0]), ("alpha two", [0, 1, 0, 0])])
    put(store, "d2", [("beta", [0, 0, 1, 0])])


# --- crash recovery ----------------------------------------------------------------------------


@pytest.mark.parametrize("after_vectors", [False, True], ids=["before-chroma", "before-ready"])
def test_an_upsert_that_crashed_is_hidden_then_removed_on_open(
    store: ChromaStore,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_vectors: bool,
    reopen: Reopen,
) -> None:
    _two_docs(store)
    original = store._add_vectors

    def crash(*args: Any) -> None:
        if after_vectors:
            original(*args)
        raise Crash

    monkeypatch.setattr(store, "_add_vectors", crash)
    with pytest.raises(Crash):
        put(store, "d1", [("alpha three", [1, 0, 0, 0])], content_hash="h2")
    monkeypatch.undo()

    assert _hidden(store, "d1")  # rows `pending`: never read, even before recovery
    assert _rows(data_dir, "documents") == 2
    stats = store.stats(KB)
    assert stats is not None and (stats.documents, stats.chunks) == (1, 1)

    store = reopen()
    assert (_rows(data_dir, "documents"), _rows(data_dir, "chunks")) == (1, 1)
    assert _vectors(store, "d1") == [] and _vectors(store, "d2") == ["d2#0"]
    put(store, "d1", [("alpha three", [1, 0, 0, 0])], content_hash="h2")  # ingested again
    assert store.document_hash(KB, "d1") == "h2"
    assert [c for c, _ in store.vector_query(KB, [1, 0, 0, 0], 1)] == ["d1#0"]


@pytest.mark.parametrize("vectors_removed", [False, True], ids=["before-chroma", "before-rows"])
def test_a_delete_that_crashed_is_hidden_then_finished_on_open(
    store: ChromaStore,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    vectors_removed: bool,
    reopen: Reopen,
) -> None:
    _two_docs(store)

    def crash(kb: KbRef, doc_id: str) -> None:
        if vectors_removed:
            store._collection(kb).delete(where={"doc_id": doc_id})
        raise Crash

    monkeypatch.setattr(store, "_discard", crash)
    with pytest.raises(Crash):
        store.delete_document(KB, "d1")
    monkeypatch.undo()

    assert _hidden(store, "d1")  # `deleting`

    store = reopen()
    assert _rows(data_dir, "documents") == 1
    assert _vectors(store, "d1") == [] and _vectors(store, "d2") == ["d2#0"]


def test_a_kb_delete_that_crashed_leaves_no_collection_behind(
    store: ChromaStore, data_dir: Path, monkeypatch: pytest.MonkeyPatch, reopen: Reopen
) -> None:
    _two_docs(store)

    def crash(name: str) -> None:
        raise Crash

    monkeypatch.setattr(store, "_drop_collection", crash)
    with pytest.raises(Crash):
        store.delete_kb(KB)
    monkeypatch.undo()

    store = reopen()
    names = [c.name for c in store._client.list_collections()]
    assert collection_name(KB) not in names


def test_a_kb_created_over_an_orphan_collection_starts_empty(
    store: ChromaStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_docs(store)
    monkeypatch.setattr(store, "_drop_collection", lambda name: None)  # the crash, in effect
    store.delete_kb(KB)
    monkeypatch.undo()

    store.create_kb(info())
    assert store.vector_query(KB, [1, 0, 0, 0], 10) == []
    assert _vectors(store, "d1") == []


# --- failures (not crashes) ---------------------------------------------------------------------


def test_a_failed_vector_write_removes_the_document_at_once(
    store: ChromaStore, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_docs(store)

    monkeypatch.setattr(store, "_add_vectors", _raise(RuntimeError("disk full")))
    with pytest.raises(StoreError, match="disk full"):
        put(store, "d1", [("alpha three", [1, 0, 0, 0])], content_hash="h2")

    assert _rows(data_dir, "documents") == 1  # no `pending` row waits for the next open
    assert _vectors(store, "d1") == [] and store.document_hash(KB, "d2") == "h1"


def test_when_the_cleanup_fails_too_the_document_stays_hidden_until_the_next_open(
    store: ChromaStore, data_dir: Path, monkeypatch: pytest.MonkeyPatch, reopen: Reopen
) -> None:
    _two_docs(store)

    fail = _raise(RuntimeError("chroma is gone"))
    monkeypatch.setattr(store, "_add_vectors", fail)
    monkeypatch.setattr(store, "_discard", fail)
    with pytest.raises(StoreError) as caught:
        put(store, "d1", [("alpha three", [1, 0, 0, 0])], content_hash="h2")
    monkeypatch.undo()

    assert any("next opens" in note for note in caught.value.__cause__.__notes__)  # type: ignore[union-attr]
    assert _hidden(store, "d1")
    store = reopen()
    assert _rows(data_dir, "documents") == 1 and _vectors(store, "d1") == []


def test_vectors_are_added_in_batches_chroma_accepts(
    store: ChromaStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, sizes = store._collection(KB), []
    add = collection.add

    def spy(**kwargs: Any) -> None:
        sizes.append(len(kwargs["ids"]))
        add(**kwargs)

    monkeypatch.setattr(store._client, "get_max_batch_size", lambda: 2)
    monkeypatch.setattr(collection, "add", spy)
    put(store, "d1", [(f"chunk {i}", [1, i, 0, 0]) for i in range(5)])
    assert sizes == [2, 2, 1]
    assert sorted(_vectors(store, "d1")) == [f"d1#{i}" for i in range(5)]


# --- metadata filters -------------------------------------------------------------------------


@pytest.mark.parametrize(("where", "found"), [({"flag": True}, 1), ({"flag": 1}, 0)])
def test_both_paths_filter_alike_and_a_bool_never_equals_a_number(
    store: ChromaStore, where: dict[str, Any], found: int
) -> None:
    put(store, "d1", [("alpha", [1, 0, 0, 0])], metadata={"flag": True, "lang": "en"})
    both = {**where, "lang": "en"}  # two conditions: Chroma's $and
    for flt in (where, both):
        assert len(store.vector_query(KB, [1, 0, 0, 0], 5, flt)) == found
        assert len(store.bm25_query(KB, ["alpha"], 5, 1.2, 0.75, flt)) == found


# --- opening ----------------------------------------------------------------------------------


def test_the_store_needs_migrate_first(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="literag migrate"):
        ChromaStore(tmp_path / "nothing")


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    assert migrate_sqlite(tmp_path / "d") == ["0001_initial"]
    assert migrate_sqlite(tmp_path / "d") == []


def test_telemetry_is_off(store: ChromaStore) -> None:
    assert store._client.get_settings().anonymized_telemetry is False


def test_a_kb_identity_survives_a_reopen(
    store: ChromaStore, data_dir: Path, reopen: Reopen
) -> None:
    other = KbInfo(KbRef("ns", "x"), "p", "m", 3, "", "", {"a": [1, {"b": None}]})
    store.create_kb(other)
    store = reopen()
    assert store.get_kb(other.ref) == other


# --- wiring -----------------------------------------------------------------------------------


def test_cli_migrate_creates_the_data_dir_then_is_up_to_date(tmp_path: Path) -> None:
    env = {"LITERAG_STORE__BACKEND": "chroma", "LITERAG_STORE__DATA_DIR": str(tmp_path / "d")}
    runner = CliRunner()
    first = runner.invoke(app, ["migrate"], env=env)
    assert first.exit_code == 0, first.output
    assert "applied 0001_initial" in first.output and (tmp_path / "d" / SQLITE_FILE).is_file()
    second = runner.invoke(app, ["migrate"], env=env)
    assert second.exit_code == 0 and "up to date" in second.output


def test_the_engine_opens_the_configured_backend(tmp_path: Path) -> None:
    migrate_sqlite(tmp_path / "d")
    settings = load_settings(store={"backend": "chroma", "data_dir": str(tmp_path / "d")})
    engine = RagEngine.from_settings(settings, NS)
    try:
        assert isinstance(engine.runtime.store, ChromaStore) and engine.list_kbs() == []
    finally:
        engine.close()


def test_eval_keeps_its_kbs_apart_from_application_data(tmp_path: Path) -> None:
    store = load_settings(store={"backend": "chroma", "data_dir": str(tmp_path / "d")}).store
    assert migrate_store(evaluation_store(store, "ev"), None) == ["0001_initial"]
    assert (tmp_path / "d" / "ev" / SQLITE_FILE).is_file()
    assert not (tmp_path / "d" / SQLITE_FILE).exists()


def test_eval_refuses_configs_on_different_data_dirs(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("store: {backend: chroma, data_dir: one}\n")
    b.write_text("store: {backend: chroma, data_dir: two}\n")
    result = CliRunner().invoke(app, ["eval", "-c", str(a), "-c", str(b)])
    assert result.exit_code != 0 and "one database" in result.output
