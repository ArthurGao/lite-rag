"""What a `RagRuntime` shares between the engines it hands out (design §3)."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest

from lite_rag import engine as engine_module
from lite_rag.config import load_settings
from lite_rag.contextual import Contextualizer
from lite_rag.engine import RagRuntime
from lite_rag.errors import ConfigError
from lite_rag.llm import Completion, Message
from lite_rag.stores.base import CachedStore
from lite_rag.testing import FakeEmbedder, FakeLLM
from lite_rag.text import CharCounter
from lite_rag.types import Chunk, Document


class _Store:
    closed = False

    def close(self) -> None:
        self.closed = True


def _runtime() -> RagRuntime:
    store = cast(CachedStore, _Store())
    return RagRuntime(load_settings(), store, FakeEmbedder(dim=8), context_llm=FakeLLM())


def test_engines_share_the_runtime_and_keep_their_own_namespace() -> None:
    rt = _runtime()
    a, b = rt.engine("proj/a"), rt.engine("proj/b")
    assert (a.namespace, b.namespace) == ("proj/a", "proj/b")
    assert a.runtime is b.runtime is rt
    # one contextualizer: its bound covers every ingest of every engine
    assert rt.contextualizer() is rt.contextualizer()


def test_a_model_is_built_once_even_when_requests_ask_at_the_same_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = []

    def slow_build(config: Any) -> object:
        time.sleep(0.02)  # wide enough for every thread to arrive while it builds
        built.append(config)
        return object()

    monkeypatch.setattr(engine_module, "build_llm", slow_build)
    rt = _runtime()
    with ThreadPoolExecutor(8) as pool:
        models = list(pool.map(lambda _: rt.llm(), range(8)))
    assert len(built) == 1 and all(m is models[0] for m in models)


def test_context_calls_stay_within_the_bound_across_concurrent_ingests() -> None:
    class Counting:
        model_id = "fake/counting"

        def __init__(self) -> None:
            self.running = self.peak = 0
            self._lock = threading.Lock()

        def complete(self, messages: list[Message]) -> Completion:
            with self._lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
            time.sleep(0.02)
            with self._lock:
                self.running -= 1
            return Completion("context", self.model_id, {})

    llm = Counting()
    contextualizer = Contextualizer(llm, CharCounter(), 6000, concurrency=2)
    doc = Document("d", "T", "x " * 50)
    chunks = [Chunk(f"d#{i}", "d", i, "x", i, i + 1, 1) for i in range(4)]
    with ThreadPoolExecutor(3) as pool:  # three documents ingested at once
        list(pool.map(lambda _: contextualizer.contexts(doc, chunks), range(3)))
    assert llm.peak == 2


@pytest.mark.parametrize("failing", ["build_embedder", "build_chunker"])
def test_from_settings_closes_the_store_when_a_later_step_fails(
    monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    store = _Store()
    monkeypatch.setattr(engine_module, "open_store", lambda *a, **k: store)
    if failing == "build_chunker":
        monkeypatch.setattr(engine_module, "build_embedder", lambda *a, **k: FakeEmbedder(dim=8))

    def broken(*args: Any, **kwargs: Any) -> None:
        raise ConfigError("cannot build it")

    monkeypatch.setattr(engine_module, failing, broken)
    with pytest.raises(ConfigError):
        RagRuntime.from_settings(load_settings())
    assert store.closed
