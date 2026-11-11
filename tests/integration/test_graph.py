"""Graph retrieval on PostgreSQL (design §5, S11), with a scripted extraction model."""

import json
from typing import Any

import psycopg
import pytest

from lite_rag.engine import RagEngine
from lite_rag.errors import InvalidRequest, KbMismatch
from lite_rag.graph import GraphError
from lite_rag.llm import Message
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeLLM
from tests.conftest import TEST_SCHEMA
from tests.integration.support import make_engine

ATLAS = "Project Atlas is led by Maria Keller."
MARIA = "Maria Keller chairs the Data Safety Council."
MARIA_NOW = "Maria Keller chairs the Spending Council."
SAFETY = "The Data Safety Council meets every second Thursday."
SPENDING = "The Spending Council meets on the first Monday."

# What the scripted model "finds" in each passage.
GRAPHS: dict[str, dict[str, Any]] = {
    ATLAS: {
        "entities": [{"name": "Project Atlas", "type": "project"}, {"name": "Maria Keller"}],
        "relations": [
            {"source": "Project Atlas", "target": "Maria Keller", "description": "led by"}
        ],
    },
    MARIA: {
        "entities": [{"name": "Maria Keller"}, {"name": "Data Safety Council"}],
        "relations": [{"source": "Maria Keller", "target": "Data Safety Council"}],
    },
    MARIA_NOW: {
        "entities": [{"name": "Maria Keller"}, {"name": "Spending Council"}],
        "relations": [{"source": "Maria Keller", "target": "Spending Council"}],
    },
    SAFETY: {"entities": [{"name": "Data Safety Council"}], "relations": []},
    SPENDING: {"entities": [{"name": "Spending Council"}], "relations": []},
}


def extract(messages: list[Message]) -> str:
    passage = next(p for p in GRAPHS if p in messages[-1].content)
    return json.dumps(GRAPHS[passage])


def engine(
    store: PgStore,
    llm: FakeLLM | None = None,
    *,
    graph: dict[str, Any] | None = None,
    **retrieval: Any,
) -> RagEngine:
    return make_engine(
        store,
        retrieval=retrieval,
        graph={"enabled": True, **(graph or {})},
        graph_llm=llm or FakeLLM(extract),
        size=60,  # one sentence per chunk: paragraph chunks merge short paragraphs up to `size`
    )


def ingest(eng: RagEngine, kb: str = "kb") -> None:
    eng.upsert_text(kb, "projects", "Projects", ATLAS)
    eng.upsert_text(kb, "people", "People", MARIA)
    eng.upsert_text(kb, "councils", "Councils", f"{SAFETY}\n\n{SPENDING}")


def rows(pg: psycopg.Connection, sql: str) -> list[tuple[Any, ...]]:
    return pg.execute(sql.format(s=TEST_SCHEMA)).fetchall()


def test_ingest_stores_each_chunks_entities_and_relations(
    store: PgStore, pg: psycopg.Connection
) -> None:
    llm = FakeLLM(extract)
    ingest(engine(store, llm))

    assert len(llm.calls) == 4  # one call per chunk
    assert rows(
        pg, "SELECT chunk_id, entity, name, type FROM {s}.graph_mentions ORDER BY 1, 2"
    ) == [
        ("councils#0", "data safety council", "Data Safety Council", ""),
        ("councils#1", "spending council", "Spending Council", ""),
        ("people#0", "data safety council", "Data Safety Council", ""),
        ("people#0", "maria keller", "Maria Keller", ""),
        ("projects#0", "maria keller", "Maria Keller", ""),
        ("projects#0", "project atlas", "Project Atlas", "project"),
    ]
    assert rows(
        pg, "SELECT chunk_id, source, target, description FROM {s}.graph_relations ORDER BY 1"
    ) == [
        ("people#0", "maria keller", "data safety council", ""),
        ("projects#0", "project atlas", "maria keller", "led by"),
    ]


def test_the_graph_path_reaches_the_answer_two_hops_from_the_named_entity(store: PgStore) -> None:
    eng = engine(store)
    ingest(eng)
    query = "When does the council chaired by the lead of Project Atlas meet?"
    hits = eng.search("kb", query, mode="graph")

    # "project atlas" (named) → projects#0 → maria keller → people#0 → data safety council →
    # councils#0. The Spending Council's chunk is never reached, though it matches the query's
    # words as well as the answer does.
    assert {h.chunk_id for h in hits} == {"projects#0", "people#0", "councils#0"}
    # the graph narrows, the query vector orders: the same scores as the vector path
    vector = {h.chunk_id: h.score for h in eng.search("kb", query, mode="vector", top_k=10)}
    assert [h.score for h in hits] == [vector[h.chunk_id] for h in hits]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    answer = next(h for h in hits if h.chunk_id == "councils#0")
    assert answer.text == SAFETY and set(answer.scores) == {"graph"}


def test_hops_limit_how_far_the_graph_reaches(store: PgStore) -> None:
    eng = engine(store, graph={"hops": 1})
    ingest(eng)
    hits = eng.search("kb", "the lead of Project Atlas", mode="graph")
    assert {h.chunk_id for h in hits} == {"projects#0", "people#0"}


def test_a_hub_entity_is_not_walked_through(store: PgStore) -> None:
    eng = engine(store, graph={"hub_chunks": 1})  # maria keller is in two chunks: a hub
    ingest(eng)
    hits = eng.search("kb", "the lead of Project Atlas", mode="graph")
    assert [h.chunk_id for h in hits] == ["projects#0"]


def test_a_category_word_is_not_a_seed(store: PgStore) -> None:
    eng = engine(store)
    ingest(eng)
    # "council" is an entity of its own here, but it sits inside two other names
    GRAPHS["Every council meets in public."] = {"entities": [{"name": "Council"}], "relations": []}
    try:
        eng.upsert_text("kb", "rules", "Rules", "Every council meets in public.")
    finally:
        del GRAPHS["Every council meets in public."]
    assert eng.search("kb", "When does the council meet?", mode="graph") == []
    hits = eng.search("kb", "When does the Spending Council meet?", mode="graph")
    assert {h.chunk_id for h in hits} == {"councils#1"}


def test_hybrid_can_fuse_the_graph_with_vector_and_bm25(store: PgStore) -> None:
    eng = engine(store, hybrid_paths=["vector", "bm25", "graph"])
    ingest(eng)
    hits = eng.search("kb", "Project Atlas", top_k=4)
    assert hits[0].chunk_id == "projects#0"
    assert set(hits[0].scores) == {"vector", "bm25", "graph", "fused"}


def test_a_metadata_filter_also_scopes_the_walk(store: PgStore) -> None:
    eng = engine(store)
    eng.upsert_text("kb", "projects", "Projects", ATLAS, metadata={"team": "ops"})
    eng.upsert_text("kb", "people", "People", MARIA, metadata={"team": "finance"})
    eng.upsert_text("kb", "councils", "Councils", SAFETY, metadata={"team": "ops"})
    # Unfiltered, the finance page bridges Atlas to the Data Safety Council; scoped to ops it
    # is not part of the corpus, so it cannot carry the walk either.
    assert {h.chunk_id for h in eng.search("kb", "the lead of Project Atlas", mode="graph")} == {
        "projects#0",
        "people#0",
        "councils#0",
    }
    hits = eng.search("kb", "the lead of Project Atlas", mode="graph", where={"team": "ops"})
    assert [h.chunk_id for h in hits] == ["projects#0"]


def test_replacing_and_deleting_a_document_replaces_and_removes_its_graph(
    store: PgStore, pg: psycopg.Connection
) -> None:
    eng = engine(store)
    ingest(eng)
    eng.upsert_text("kb", "people", "People", MARIA_NOW)  # Maria now chairs the Spending Council
    hits = eng.search("kb", "the lead of Project Atlas", mode="graph")
    assert "councils#1" in {h.chunk_id for h in hits}  # reached through the new text
    assert "councils#0" not in {h.chunk_id for h in hits}

    eng.delete("kb", "people")
    assert rows(pg, "SELECT DISTINCT chunk_id FROM {s}.graph_mentions ORDER BY 1") == [
        ("councils#0",),
        ("councils#1",),
        ("projects#0",),
    ]
    assert rows(pg, "SELECT chunk_id FROM {s}.graph_relations") == [("projects#0",)]


def test_a_reply_that_is_not_a_graph_writes_nothing(store: PgStore, pg: psycopg.Connection) -> None:
    with pytest.raises(GraphError):
        engine(store, FakeLLM("I could not find anything.")).upsert_text("kb", "d", "D", ATLAS)
    assert rows(pg, "SELECT count(*) FROM {s}.chunks") == [(0,)]
    assert rows(pg, "SELECT count(*) FROM {s}.graph_mentions") == [(0,)]


def test_the_graph_mode_needs_a_kb_built_with_the_graph(store: PgStore) -> None:
    plain = make_engine(store, size=60)
    plain.upsert_text("kb", "projects", "Projects", ATLAS)
    with pytest.raises(InvalidRequest, match=r"graph\.enabled"):
        plain.search("kb", "Project Atlas", mode="graph")


def test_a_kb_refuses_another_extraction_model(store: PgStore) -> None:
    ingest(engine(store))
    other = engine(store, graph={"llm": {"model": "qwen3:14b"}})
    with pytest.raises(KbMismatch, match="graph"):
        other.search("kb", "Project Atlas", mode="graph")


def test_several_kbs_are_walked_each_in_its_own_graph(store: PgStore) -> None:
    eng = engine(store)
    eng.upsert_text("a", "projects", "Projects", ATLAS)
    eng.upsert_text("b", "people", "People", MARIA)
    hits = eng.search_kbs(["a", "b"], "the lead of Project Atlas", mode="graph")
    # kb b has no "project atlas": its graph starts from nothing the query names
    assert [(h.kb, h.chunk_id) for h in hits] == [("a", "projects#0")]


def test_a_one_letter_name_is_not_a_seed(store: PgStore) -> None:
    eng = engine(store)
    GRAPHS["Grade A is the best grade."] = {"entities": [{"name": "A"}], "relations": []}
    try:
        eng.upsert_text("kb", "grades", "Grades", "Grade A is the best grade.")
    finally:
        del GRAPHS["Grade A is the best grade."]
    # "a" is inside almost any query: matching it would seed every search
    assert eng.search("kb", "what is a grade", mode="graph") == []
