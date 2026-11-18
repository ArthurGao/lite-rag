import threading
from typing import Any

import pytest
from lite_model import strict_schema
from pydantic import ValidationError

from lite_rag.graph import (
    ChunkGraph,
    Entity,
    GraphError,
    GraphExtractor,
    GraphReply,
    Relation,
    entity_key,
    seeds,
    to_graph,
    walk,
)
from lite_rag.llm import LLMError, Message
from lite_rag.testing import FakeLLM
from lite_rag.types import Chunk

EMPTY = '{"entities": [], "relations": []}'


def test_entity_names_merge_across_case_width_and_spacing() -> None:
    assert entity_key("  Maria   KELLER ") == "maria keller"
    full_width = "".join(chr(ord(c) + 0xFEE0) for c in "ATLAS")  # ＡＴＬＡＳ  # noqa: RUF003
    assert entity_key(full_width) == "atlas"  # NFKC
    assert entity_key("李明") == "李明"


def _graph(reply: dict[str, Any]) -> ChunkGraph:
    return to_graph(GraphReply.model_validate(reply))


def test_a_reply_becomes_entities_and_relations() -> None:
    reply = {
        "entities": [
            {"name": "Project Atlas", "type": "project"},
            {"name": "Maria Keller", "type": "person"},
        ],
        "relations": [
            {"source": "Project Atlas", "target": "Maria Keller", "description": "led by"}
        ],
    }
    assert _graph(reply) == ChunkGraph(
        (
            Entity("project atlas", "Project Atlas", "project"),
            Entity("maria keller", "Maria Keller", "person"),
        ),
        (Relation("project atlas", "maria keller", "led by"),),
    )


def test_relation_ends_missing_from_the_entities_become_entities() -> None:
    reply = {
        "entities": [{"name": "Tom Osei", "type": "person"}],
        "relations": [{"source": "Tom Osei", "target": "Berlin office"}],
    }
    graph = _graph(reply)
    assert [e.key for e in graph.entities] == ["tom osei", "berlin office"]
    assert graph.entities[1].type == ""
    assert graph.relations == (Relation("tom osei", "berlin office", ""),)


def test_duplicates_blank_names_and_self_relations_are_dropped() -> None:
    reply = {
        "entities": [
            {"name": "Atlas", "type": "project"},
            {"name": "ATLAS", "type": "system"},
            {"name": "  ", "type": "x"},
        ],
        "relations": [
            {"source": "Atlas", "target": "atlas", "description": "is"},
            {"source": "Atlas", "target": "Maria", "description": "led by"},
            {"source": "atlas", "target": "MARIA", "description": "again"},
        ],
    }
    graph = _graph(reply)
    assert graph.entities == (Entity("atlas", "Atlas", "project"), Entity("maria", "Maria", ""))
    assert graph.relations == (Relation("atlas", "maria", "led by"),)


@pytest.mark.parametrize(
    "reply",
    [
        "no json here",
        "{not json}",
        '["a list"]',
        '{"entities": "Atlas"}',
        '{"entities": [{"type": "person"}], "relations": []}',
        '{"entities": [], "relations": [{"source": "a"}]}',
        '{"entities": [{"name": 7}], "relations": []}',
        "{}",  # an empty object is not "nothing found": the model ignored the shape
        '{"entities": [{"name": "Atlas"}]}',  # relations missing
        '{"relations": []}',  # entities missing
        '{"entities": [], "relations": [], "notes": "x"}',  # nothing else
    ],
)
def test_a_reply_off_the_schema_is_refused(reply: str) -> None:
    with pytest.raises(ValidationError):
        GraphReply.model_validate_json(reply)


def test_the_schema_the_model_is_held_to_requires_both_lists() -> None:
    # sent with every graph call (S16): without it a local model wrote invalid JSON
    schema = strict_schema(GraphReply)
    assert schema["required"] == ["entities", "relations"]
    assert schema["additionalProperties"] is False


# A small graph as (entity, chunk) mentions: atlas → maria → safety council → its meeting chunk,
# and a company named in four chunks (a hub with hub_chunks=3).
MENTIONS = [
    ("atlas", "projects#1"),
    ("maria", "projects#1"),
    ("maria", "people#1"),
    ("safety council", "people#1"),
    ("safety council", "councils#1"),
    ("acme", "people#1"),
    ("acme", "people#2"),
    ("acme", "offices#1"),
    ("acme", "projects#0"),
    ("tom", "people#2"),
    ("spending council", "people#2"),
    ("spending council", "councils#2"),
]


def chunks_of(entities: list[str]) -> list[tuple[str, str]]:
    return [(e, c) for e, c in MENTIONS if e in entities]


def entities_of(chunks: list[str]) -> list[tuple[str, str]]:
    return [(e, c) for e, c in MENTIONS if c in chunks]


def reach(seeds: list[str], hops: int, hub_chunks: int = 3, max_neighbors: int = 10) -> list[str]:
    return walk(
        seeds, chunks_of, entities_of, hops=hops, hub_chunks=hub_chunks, max_neighbors=max_neighbors
    )


def test_each_hop_reaches_the_chunks_of_the_entities_one_chunk_further() -> None:
    assert reach(["atlas"], hops=0) == ["projects#1"]
    assert reach(["atlas"], hops=1) == ["people#1", "projects#1"]
    assert reach(["atlas"], hops=2) == ["councils#1", "people#1", "projects#1"]


def test_a_hub_is_neither_walked_through_nor_a_way_to_reach_chunks() -> None:
    # people#1 names "acme" (4 chunks > 3): its other chunks — and the Spending Council — stay out
    assert "people#2" not in reach(["atlas"], hops=3)
    assert reach(["acme"], hops=2) == []
    # with a higher limit acme is an ordinary entity, and the walk floods through it
    assert "councils#2" in reach(["atlas"], hops=3, hub_chunks=4)


def test_each_hop_keeps_the_entities_linked_by_the_most_chunks() -> None:
    # from people#1 the new entities are acme and safety council (one chunk each): ties by key
    assert reach(["maria"], hops=1, hub_chunks=10, max_neighbors=1) == [
        "offices#1",
        "people#1",
        "people#2",
        "projects#0",
        "projects#1",
    ]


def test_seeds_skip_one_letter_names_and_category_words() -> None:
    found = [("atlas", 0), ("a", 0), ("council", 2), ("data safety council", 0), ("项目", 1)]
    # "council" sits inside two other names: a category word; "项目" inside one: still a name
    assert seeds(found) == ["atlas", "data safety council", "项目"]


def test_no_seed_reaches_nothing() -> None:
    assert reach([], hops=2) == []


def _chunks(texts: list[str]) -> list[Chunk]:
    out, start = [], 0
    for n, text in enumerate(texts):
        end = start + len(text)
        out.append(Chunk(f"d#{n}", "d", n, text, start, end, len(text.split()), ("Leads",)))
        start += len(text) + 2
    return out


def test_the_extractor_calls_the_llm_once_per_chunk_with_the_passage_fenced() -> None:
    reply = '{"entities": [{"name": "Maria Keller", "type": "person"}], "relations": []}'
    llm = FakeLLM(reply=reply)
    graphs = GraphExtractor(llm).extract(_chunks(["Maria leads Atlas.", "Tom leads Beacon."]))

    assert (
        graphs
        == [_graph({"entities": [{"name": "Maria Keller", "type": "person"}], "relations": []})] * 2
    )
    assert len(llm.calls) == 2
    assert all(c.schema is GraphReply for c in llm.fake.calls)  # constrained to the schema
    user = llm.calls[1][-1]
    assert isinstance(user, Message) and "Tom leads Beacon." in user.content
    assert "Maria leads Atlas." not in user.content  # one chunk per call
    # no heading or title: a model lists them as entities of every chunk, hubs linking everything
    assert "Leads" not in user.content


def test_the_extractor_runs_calls_concurrently_up_to_its_limit() -> None:
    barrier = threading.Barrier(2, timeout=5)

    def together(messages: list[Message]) -> str:
        barrier.wait()  # each call waits for the other: one at a time would time out
        return EMPTY

    llm = FakeLLM(reply=together)
    assert len(GraphExtractor(llm, concurrency=2).extract(_chunks(["a", "b"]))) == 2
    assert len(llm.calls) == 2


def _replies(*replies: str) -> FakeLLM:
    """A FakeLLM that answers with `replies` in turn."""
    it = iter(replies)
    return FakeLLM(reply=lambda messages: next(it))


def test_a_reply_that_is_not_a_graph_is_asked_for_once_more() -> None:
    # S16: a local model at temperature 0 still samples; one bad reply must not fail the ingest
    llm = _replies("not json {", '{"entities": [{"name": "Atlas"}], "relations": []}')
    assert GraphExtractor(llm).extract(_chunks(["Atlas."])) == [
        ChunkGraph((Entity("atlas", "Atlas", ""),), ())
    ]
    assert len(llm.calls) == 2 and llm.calls[0] == llm.calls[1]  # the same prompt again


def test_asking_again_is_logged_without_the_reply(caplog: pytest.LogCaptureFixture) -> None:
    # how often the model fails the shape stays visible; the reply itself is not logged
    llm = _replies("secret text {", EMPTY)
    with caplog.at_level("WARNING", logger="lite_rag.graph"):
        GraphExtractor(llm).extract(_chunks(["Atlas."]))
    [record] = caplog.records
    assert record.getMessage() == "graph reply for chunk d#0 is not a graph; asking once more"


def test_a_second_reply_that_is_not_a_graph_fails() -> None:
    llm = _replies("not json {", '{"entities": []}', EMPTY)
    with pytest.raises(GraphError, match=r"chunk d#0: role answer \(fake:fake-model\)"):
        GraphExtractor(llm).extract(_chunks(["Atlas."]))
    assert len(llm.calls) == 2  # once more, not until it parses


def test_a_failed_call_is_not_asked_again_here() -> None:
    # a call that failed was already retried by lite-model; only an unparsable reply is repeated
    def fail(messages: list[Message]) -> str:
        raise LLMError("role graph (ollama:qwen3:8b): ConnectError")

    llm = FakeLLM(reply=fail)
    with pytest.raises(LLMError) as caught:
        GraphExtractor(llm).extract(_chunks(["Atlas."]))
    assert not isinstance(caught.value, GraphError) and len(llm.calls) == 1


def test_an_entity_linked_by_more_chunks_goes_first_whatever_its_name() -> None:
    mentions = [
        ("seed", "a"),
        ("seed", "b"),
        ("alpha", "a"),  # linked by one chunk
        ("zulu", "a"),  # linked by two
        ("zulu", "b"),
        ("alpha", "x"),
        ("zulu", "z"),
    ]
    reached = walk(
        ["seed"],
        lambda es: [(e, c) for e, c in mentions if e in es],
        lambda cs: [(e, c) for e, c in mentions if c in cs],
        hops=1,
        hub_chunks=10,
        max_neighbors=1,
    )
    assert reached == ["a", "b", "z"]
