import json
import threading

import pytest

from lite_rag.graph import (
    ChunkGraph,
    Entity,
    GraphError,
    GraphExtractor,
    Relation,
    entity_key,
    parse_graph,
    seeds,
    walk,
)
from lite_rag.llm import Message
from lite_rag.testing import FakeLLM
from lite_rag.types import Chunk


def test_entity_names_merge_across_case_width_and_spacing() -> None:
    assert entity_key("  Maria   KELLER ") == "maria keller"
    full_width = "".join(chr(ord(c) + 0xFEE0) for c in "ATLAS")  # ＡＴＬＡＳ  # noqa: RUF003
    assert entity_key(full_width) == "atlas"  # NFKC
    assert entity_key("李明") == "李明"


def test_a_reply_parses_into_entities_and_relations() -> None:
    reply = json.dumps(
        {
            "entities": [
                {"name": "Project Atlas", "type": "project"},
                {"name": "Maria Keller", "type": "person"},
            ],
            "relations": [
                {"source": "Project Atlas", "target": "Maria Keller", "description": "led by"}
            ],
        }
    )
    assert parse_graph(reply) == ChunkGraph(
        (
            Entity("project atlas", "Project Atlas", "project"),
            Entity("maria keller", "Maria Keller", "person"),
        ),
        (Relation("project atlas", "maria keller", "led by"),),
    )


def test_a_reply_in_a_code_fence_with_text_around_it_still_parses() -> None:
    body = '{"entities": [{"name": "李明", "type": "人物"}], "relations": []}'
    reply = f"Here you go:\n```json\n{body}\n```"
    assert parse_graph(reply).entities == (Entity("李明", "李明", "人物"),)


def test_relation_ends_missing_from_the_entities_become_entities() -> None:
    reply = json.dumps(
        {
            "entities": [{"name": "Tom Osei", "type": "person"}],
            "relations": [{"source": "Tom Osei", "target": "Berlin office"}],
        }
    )
    graph = parse_graph(reply)
    assert [e.key for e in graph.entities] == ["tom osei", "berlin office"]
    assert graph.entities[1].type == ""
    assert graph.relations == (Relation("tom osei", "berlin office", ""),)


def test_duplicates_blank_names_and_self_relations_are_dropped() -> None:
    reply = json.dumps(
        {
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
    )
    graph = parse_graph(reply)
    assert graph.entities == (Entity("atlas", "Atlas", "project"), Entity("maria", "Maria", ""))
    assert graph.relations == (Relation("atlas", "maria", "led by"),)


@pytest.mark.parametrize(
    "reply",
    [
        "no json here",
        "{not json}",
        '["a list"]',
        '{"entities": "Atlas"}',
        '{"entities": [{"type": "person"}]}',
        '{"entities": [], "relations": [{"source": "a"}]}',
        '{"entities": [{"name": 7}]}',
        "{}",  # an empty object is not "nothing found": the model ignored the shape
        '{"entities": [{"name": "Atlas"}]}',  # relations missing
        '{"relations": []}',  # entities missing
    ],
)
def test_a_malformed_reply_raises(reply: str) -> None:
    with pytest.raises(GraphError):
        parse_graph(reply)


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

    assert graphs == [parse_graph(reply)] * 2
    assert len(llm.calls) == 2
    user = llm.calls[1][-1]
    assert isinstance(user, Message) and "Tom leads Beacon." in user.content
    assert "Maria leads Atlas." not in user.content  # one chunk per call
    # no heading or title: a model lists them as entities of every chunk, hubs linking everything
    assert "Leads" not in user.content


def test_the_extractor_runs_calls_concurrently_up_to_its_limit() -> None:
    class Together(FakeLLM):
        """Each call waits for the other: one at a time would time out."""

        barrier = threading.Barrier(2, timeout=5)

        def complete(self, messages):  # type: ignore[no-untyped-def]
            self.barrier.wait()
            return super().complete(messages)

    llm = Together(reply='{"entities": [], "relations": []}')
    assert len(GraphExtractor(llm, concurrency=2).extract(_chunks(["a", "b"]))) == 2


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
