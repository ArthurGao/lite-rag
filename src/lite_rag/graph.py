"""Graph retrieval, local mode (design §5, S11).

Ingest: an LLM reads each chunk once and lists the entities it names and the relations it states
between them (`GraphExtractor`, prompts `graph_*.j2`). Entities merge across chunks and documents
by a normalised name (`entity_key`). A reply that is not the expected JSON fails the upsert — no
retry with another model, no partial graph.

Query: the graph narrows, the query vector orders. Seed entities are the ones the query names
(not category words such as "council", which sit inside several other names) → `walk` goes from
an entity to the chunks that mention it and on to the other entities those chunks mention, `hops`
times, never through a hub (an entity in more than `hub_chunks` chunks: a company name, a running
theme) → the chunks reached are ranked by their similarity to the query vector. Walking
co-mentions rather than the extracted relations measured better: a local model leaves out many
relations it could have stated (eval, 2026-11-12). The walk lives here; the store answers two
lookups (an entity's chunks, a chunk's entities).
"""

import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any

from lite_rag.llm import LLM, BoundedCalls, LLMError
from lite_rag.prompts import digest, render_pair
from lite_rag.types import Chunk, ChunkGraph, Entity, Relation

TEMPLATES = ("graph_system.j2", "graph_user.j2")

# Bump when the code that turns a reply into a graph changes (`parse_graph`, `entity_key`):
# stored graphs then no longer match, and the kb identity says so.
CODE_VERSION = "1"

MIN_SEED_CHARS = 2  # an entity name matched inside the query: shorter ones match by accident
GENERIC_INSIDE = 2  # a name inside this many other entity names is a category word, not a seed


class GraphError(LLMError):
    """The model's reply is not a graph in the expected shape."""


def prompt_digest() -> str:
    return f"{digest(*TEMPLATES)}.{CODE_VERSION}"


def entity_key(name: str) -> str:
    """One key per thing: Unicode-normalised (full-width → ASCII), case-folded, single spaces."""
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


# --- ingest ---------------------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _json_object(reply: str) -> dict[str, Any]:
    fenced = _FENCE.search(reply)
    text = fenced.group(1) if fenced else reply
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise GraphError(f"no JSON object in the reply: {reply[:200]!r}")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise GraphError(f"the reply is not valid JSON ({exc}): {reply[:200]!r}") from exc
    if not isinstance(value, dict):
        raise GraphError(f"the reply is not a JSON object: {reply[:200]!r}")
    return value


def _items(value: dict[str, Any], field: str) -> list[dict[str, Any]]:
    if field not in value:  # "nothing found" is an empty list, not a missing field
        raise GraphError(f"the reply has no {field!r} list: {value!r:.200}")
    items = value[field]
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise GraphError(f"{field!r} must be a list of objects, got {items!r:.200}")
    return items


def _text(item: dict[str, Any], field: str, *, required: bool) -> str:
    value = item.get(field)
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise GraphError(f"{field!r} must be a string, got {item!r:.200}")
    return value.strip()


def parse_graph(reply: str) -> ChunkGraph:
    """The graph in a model's reply: `{"entities": [{name, type}], "relations": [{source,
    target, description}]}` — both lists present, possibly empty — perhaps in a code fence. Blank
    names, repeated entities and
    relations, and relations from an entity to itself are dropped; a relation end that is not
    among the entities is added as one (without a type). Anything else malformed raises."""
    value = _json_object(reply)
    entities: dict[str, Entity] = {}
    for item in _items(value, "entities"):
        name = _text(item, "name", required=True)
        key = entity_key(name)
        if key and key not in entities:
            entities[key] = Entity(key, name, _text(item, "type", required=False))
    relations: dict[tuple[str, str], Relation] = {}
    for item in _items(value, "relations"):
        ends = [_text(item, "source", required=True), _text(item, "target", required=True)]
        source, target = (entity_key(e) for e in ends)
        if not source or not target or source == target or (source, target) in relations:
            continue
        for key, name in ((source, ends[0]), (target, ends[1])):
            entities.setdefault(key, Entity(key, name, ""))
        description = _text(item, "description", required=False)
        relations[(source, target)] = Relation(source, target, description)
    return ChunkGraph(tuple(entities.values()), tuple(relations.values()))


class GraphExtractor:
    """One LLM call per chunk — the chunk's text only: given the document title, a local model
    lists the title itself as an entity of every chunk, and that hub links everything (eval,
    2026-11-12). Up to `concurrency` calls at once across every ingest sharing the extractor."""

    def __init__(self, llm: LLM, concurrency: int = 1) -> None:
        self._calls = BoundedCalls(llm, concurrency)

    def extract(self, chunks: list[Chunk]) -> list[ChunkGraph]:
        """One graph per chunk, in chunk order."""
        return self._calls.map(
            lambda c: parse_graph(self._calls.text(render_pair(*TEMPLATES, passage=c.text))), chunks
        )


# --- query ----------------------------------------------------------------------------------------


def seeds(names_in_query: Iterable[tuple[str, int]]) -> list[str]:
    """The entities a walk starts from, given (entity, inside) for each entity name found in the
    query (`inside`: how many other entity names contain it): at least `MIN_SEED_CHARS` long,
    and not a category word — a name inside `GENERIC_INSIDE` or more others ("council",
    "委员会")."""
    return sorted(
        entity
        for entity, inside in names_in_query
        if len(entity) >= MIN_SEED_CHARS and inside < GENERIC_INSIDE
    )


Mentions = Callable[[list[str]], Iterable[tuple[str, str]]]  # keys → (entity, chunk_id) pairs


def walk(
    seeds: list[str],
    chunks_of: Mentions,
    entities_of: Mentions,
    *,
    hops: int,
    hub_chunks: int,
    max_neighbors: int,
) -> list[str]:
    """The chunks reached from `seeds`, sorted: each hop goes from the frontier entities to the
    chunks that mention them, then to the other entities in those chunks. `chunks_of(entities)`
    and `entities_of(chunk_ids)` give (entity, chunk_id) pairs. A hub — an entity in more than
    `hub_chunks` chunks — is neither walked through nor a way to reach chunks. Each hop keeps the
    `max_neighbors` new entities linked by the most chunks (ties by key)."""
    seen = set(seeds)
    frontier = sorted(seeds)
    reached: set[str] = set()
    for hop in range(hops + 1):
        by_entity: dict[str, set[str]] = {}
        for entity, chunk in chunks_of(frontier):
            by_entity.setdefault(entity, set()).add(chunk)
        chunks = {c for cs in by_entity.values() if len(cs) <= hub_chunks for c in cs}
        reached |= chunks
        if hop == hops or not chunks:
            break
        links: dict[str, int] = {}
        for entity, _ in entities_of(sorted(chunks)):
            if entity not in seen:
                links[entity] = links.get(entity, 0) + 1
        best = sorted(links.items(), key=lambda kv: (-kv[1], kv[0]))[:max_neighbors]
        frontier = [entity for entity, _ in best]
        seen.update(frontier)
    return sorted(reached)
