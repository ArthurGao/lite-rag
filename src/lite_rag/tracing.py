"""Query traces (lineage design §6.3): which chunk versions each stage of a search or answer
used, stored next to the ledger and read back against it — was the version still current,
has it since been replaced or deleted (`stale`, with its old text unless gc cleared it or the
document was deleted), or did the ledger never record it."""

import sqlite3
from collections.abc import Sequence
from dataclasses import astuple, dataclass, replace
from datetime import datetime
from typing import Any, Literal

import psycopg

from lite_rag.lineage import Artifact, Ledger, dialect_for
from lite_rag.stores.base import KbRef

Stage = Literal["retrieved", "reranked", "in_context", "cited"]
STAGES: tuple[Stage, ...] = ("retrieved", "reranked", "in_context", "cited")  # pipeline order
State = Literal["current", "stale", "unrecorded"]


@dataclass(frozen=True)
class TraceRecord:
    """One traced search or answer. `kbs` are names in `namespace`; `query` is the question,
    its sha256 or None (settings `trace.query`)."""

    id: str
    namespace: str
    kbs: list[str]
    operation: Literal["search", "ask"]
    query: str | None
    retrieval: dict[str, Any]
    created_at: datetime  # when the chunks were read
    model_id: str | None = None
    usage: dict[str, int] | None = None
    cached: bool | None = None
    status: Literal["answered", "no_sources", "failed"] | None = None


@dataclass(frozen=True)
class TraceSource:
    """A chunk one stage used: its kb (`artifact_ns`, the kb's physical key) and ledger key as
    read at retrieval (None: written before lineage and not backfilled)."""

    artifact_ns: str
    artifact_key: str | None
    chunk_id: str
    stage: Stage
    rank: int | None
    score: float | None


TRACE_COLUMNS = (
    "id, namespace, kbs, operation, query, retrieval, created_at, model_id, usage, cached, status"
)
SOURCE_COLUMNS = "artifact_ns, artifact_key, chunk_id, stage, rank, score"


class TraceLog:
    """The trace tables, inside the caller's transaction, on PostgreSQL or SQLite."""

    def __init__(self, conn: psycopg.Connection | sqlite3.Connection) -> None:
        self._d = dialect_for(conn)

    def write(self, record: TraceRecord, sources: Sequence[TraceSource]) -> None:
        d, r = self._d, record
        d.run(
            f"INSERT INTO rag_query_traces ({TRACE_COLUMNS})"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                r.id,
                r.namespace,
                d.to_json(r.kbs),
                r.operation,
                r.query,
                d.to_json(r.retrieval),
                d.to_time(r.created_at),
                r.model_id,
                None if r.usage is None else d.to_json(r.usage),
                r.cached,
                r.status,
            ),
        )
        self.add_sources(r.id, sources)

    def add_sources(self, trace_id: str, sources: Sequence[TraceSource]) -> None:
        """One already there is kept as it was."""
        self._d.run_many(
            f"INSERT INTO rag_query_sources (trace_id, {SOURCE_COLUMNS})"
            " VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [(trace_id, *astuple(s)) for s in sources],
        )

    def get(self, namespace: str, trace_id: str) -> tuple[TraceRecord, list[TraceSource]] | None:
        """A trace of `namespace` with its sources in pipeline order; None for one that is not
        there (or belongs to another namespace)."""
        d = self._d
        rows = d.rows(
            f"SELECT {TRACE_COLUMNS} FROM rag_query_traces WHERE id = ? AND namespace = ?",
            (trace_id, namespace),
        )
        if not rows:
            return None
        id_, ns, kbs, op, query, retrieval, created, model, usage, cached, status = rows[0]
        record = TraceRecord(
            id_,
            ns,
            list(d.from_json(kbs)),
            op,
            query,
            dict(d.from_json(retrieval)),
            d.from_time(created),
            model,
            None if usage is None else dict(d.from_json(usage)),
            None if cached is None else bool(cached),
            status,
        )
        rows = d.rows(
            f"SELECT {SOURCE_COLUMNS} FROM rag_query_sources WHERE trace_id = ?", (trace_id,)
        )
        stored = [TraceSource(*r) for r in rows]
        found = stored
        if not any(s.stage == "in_context" for s in stored):  # S25c traces stored it
            found = stored + in_context(stored, int(record.retrieval["top_k"]))
        return record, sorted(found, key=lambda s: (STAGES.index(s.stage), s.rank or 0))

    def delete_before(self, before: datetime, *, limit: int) -> int:
        """Delete up to `limit` traces made before `before` (their sources cascade); returns
        how many went."""
        return self._d.run(
            "DELETE FROM rag_query_traces WHERE id IN"
            " (SELECT id FROM rag_query_traces WHERE created_at < ? ORDER BY created_at LIMIT ?)",
            (self._d.to_time(before), limit),
        )

    def close(self, namespace: str, trace_id: str, cited: Sequence[TraceSource]) -> bool:
        """Give an open trace (a traced search: no outcome yet) its answer's citations, once:
        the status update and the rows in one statement pair, so of two callers one wins.
        False when the trace is not open (an answer's own, or closed before)."""
        closed = self._d.run(
            "UPDATE rag_query_traces SET status = 'answered'"
            " WHERE id = ? AND namespace = ? AND status IS NULL",
            (trace_id, namespace),
        )
        if closed:
            self.add_sources(trace_id, cited)
        return bool(closed)

    def delete_namespace(self, namespace: str, *, limit: int) -> int:
        """Delete up to `limit` traces of `namespace` (their sources cascade)."""
        return self._d.run(
            "DELETE FROM rag_query_traces WHERE id IN"
            " (SELECT id FROM rag_query_traces WHERE namespace = ? LIMIT ?)",
            (namespace, limit),
        )

    def ids(self, namespace: str) -> list[str]:
        """The namespace's traces, oldest first."""
        rows = self._d.rows(
            "SELECT id FROM rag_query_traces WHERE namespace = ? ORDER BY created_at, id",
            (namespace,),
        )
        return [i for (i,) in rows]


def in_context(stored: Sequence[TraceSource], top_k: int) -> list[TraceSource]:
    """The hits that were returned (and put in the prompt): the first `top_k` of the last
    ranking stage — derived on read rather than stored, since with rerank off (the default)
    they are exactly the retrieved stage. The engine's side of this rule: `_Retrieved.context`
    is `candidates[:top_k]` of that same ranking."""
    last = "reranked" if any(s.stage == "reranked" for s in stored) else "retrieved"
    return [
        replace(s, stage="in_context") for s in stored if s.stage == last and (s.rank or 0) <= top_k
    ]


@dataclass(frozen=True)
class TracedSource:
    source: TraceSource
    kb: str  # the kb's name in the trace's namespace
    artifact: Artifact | None  # the version the trace used; None when unrecorded

    @property
    def state(self) -> State:
        if self.artifact is None:
            return "unrecorded"
        return "current" if self.artifact.ended_at is None else "stale"


@dataclass(frozen=True)
class TraceReport:
    record: TraceRecord
    sources: list[TracedSource]


def report(
    record: TraceRecord, sources: Sequence[TraceSource], ledger: Ledger | None
) -> TraceReport:
    """`ledger` None (lineage off): every source is unrecorded. A chunk several stages used is
    looked up once."""
    names = {KbRef(record.namespace, kb).key: kb for kb in record.kbs}
    versions: dict[tuple[str, str], Artifact | None] = {}
    out = []
    for s in sources:
        version = None
        if ledger is not None and s.artifact_key:
            at = (s.artifact_ns, s.artifact_key)
            if at not in versions:
                # The key names the text read; the version that had begun by the trace time is
                # the one read, even if replaced between the read and the clock reading.
                versions[at] = ledger.started_by(at[0], "chunk", at[1], record.created_at)
            version = versions[at]
        out.append(TracedSource(s, names.get(s.artifact_ns, s.artifact_ns), version))
    return TraceReport(record, out)
