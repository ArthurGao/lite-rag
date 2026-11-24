"""The ledger itself (lineage design §5.3). It works inside the caller's connection and
transaction — never opening, committing or rolling back one — so a host records provenance in
the same transaction as the data it describes. `PgLedger` (psycopg) and `SqliteLedger` (sqlite3)
run the same SQL; each states its own parameter style and how it encodes JSON and times.
`ledger_for(conn)` picks the one that fits the connection.

Times come from `clock` (Python, UTC), not the database: SQLite's CURRENT_TIMESTAMP has second
resolution, and `as_of` needs versions ordered exactly."""

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Jsonb

from lite_rag.lineage.types import Artifact, Edge, Ended, Recorded, Related

Clock = Callable[[], datetime]
# ids per IN (...) list: PostgreSQL allows 65535 parameters, SQLite 32766
BATCH = 1000
MAX_DEPTH = 64  # derivation chains are a few levels deep; this only stops a linked cycle

FIELDS = (
    "id",
    "namespace",
    "kind",
    "key",
    "version",
    "content_hash",
    "producer",
    "payload",
    "metadata",
    "status",
    "valid_from",
    "ended_at",
)
COLUMNS = ", ".join(FIELDS)
A_COLUMNS = ", ".join(f"a.{f}" for f in FIELDS)  # the same, from lineage_artifacts AS a
N = len(FIELDS)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Ledger(Protocol):
    """Concurrency: the caller serialises writers to the namespaces a write touches (lite-rag
    holds its kb row lock, which is the namespace). Two transactions recording the same key, or
    retiring two parents of one child, at the same time could otherwise collide on a version
    or leave an orphan active."""

    def record(
        self,
        namespace: str,
        kind: str,
        key: str,
        content_hash: str,
        *,
        parents: Sequence[str] = (),
        producer: str | None = None,
        payload: str | None = None,
        metadata: dict[str, Any] | None = None,
        edge_metadata: dict[str, Any] | None = None,
    ) -> Recorded:
        """The active artifact of `(namespace, kind, key)` when it has this `content_hash`
        (reused: edges from `parents` added, an existing edge's metadata replaced); otherwise a
        new version, superseding the active one without touching its descendants."""
        ...

    def retire(self, artifact_id: str, *, status: Ended = "superseded") -> list[Artifact]:
        """End the artifact if active, then every descendant left without an active parent;
        returns what ended now, level by level (the artifact first)."""
        ...

    def retire_namespace(self, namespace: str, *, status: Ended = "deleted") -> int: ...

    def active(self, namespace: str, kind: str, key: str) -> Artifact | None: ...

    def get(self, ids: Sequence[str]) -> dict[str, Artifact]: ...

    def children(self, artifact_id: str) -> list[Related]: ...

    def as_of(self, namespace: str, kind: str, key: str, at: datetime) -> Artifact | None:
        """The version that was current at `at` (None: none yet, or deleted by then)."""
        ...

    def upstream(self, artifact_id: str) -> list[Related]:
        """Everything it was derived from, nearest first, each once."""
        ...

    def downstream(self, artifact_id: str) -> list[Related]:
        """Everything derived from it (impact analysis), nearest first, each once."""
        ...

    def gc(self, ended_before: datetime) -> int:
        """Clear the payload of artifacts that ended before `ended_before`; rows stay, so
        history and traces keep resolving. Returns how many were cleared."""
        ...


class _SqlLedger(ABC):
    """The shared SQL, written with `?` placeholders; a dialect states the rest."""

    def __init__(self, conn: Any, clock: Clock) -> None:
        self._conn = conn
        self._clock = clock

    # --- dialect ------------------------------------------------------------------------------

    @abstractmethod
    def _sql(self, query: str) -> str: ...

    @abstractmethod
    def _to_json(self, value: dict[str, Any]) -> Any: ...

    @abstractmethod
    def _from_json(self, value: Any) -> dict[str, Any]: ...

    @abstractmethod
    def _to_time(self, value: datetime) -> Any: ...

    @abstractmethod
    def _from_time(self, value: Any) -> datetime: ...

    # --- plumbing -----------------------------------------------------------------------------

    def _rows(self, query: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        return list(self._conn.execute(self._sql(query), tuple(params)).fetchall())

    def _run(self, query: str, params: Sequence[Any] = ()) -> int:
        return int(self._conn.execute(self._sql(query), tuple(params)).rowcount)

    def _artifact(self, row: Sequence[Any]) -> Artifact:
        id_, ns, kind, key, version, h, producer, payload, meta, status, start, end = row
        return Artifact(
            id=id_,
            namespace=ns,
            kind=kind,
            key=key,
            version=int(version),
            content_hash=h,
            producer=producer,
            payload=payload,
            metadata=self._from_json(meta),
            status=status,
            valid_from=self._from_time(start),
            ended_at=None if end is None else self._from_time(end),
        )

    def _artifacts(self, query: str, params: Sequence[Any] = ()) -> list[Artifact]:
        return [self._artifact(r) for r in self._rows(query, params)]

    @staticmethod
    def _marks(values: Sequence[Any]) -> str:
        return ", ".join("?" * len(values))

    # --- reads --------------------------------------------------------------------------------

    def active(self, namespace: str, kind: str, key: str) -> Artifact | None:
        found = self._artifacts(
            f"SELECT {COLUMNS} FROM lineage_artifacts"
            " WHERE namespace = ? AND kind = ? AND key = ? AND status = 'active'",
            (namespace, kind, key),
        )
        return found[0] if found else None

    def get(self, ids: Sequence[str]) -> dict[str, Artifact]:
        if not ids:
            return {}
        return {
            a.id: a
            for a in self._artifacts(
                f"SELECT {COLUMNS} FROM lineage_artifacts WHERE id IN ({self._marks(ids)})", ids
            )
        }

    def children(self, artifact_id: str) -> list[Related]:
        rows = self._rows(
            f"SELECT {A_COLUMNS}, e.parent_id, e.child_id, e.metadata FROM lineage_edges e"
            " JOIN lineage_artifacts a ON a.id = e.child_id WHERE e.parent_id = ?"
            " ORDER BY a.valid_from, a.key",
            (artifact_id,),
        )
        return [
            Related(self._artifact(r[:N]), Edge(r[N], r[N + 1], self._from_json(r[N + 2])))
            for r in rows
        ]

    def as_of(self, namespace: str, kind: str, key: str, at: datetime) -> Artifact | None:
        when = self._to_time(at)
        found = self._artifacts(
            f"SELECT {COLUMNS} FROM lineage_artifacts"
            " WHERE namespace = ? AND kind = ? AND key = ? AND valid_from <= ?"
            " AND (ended_at IS NULL OR ended_at > ?) ORDER BY version DESC LIMIT 1",
            (namespace, kind, key, when, when),
        )
        return found[0] if found else None

    def upstream(self, artifact_id: str) -> list[Related]:
        return self._walk(artifact_id, start="child_id", step="parent_id")

    def downstream(self, artifact_id: str) -> list[Related]:
        return self._walk(artifact_id, start="parent_id", step="child_id")

    def _walk(self, artifact_id: str, *, start: str, step: str) -> list[Related]:
        """Breadth-first over edges from `start` to `step` in one recursive query; a node
        reached several ways keeps its nearest edge. Depth is bounded: a host that linked a
        cycle gets a finite walk, not a hung query."""
        rows = self._rows(
            "WITH RECURSIVE walk (node, parent_id, child_id, metadata, depth) AS ("
            f" SELECT e.{step}, e.parent_id, e.child_id, e.metadata, 1 FROM lineage_edges e"
            f" WHERE e.{start} = ?"
            f" UNION ALL SELECT e.{step}, e.parent_id, e.child_id, e.metadata, w.depth + 1"
            f" FROM lineage_edges e JOIN walk w ON e.{start} = w.node WHERE w.depth < ?)"
            f" SELECT {A_COLUMNS}, w.parent_id, w.child_id, w.metadata FROM walk w"
            " JOIN lineage_artifacts a ON a.id = w.node"
            " ORDER BY w.depth, a.valid_from, a.key",
            (artifact_id, MAX_DEPTH),
        )
        seen = {artifact_id}
        out = []
        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                out.append(
                    Related(self._artifact(r[:N]), Edge(r[N], r[N + 1], self._from_json(r[N + 2])))
                )
        return out

    # --- writes -------------------------------------------------------------------------------

    def record(
        self,
        namespace: str,
        kind: str,
        key: str,
        content_hash: str,
        *,
        parents: Sequence[str] = (),
        producer: str | None = None,
        payload: str | None = None,
        metadata: dict[str, Any] | None = None,
        edge_metadata: dict[str, Any] | None = None,
    ) -> Recorded:
        if parents:  # checked before anything is written
            known = {
                i
                for (i,) in self._rows(
                    f"SELECT id FROM lineage_artifacts WHERE id IN ({self._marks(parents)})",
                    parents,
                )
            }
            if missing := set(parents) - known:
                raise ValueError(f"unknown parent artifact(s): {', '.join(sorted(missing))}")
        current = self.active(namespace, kind, key)
        if current is not None and current.content_hash == content_hash:
            self._link(parents, current.id, edge_metadata)
            return Recorded(current, replaced=None, reused=True)
        now = self._to_time(self._clock())
        replaced = None
        if current is not None:
            [replaced] = self._end([current.id], "superseded", now)
            version = current.version + 1  # the active version is always the latest
        else:
            (last,) = self._rows(
                "SELECT max(version) FROM lineage_artifacts"
                " WHERE namespace = ? AND kind = ? AND key = ?",
                (namespace, kind, key),
            )[0]
            version = (last or 0) + 1
        [created] = self._artifacts(
            f"INSERT INTO lineage_artifacts ({COLUMNS})"
            f" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, NULL) RETURNING {COLUMNS}",
            (
                uuid.uuid4().hex,
                namespace,
                kind,
                key,
                version,
                content_hash,
                producer,
                payload,
                self._to_json(metadata or {}),
                now,
            ),
        )
        self._link(parents, created.id, edge_metadata)
        return Recorded(created, replaced=replaced, reused=False)

    def _link(self, parents: Sequence[str], child_id: str, metadata: dict[str, Any] | None) -> None:
        for parent in parents:
            self._run(
                "INSERT INTO lineage_edges (parent_id, child_id, metadata) VALUES (?, ?, ?)"
                " ON CONFLICT (parent_id, child_id) DO UPDATE SET metadata = excluded.metadata",
                (parent, child_id, self._to_json(metadata or {})),
            )

    def _end(self, ids: Sequence[str], status: Ended, now: Any) -> list[Artifact]:
        """End those of `ids` still active; returns them."""
        if not ids:
            return []
        return self._artifacts(
            "UPDATE lineage_artifacts SET status = ?, ended_at = ?"
            f" WHERE status = 'active' AND id IN ({self._marks(ids)}) RETURNING {COLUMNS}",
            (status, now, *ids),
        )

    def retire(self, artifact_id: str, *, status: Ended = "superseded") -> list[Artifact]:
        now = self._to_time(self._clock())
        ended = self._end([artifact_id], status, now)
        return ended + self._cascade([artifact_id], status, now)

    def _cascade(self, frontier: list[str], status: Ended, now: Any) -> list[Artifact]:
        """End, level by level, the descendants of `frontier` left without an active parent —
        one statement per level."""
        ended: list[Artifact] = []
        while frontier:
            level = []
            for i in range(0, len(frontier), BATCH):
                part = frontier[i : i + BATCH]
                level += self._artifacts(
                    "UPDATE lineage_artifacts SET status = ?, ended_at = ?"
                    " WHERE status = 'active' AND id IN (SELECT child_id FROM lineage_edges"
                    f" WHERE parent_id IN ({self._marks(part)}))"
                    " AND NOT EXISTS (SELECT 1 FROM lineage_edges e"
                    " JOIN lineage_artifacts p ON p.id = e.parent_id"
                    " WHERE e.child_id = lineage_artifacts.id AND p.status = 'active')"
                    f" RETURNING {COLUMNS}",
                    (status, now, *part),
                )
            level.sort(key=lambda a: (a.valid_from, a.key))
            ended += level
            frontier = [a.id for a in level]
        return ended

    def retire_namespace(self, namespace: str, *, status: Ended = "deleted") -> int:
        """End every active artifact of `namespace`, and the orphans that leaves in other
        namespaces; returns how many ended."""
        now = self._to_time(self._clock())
        roots = [
            i
            for (i,) in self._rows(
                "UPDATE lineage_artifacts SET status = ?, ended_at = ?"
                " WHERE namespace = ? AND status = 'active' RETURNING id",
                (status, now, namespace),
            )
        ]
        return len(roots) + len(self._cascade(roots, status, now))

    def gc(self, ended_before: datetime) -> int:
        return self._run(
            "UPDATE lineage_artifacts SET payload = NULL"
            " WHERE status <> 'active' AND ended_at < ? AND payload IS NOT NULL",
            (self._to_time(ended_before),),
        )


class PgLedger(_SqlLedger):
    """psycopg: `%s` parameters (no query here contains a literal `?`), jsonb, timestamptz."""

    def __init__(self, conn: psycopg.Connection, *, clock: Clock = utcnow) -> None:
        super().__init__(conn, clock)

    def _sql(self, query: str) -> str:
        return query.replace("?", "%s")

    def _to_json(self, value: dict[str, Any]) -> Any:
        return Jsonb(value)

    def _from_json(self, value: Any) -> dict[str, Any]:
        return dict(value)

    def _to_time(self, value: datetime) -> Any:
        return value

    def _from_time(self, value: Any) -> datetime:
        assert isinstance(value, datetime)
        return value


class SqliteLedger(_SqlLedger):
    """sqlite3: JSON as text; times as fixed-width ISO-8601 UTC text, so text order is time
    order. Needs SQLite >= 3.35 (RETURNING) and foreign keys on."""

    def __init__(self, conn: sqlite3.Connection, *, clock: Clock = utcnow) -> None:
        super().__init__(conn, clock)

    def _sql(self, query: str) -> str:
        return query

    def _to_json(self, value: dict[str, Any]) -> Any:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _from_json(self, value: Any) -> dict[str, Any]:
        return dict(json.loads(value))

    def _to_time(self, value: datetime) -> Any:
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

    def _from_time(self, value: Any) -> datetime:
        return datetime.fromisoformat(value)


def ledger_for(conn: psycopg.Connection | sqlite3.Connection, *, clock: Clock = utcnow) -> Ledger:
    """The ledger for this connection's dialect."""
    if isinstance(conn, sqlite3.Connection):
        return SqliteLedger(conn, clock=clock)
    return PgLedger(conn, clock=clock)
