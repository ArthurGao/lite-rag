"""What the ledger stores (lineage design §5.1). It knows artifacts, versions and edges only:
`kind` is the caller's word ("document", "chunk", …) and is never interpreted here."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

Status = Literal["active", "superseded", "deleted"]
Ended = Literal["superseded", "deleted"]


@dataclass(frozen=True)
class Artifact:
    """One immutable version of something produced. `(namespace, kind, key)` is its logical
    identity, `version` counts from 1 within it; at most one version is active at a time."""

    id: str
    namespace: str
    kind: str
    key: str
    version: int
    content_hash: str
    producer: str | None
    payload: str | None  # None when not kept, or cleared by gc
    metadata: dict[str, Any]
    status: Status
    valid_from: datetime
    ended_at: datetime | None


@dataclass(frozen=True)
class Edge:
    """`child` was derived from `parent`; `metadata` describes the child within this parent
    (a position, say), which can differ between two parents of one child."""

    parent_id: str
    child_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Recorded:
    """What `record` did: `reused` an active artifact with the same content, or created a new
    version — `replaced` is the version it superseded (its descendants are not touched: retire
    it once the new version's children are recorded)."""

    artifact: Artifact
    replaced: Artifact | None
    reused: bool


@dataclass(frozen=True)
class Related:
    """A node found by walking edges, with the edge that reached it."""

    artifact: Artifact
    via: Edge


@dataclass(frozen=True)
class Item:
    """One of several artifacts recorded at once under the same parents (`record_many`)."""

    key: str
    content_hash: str
    payload: str | None = None
    edge_metadata: dict[str, Any] | None = None
