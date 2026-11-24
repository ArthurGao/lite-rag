"""A generic provenance ledger (lite-rag lineage design §5): what was derived from what, in
which version, and when each version stopped being current. It knows nothing of RAG and imports
nothing else of lite-rag, so other projects can use it on their own connection:

    migrate(conn)                       # once, in the host's migration step
    ledger = ledger_for(conn)           # inside the host's transaction
    doc = ledger.record("ns", "document", "a", content_hash)
"""

from lite_rag.lineage.ledger import Ledger, PgLedger, SqliteLedger, ledger_for, utcnow
from lite_rag.lineage.migrate import migrate, pending
from lite_rag.lineage.types import Artifact, Edge, Ended, Item, Recorded, Related, Status

__all__ = [
    "Artifact",
    "Edge",
    "Ended",
    "Item",
    "Ledger",
    "PgLedger",
    "Recorded",
    "Related",
    "SqliteLedger",
    "Status",
    "ledger_for",
    "migrate",
    "pending",
    "utcnow",
]
