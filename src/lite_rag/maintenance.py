"""Store-wide upkeep of the lineage tables (`literag lineage gc`): every namespace at once."""

from datetime import datetime, timedelta

from lite_rag.errors import InvalidRequest
from lite_rag.lineage import utcnow
from lite_rag.stores.base import Store


def collect(
    store: Store,
    *,
    keep_text_days: int | None,
    retention_days: int,
    now: datetime | None = None,
    batch: int = 1000,
) -> tuple[int, int]:
    """Clear the text of artifacts that ended more than `keep_text_days` ago (None: lineage is
    off, leave the ledger alone) and delete traces older than `retention_days` — `batch` rows
    per transaction, so writers are not held up. Returns (texts cleared, traces deleted)."""
    if batch < 1:  # each round must be able to return less than a full batch, or it never ends
        raise InvalidRequest(f"batch must be at least 1, got {batch}")
    now = now or utcnow()
    cleared = 0
    if keep_text_days is not None:
        before = now - timedelta(days=keep_text_days)
        while True:
            with store.ledger() as ledger:
                n = ledger.gc(before, limit=batch)
            cleared += n
            if n < batch:
                break
    cutoff = now - timedelta(days=retention_days)
    deleted = 0
    while True:
        with store.traces() as log:
            n = log.delete_before(cutoff, limit=batch)
        deleted += n
        if n < batch:
            return cleared, deleted
