"""lite-rag's side of the lineage ledger (lineage design §6): what a document and its chunks are
recorded as. A chunk is known by its text, not its position: `{doc_id}/{text hash}/{k}`, `k`
counting repeats of the same text in the document — so a renumbered or re-contextualised chunk
with the same text stays the same artifact, and "stale" means its text is gone."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lite_rag.lineage import Item, Ledger
from lite_rag.stores.base import KbInfo
from lite_rag.types import Chunk, model_id


@dataclass(frozen=True)
class DocVersion:
    """What the ledger records of a document version."""

    doc_id: str
    title: str
    source: str | None
    content_hash: str


HASH_CHARS = 16  # of the sha256 hex in a key: enough to tell texts of one document apart


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_keys(doc_id: str, texts: Sequence[str]) -> list[str]:
    """One ledger key per text, in order."""
    seen: dict[str, int] = {}
    keys = []
    for text in texts:
        h = text_hash(text)[:HASH_CHARS]
        k = seen.get(h, 0)
        seen[h] = k + 1
        keys.append(f"{doc_id}/{h}/{k}")
    return keys


def producer(chunker: dict[str, Any]) -> str:
    """How a kb's chunks are made: its strategy and a digest of the whole chunker identity."""
    digest = hashlib.sha256(json.dumps(chunker, sort_keys=True).encode()).hexdigest()[:8]
    return f"chunker:{chunker.get('strategy', '?')}@{digest}"


def record_document(
    ledger: Ledger,
    kb: KbInfo,
    doc: DocVersion,
    chunks: Sequence[tuple[Chunk, str | None]],
    keys: Sequence[str],
) -> None:
    """Record a document version and its chunks (each with its contextual header, or None),
    inside the store's write transaction: chunks with unchanged text keep their artifact and
    gain an edge from the new version; those the new version no longer has end with the old
    one. What the kb embeds with is recorded once, on the document version."""
    ns = kb.ref.key
    rec = ledger.record(
        ns,
        "document",
        doc.doc_id,
        doc.content_hash,
        metadata={
            "title": doc.title,
            "source": doc.source,
            "embedding": model_id(kb.embedding_provider, kb.embedding_model),
            "dim": kb.dim,
        },
    )
    ledger.record_many(
        ns,
        "chunk",
        [
            Item(
                key,
                text_hash(ch.text),
                payload=ch.text,
                edge_metadata={
                    "chunk_id": ch.chunk_id,
                    "n": ch.n,
                    "start": ch.start,
                    "end": ch.end,
                    "page": ch.page,
                    "parent_n": ch.parent,
                    "context": context,
                },
            )
            for (ch, context), key in zip(chunks, keys, strict=True)
        ],
        parents=[rec.artifact.id],
        producer=producer(kb.chunker),
    )
    if rec.replaced is not None:
        ledger.retire(rec.replaced.id)


def recorded(ledger: Ledger, kb_key: str, doc_id: str, content_hash: str) -> bool:
    """Whether the ledger has this version of the document as its current one (backfill skips
    it without reading its chunks)."""
    current = ledger.active(kb_key, "document", doc_id)
    return current is not None and current.content_hash == content_hash


def retire_document(ledger: Ledger, kb_key: str, doc_id: str) -> None:
    """A document was deleted: it and every chunk only it held end as `deleted`."""
    doc = ledger.active(kb_key, "document", doc_id)
    if doc is not None:
        ledger.retire(doc.id, status="deleted")
