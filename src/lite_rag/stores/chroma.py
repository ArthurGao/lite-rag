"""Chroma + SQLite store (design §4): the optional embedded backend, extra `[chroma]`.

- `<data_dir>/lite_rag.sqlite` holds what PgStore keeps in PostgreSQL except the vectors: kbs,
  documents, chunks, parents, BM25 postings and the vector cache. `<data_dir>/chroma/` holds the
  vectors, one collection per kb (cosine space), named from a hash of the kb key.
- BM25 is scored in Python (`keyword.bm25`); N, n(t) and avgdl are counted over the kb's ready
  chunks, before any metadata filter — the same statistics as PgStore.
- Metadata filters compare scalar values for equality (the engine checks `where`): the
  document's scalar metadata is copied onto each of its Chroma records.
- Rows and vectors share no transaction. A write runs in a fixed order — SQLite rows as
  `pending`, then Chroma, then `ready`; a delete marks `deleting`, removes the vectors, then the
  rows — and reads see `ready` documents only. Opening the store removes every document left
  `pending` or `deleting` by a crash (from both sides) and every collection without a kb; the
  caller ingests such a document again. A write that fails (rather than crashes) is cleaned up
  at once, so on this backend a failed upsert removes the document instead of keeping its old
  version.
- One process per data directory: a lock serialises every operation of the store.
"""

import hashlib
import json
import sqlite3
import threading
from array import array
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.client import Client
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from lite_rag.db import migration_files
from lite_rag.errors import KbNotFound, StoreError
from lite_rag.keyword import Bm25Inputs, Posting, bm25
from lite_rag.ranking import ranked
from lite_rag.stores.base import (
    SCALARS,
    IndexedChunk,
    KbInfo,
    KbRef,
    KbStats,
    StoredChunk,
    StoredDocument,
    StoredParent,
)
from lite_rag.types import Chunk, Document

MIGRATIONS = "lite_rag.migrations_sqlite"
SQLITE_FILE = "lite_rag.sqlite"
CHROMA_DIR = "chroma"
COLLECTION_PREFIX = "kb-"
META_PREFIX = "m."  # document metadata on a Chroma record, apart from our own `doc_id`


def _connect(path: Path) -> sqlite3.Connection:
    # Autocommit; transactions are explicit (`_tx`). One connection, guarded by the store's lock.
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _applied(conn: sqlite3.Connection) -> set[str]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY,"
        " applied_at text NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    return {v for (v,) in conn.execute("SELECT version FROM schema_migrations")}


def migrate_sqlite(data_dir: str | Path) -> list[str]:
    """Create `data_dir` and apply pending SQLite migrations; returns the versions applied."""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    conn = _connect(path / SQLITE_FILE)
    try:
        done = _applied(conn)
        applied = []
        for version, script in migration_files(MIGRATIONS):
            if version in done:
                continue
            # executescript commits first; BEGIN … COMMIT makes the file and its record atomic
            conn.executescript(
                f"BEGIN IMMEDIATE;\n{script}\n"
                f"INSERT INTO schema_migrations (version) VALUES ('{version}');\nCOMMIT;"
            )
            applied.append(version)
        return applied
    finally:
        conn.close()


def collection_name(kb: KbRef) -> str:
    """Chroma allows [a-zA-Z0-9._-] only; a kb key has ':' and '/'."""
    return COLLECTION_PREFIX + hashlib.sha256(kb.key.encode()).hexdigest()[:40]


def _chroma_where(where: dict[str, Any]) -> dict[str, Any] | None:
    conditions = [{META_PREFIX + k: {"$eq": v}} for k, v in where.items()]
    if len(conditions) < 2:  # Chroma's $and needs two or more
        return conditions[0] if conditions else None
    return {"$and": conditions}


def _same(a: Any, b: Any) -> bool:
    """Equality as Chroma's `$eq` sees it: a bool never equals a number."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return bool(a == b)


def _matches(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    return all(k in metadata and _same(metadata[k], v) for k, v in where.items())


def _record_metadata(doc: Document) -> dict[str, Any]:
    scalars = {META_PREFIX + k: v for k, v in doc.metadata.items() if isinstance(v, SCALARS)}
    return {"doc_id": doc.doc_id, **scalars}


def _floats(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


class ChromaStore:
    """Implements both `Store` and `VectorCache`."""

    def __init__(self, data_dir: str | Path) -> None:
        path = Path(data_dir)
        sqlite_path = path / SQLITE_FILE
        if not sqlite_path.is_file():
            raise StoreError(f"no lite-rag data in {path}: run `literag migrate`")
        self._db = _connect(sqlite_path)
        pending = {v for v, _ in migration_files(MIGRATIONS)} - _applied(self._db)
        if pending:
            self._db.close()
            raise StoreError(f"the data in {path} is not up to date: run `literag migrate`")
        client = chromadb.PersistentClient(
            path=str(path / CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
        )
        assert isinstance(client, Client)  # the local client: `close` is not on ClientAPI
        self._client = client
        self._lock = threading.RLock()
        self._collections: dict[str, Collection] = {}
        self._recover()

    def close(self) -> None:
        with self._lock:
            self._db.close()
            self._client.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def _collection(self, kb: KbRef) -> Collection:
        name = collection_name(kb)
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
            )
        return self._collections[name]

    def _drop_collection(self, name: str) -> None:
        self._collections.pop(name, None)
        try:
            self._client.delete_collection(name)
        except NotFoundError:
            pass

    def _recover(self) -> None:
        """Finish what a crash interrupted: unfinished documents go from both sides, collections
        without a kb go too."""
        with self._lock:
            rows = self._db.execute(
                "SELECT k.namespace, k.kb, d.doc_id FROM documents d JOIN kbs k ON k.name = d.kb"
                " WHERE d.status <> 'ready'"
            ).fetchall()
            for namespace, name, doc_id in rows:
                self._discard(KbRef(namespace, name), doc_id)
            kbs = self._db.execute("SELECT namespace, kb FROM kbs").fetchall()
            known = {collection_name(KbRef(ns, kb)) for ns, kb in kbs}
            for col in self._client.list_collections():
                if col.name.startswith(COLLECTION_PREFIX) and col.name not in known:
                    self._drop_collection(col.name)

    def _discard(self, kb: KbRef, doc_id: str) -> None:
        """Remove a document's vectors, then its rows."""
        self._collection(kb).delete(where={"doc_id": doc_id})
        with self._tx() as db:
            db.execute("DELETE FROM documents WHERE kb = ? AND doc_id = ?", (kb.key, doc_id))

    # --- knowledge bases ---------------------------------------------------------------------

    def get_or_create_kb(self, info: KbInfo) -> KbInfo:
        with self._lock:
            self.create_kb(info)
            stored = self.get_kb(info.ref)
        assert stored is not None  # created above, or it existed
        return stored

    def create_kb(self, info: KbInfo) -> bool:
        ref = info.ref
        with self._lock:
            if self.get_kb(ref) is not None:
                return False
            # a collection left by a delete_kb that crashed holds old vectors: never reuse it
            self._drop_collection(collection_name(ref))
            with self._tx() as db:
                db.execute(
                    "INSERT INTO kbs (name, namespace, kb, embedding_provider, embedding_model,"
                    " dim, query_prefix, document_prefix, chunker)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ref.key,
                        ref.namespace,
                        ref.name,
                        info.embedding_provider,
                        info.embedding_model,
                        info.dim,
                        info.query_prefix,
                        info.document_prefix,
                        json.dumps(info.chunker),
                    ),
                )
            return True

    def get_kb(self, kb: KbRef) -> KbInfo | None:
        with self._lock:
            row = self._db.execute(
                "SELECT embedding_provider, embedding_model, dim, query_prefix, document_prefix,"
                " chunker FROM kbs WHERE name = ?",
                (kb.key,),
            ).fetchone()
        if row is None:
            return None
        provider, model, dim, query_prefix, document_prefix, chunker = row
        return KbInfo(kb, provider, model, dim, query_prefix, document_prefix, json.loads(chunker))

    def delete_kb(self, kb: KbRef) -> bool:
        with self._lock:
            with self._tx() as db:  # rows first: a crash before the next line leaves an
                # orphan collection, which opening the store (or creating the kb) removes
                existed = db.execute("DELETE FROM kbs WHERE name = ?", (kb.key,)).rowcount > 0
            self._drop_collection(collection_name(kb))
        return existed

    def list_kbs(self, namespace: str) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT kb FROM kbs WHERE namespace = ? ORDER BY kb", (namespace,)
            ).fetchall()
        return [kb for (kb,) in rows]

    def documents(self, kb: KbRef, page: int = 100) -> Iterator[StoredDocument]:
        after = ""
        while True:
            with self._lock:
                rows = self._db.execute(
                    "SELECT doc_id, title, text, metadata, source, pages FROM documents"
                    " WHERE kb = ? AND status = 'ready' AND doc_id > ? ORDER BY doc_id LIMIT ?",
                    (kb.key, after, page),
                ).fetchall()
            for doc_id, title, text, metadata, source, pages in rows:
                pages_ = tuple(json.loads(pages)) if pages else ()
                yield StoredDocument(doc_id, title, text, json.loads(metadata), source, pages_)
            if len(rows) < page:
                return
            after = rows[-1][0]

    def count_missing_text(self, kb: KbRef) -> int:
        return 0  # this backend has always stored the text (`documents.text` is NOT NULL)

    def stats(self, kb: KbRef) -> KbStats | None:
        with self._lock:
            row = self._db.execute(
                "SELECT k.embedding_provider, k.embedding_model, k.dim,"
                " (SELECT count(*) FROM documents d WHERE d.kb = k.name AND d.status = 'ready')"
                " FROM kbs k WHERE k.name = ?",
                (kb.key,),
            ).fetchone()
            if row is None:
                return None
            n_chunks, avg = self._totals(kb)
        provider, model, dim, docs = row
        return KbStats(kb.namespace, kb.name, docs, n_chunks, avg, f"{provider}/{model}", dim)

    def _totals(self, kb: KbRef) -> tuple[int, float]:
        """BM25's N and avgdl, over the kb's ready documents (avgdl 0 for an empty kb)."""
        row = self._db.execute(
            "SELECT count(*), coalesce(sum(c.length), 0) FROM chunks c"
            " JOIN documents d ON d.kb = c.kb AND d.doc_id = c.doc_id"
            " WHERE c.kb = ? AND d.status = 'ready'",
            (kb.key,),
        ).fetchone()
        n_chunks, total = int(row[0]), int(row[1])
        return n_chunks, total / n_chunks if n_chunks else 0.0

    # --- retrieval ---------------------------------------------------------------------------

    def vector_query(
        self, kb: KbRef, vector: list[float], n: int, where: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        flt = _chroma_where(where or {})
        with self._lock:
            result = self._collection(kb).query(
                query_embeddings=vector, n_results=n, where=flt, include=["distances"]
            )
            found = result["ids"][0]
            distances = (result["distances"] or [[]])[0]
            ready = self._ready(kb, found)
        # cosine distance → similarity; the same order (and tie-break) as every other ranking
        return ranked(
            {c: 1 - float(d) for c, d in zip(found, distances, strict=True) if c in ready}
        )

    def _ready(self, kb: KbRef, chunk_ids: list[str]) -> set[str]:
        rows = self._db.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.kb = c.kb AND d.doc_id = c.doc_id"
            " WHERE c.kb = ? AND d.status = 'ready' AND c.id IN (SELECT value FROM json_each(?))",
            (kb.key, json.dumps(chunk_ids)),
        ).fetchall()
        return {c for (c,) in rows}

    def bm25_query(
        self,
        kb: KbRef,
        terms: list[str],
        n: int,
        k1: float,
        b: float,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        flt = where or {}
        if not terms:
            return []
        with self._lock:
            n_chunks, avg = self._totals(kb)
            rows = self._db.execute(
                "SELECT p.term, p.chunk_id, p.tf, p.length, p.doc_id FROM postings p"
                " JOIN documents d ON d.kb = p.kb AND d.doc_id = p.doc_id"
                " WHERE p.kb = ? AND d.status = 'ready'"
                " AND p.term IN (SELECT DISTINCT value FROM json_each(?))",
                (kb.key, json.dumps(terms)),
            ).fetchall()
            allowed = self._matching_docs(kb, {doc for *_, doc in rows}, flt) if flt else None
        doc_freq = Counter(term for term, *_ in rows)  # before the filter: the kb's statistics
        postings = [
            Posting(term, chunk, tf, length)
            for term, chunk, tf, length, doc in rows
            if allowed is None or doc in allowed
        ]
        return bm25(Bm25Inputs(n_chunks, avg, dict(doc_freq), postings), n, k1, b)

    def _matching_docs(self, kb: KbRef, doc_ids: set[str], where: dict[str, Any]) -> set[str]:
        """Which of `doc_ids` match `where`: each document's metadata is decoded once."""
        rows = self._db.execute(
            "SELECT doc_id, metadata FROM documents"
            " WHERE kb = ? AND doc_id IN (SELECT value FROM json_each(?))",
            (kb.key, json.dumps(sorted(doc_ids))),
        ).fetchall()
        return {doc for doc, metadata in rows if _matches(json.loads(metadata), where)}

    def chunks(self, kb: KbRef, chunk_ids: list[str]) -> dict[str, StoredChunk]:
        if not chunk_ids:
            return {}
        with self._lock:
            rows = self._db.execute(
                "SELECT c.id, c.doc_id, d.title, c.text, c.start_offset, c.end_offset,"
                " c.heading_path, c.page,"
                " p.n, p.text, p.start_offset, p.end_offset, p.heading_path, p.page"
                " FROM chunks c"
                " JOIN documents d ON d.kb = c.kb AND d.doc_id = c.doc_id"
                " LEFT JOIN parents p ON p.kb = c.kb AND p.doc_id = c.doc_id AND p.n = c.parent_n"
                " WHERE c.kb = ? AND d.status = 'ready'"
                " AND c.id IN (SELECT value FROM json_each(?))",
                (kb.key, json.dumps(chunk_ids)),
            ).fetchall()
        out = {}
        for chunk, doc, title, text, start, end, path, page, pn, pt, ps, pe, pp, ppage in rows:
            parent = (
                StoredParent(pn, pt, ps, pe, tuple(json.loads(pp)), ppage)
                if pn is not None
                else None
            )
            out[chunk] = StoredChunk(
                chunk, doc, title, text, start, end, tuple(json.loads(path)), page, parent
            )
        return out

    # --- documents ---------------------------------------------------------------------------

    def document_hash(self, kb: KbRef, doc_id: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT content_hash FROM documents"
                " WHERE kb = ? AND doc_id = ? AND status = 'ready'",
                (kb.key, doc_id),
            ).fetchone()
        return row[0] if row else None

    def upsert_document(
        self,
        kb: KbRef,
        doc: Document,
        content_hash: str,
        chunks: list[IndexedChunk],
        parents: list[Chunk] | None = None,
    ) -> None:
        with self._lock:
            info = self.get_kb(kb)
            if info is None:
                raise KbNotFound.of(kb.name, kb.namespace)
            for c in chunks:  # before anything is written
                if len(c.vector) != info.dim:
                    raise StoreError(
                        f"{c.chunk.chunk_id}: {len(c.vector)}-d vector in a {info.dim}-d kb"
                    )
            with self._tx() as db:  # 1. rows, as pending (the old version goes)
                old = db.execute(
                    "SELECT id FROM chunks WHERE kb = ? AND doc_id = ?", (kb.key, doc.doc_id)
                ).fetchall()
                db.execute(
                    "DELETE FROM documents WHERE kb = ? AND doc_id = ?", (kb.key, doc.doc_id)
                )
                self._insert(db, kb, doc, content_hash, chunks, parents or [])
            try:
                collection = self._collection(kb)  # 2. vectors: the old ones out, the new in
                if old:  # by id: a filtered delete would scan the collection on every insert
                    collection.delete(ids=[c for (c,) in old])
                self._add_vectors(collection, doc, chunks)
                with self._tx() as db:  # 3. visible
                    db.execute(
                        "UPDATE documents SET status = 'ready' WHERE kb = ? AND doc_id = ?",
                        (kb.key, doc.doc_id),
                    )
            except Exception as exc:
                self._cleanup(kb, doc.doc_id, exc)
                raise StoreError(f"writing {doc.doc_id!r} to Chroma failed: {exc}") from exc

    def delete_document(self, kb: KbRef, doc_id: str) -> bool:
        """False when the document — or the whole kb — does not exist."""
        with self._lock:
            with self._tx() as db:
                marked = db.execute(
                    "UPDATE documents SET status = 'deleting' WHERE kb = ? AND doc_id = ?",
                    (kb.key, doc_id),
                ).rowcount
            if not marked:
                return False
            try:
                self._discard(kb, doc_id)
            except Exception as exc:  # stays `deleting`: hidden, and removed on the next open
                raise StoreError(f"deleting {doc_id!r} from Chroma failed: {exc}") from exc
            return True

    def _cleanup(self, kb: KbRef, doc_id: str, cause: Exception) -> None:
        try:
            self._discard(kb, doc_id)
        except Exception as exc:  # still `pending`: hidden, and removed on the next open
            cause.add_note(f"cleanup failed, finished when the store next opens: {exc}")

    def _add_vectors(
        self, collection: Collection, doc: Document, chunks: list[IndexedChunk]
    ) -> None:
        metadata = _record_metadata(doc)
        step = self._client.get_max_batch_size()
        for i in range(0, len(chunks), step):
            batch = chunks[i : i + step]
            vectors: list[Sequence[float] | Sequence[int]] = [c.vector for c in batch]
            collection.add(
                ids=[c.chunk.chunk_id for c in batch],
                embeddings=vectors,
                metadatas=[metadata] * len(batch),
            )

    def _insert(
        self,
        db: sqlite3.Connection,
        kb: KbRef,
        doc: Document,
        content_hash: str,
        chunks: list[IndexedChunk],
        parents: list[Chunk],
    ) -> None:
        db.execute(
            "INSERT INTO documents (kb, doc_id, title, source, content_hash, chunk_count,"
            " metadata, text, pages, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (
                kb.key,
                doc.doc_id,
                doc.title,
                doc.source,
                content_hash,
                len(chunks),
                json.dumps(doc.metadata),
                doc.text,
                json.dumps(list(doc.pages)) if doc.pages else None,
            ),
        )
        db.executemany(  # first: chunks reference them
            "INSERT INTO parents (kb, doc_id, n, text, start_offset, end_offset, heading_path,"
            " page) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (kb.key, p.doc_id, p.n, p.text, p.start, p.end, json.dumps(p.heading_path), p.page)
                for p in parents
            ],
        )
        db.executemany(
            "INSERT INTO chunks (kb, id, doc_id, n, text, start_offset, end_offset, tokens,"
            " length, heading_path, page, parent_n, context)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    kb.key,
                    c.chunk.chunk_id,
                    c.chunk.doc_id,
                    c.chunk.n,
                    c.chunk.text,
                    c.chunk.start,
                    c.chunk.end,
                    c.chunk.tokens,
                    len(c.terms),
                    json.dumps(c.chunk.heading_path),
                    c.chunk.page,
                    c.chunk.parent,
                    c.context,
                )
                for c in chunks
            ],
        )
        db.executemany(
            "INSERT INTO postings (kb, term, chunk_id, doc_id, tf, length)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (kb.key, term, c.chunk.chunk_id, c.chunk.doc_id, tf, len(c.terms))
                for c in chunks
                for term, tf in Counter(c.terms).items()
            ],
        )

    # --- vector cache ------------------------------------------------------------------------

    def get(self, model_id: str, dim: int, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        with self._lock:
            rows = self._db.execute(
                "SELECT key, embedding FROM embedding_cache WHERE model_id = ? AND dim = ?"
                " AND key IN (SELECT value FROM json_each(?))",
                (model_id, dim, json.dumps(keys)),
            ).fetchall()
        return {key: _floats(blob) for key, blob in rows}

    def put(self, model_id: str, dim: int, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            return
        with self._lock, self._tx() as db:
            db.executemany(
                "INSERT OR IGNORE INTO embedding_cache (model_id, dim, key, embedding)"
                " VALUES (?, ?, ?, ?)",
                [(model_id, dim, key, array("f", v).tobytes()) for key, v in vectors.items()],
            )
