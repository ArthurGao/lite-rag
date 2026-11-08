"""PostgreSQL + pgvector store (design §4).

- A document write — delete the old version, write chunks, vectors and postings, adjust the BM25
  statistics on `kbs` — is ONE transaction, serialised per knowledge base by locking its `kbs` row.
  Rows go in with COPY (one round trip per table, however many chunks and terms).
- Vectors live in `embeddings_<dim>`, one table per dimension (an HNSW index needs a fixed one),
  created with the first kb of that dimension; up to 2000 dimensions as `vector`, larger as
  `halfvec` (pgvector's HNSW limits).
- Every pooled connection has `search_path = <schema>, public`, so SQL uses unqualified names.
"""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from lite_rag.config import IDENTIFIER
from lite_rag.db import connection_error, require_url
from lite_rag.errors import ConfigError, StoreError
from lite_rag.stores.base import IndexedChunk, KbInfo, KbRef, KbStats
from lite_rag.types import Document

MAX_VECTOR_DIM = 2000  # HNSW on `vector`; above it we store `halfvec` (HNSW up to 4000)
MAX_HALFVEC_DIM = 4000


def vector_sql(dim: int) -> sql.Composable:
    """The column type for `dim`-d vectors: `vector(dim)`, or `halfvec(dim)` above 2000."""
    if dim > MAX_HALFVEC_DIM:
        raise ConfigError(f"{dim}-d vectors are above pgvector's HNSW limit ({MAX_HALFVEC_DIM})")
    return sql.SQL(f"{'vector' if dim <= MAX_VECTOR_DIM else 'halfvec'}({dim})")


def _table(prefix: str, dim: int) -> sql.Identifier:
    return sql.Identifier(f"{prefix}_{dim}")


def _vector_text(vector: list[float]) -> str:
    """pgvector's text form, as COPY expects it."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgStore:
    """Implements both `Store` and `VectorCache`."""

    def __init__(self, database_url: str | None, schema: str, *, max_connections: int = 4) -> None:
        if not IDENTIFIER.match(schema):  # it goes into the connection's search_path option
            raise ConfigError(f"invalid schema {schema!r}: must be a lowercase SQL identifier")
        self._pool = ConnectionPool(
            require_url(database_url),
            min_size=1,
            max_size=max_connections,
            kwargs={"options": f"-c search_path={schema},public"},
            open=False,
            name="lite-rag",
        )
        try:
            self._pool.open(wait=True, timeout=10)
        except psycopg.Error as exc:
            raise connection_error(exc) from exc
        self._vector_tables: set[int] = set()
        self._cache_tables: set[int] = set()

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def _tx(self) -> Iterator[psycopg.Connection]:
        with self._pool.connection() as conn, conn.transaction():
            yield conn

    # --- knowledge bases ---------------------------------------------------------------------

    def get_or_create_kb(self, info: KbInfo) -> KbInfo:
        ref = info.ref
        with self._tx() as conn:
            created = conn.execute(
                "INSERT INTO kbs (name, namespace, kb, embedding_provider, embedding_model, dim,"
                " query_prefix, document_prefix, chunker)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (name) DO NOTHING",
                (
                    ref.key,
                    ref.namespace,
                    ref.name,
                    info.embedding_provider,
                    info.embedding_model,
                    info.dim,
                    info.query_prefix,
                    info.document_prefix,
                    Jsonb(info.chunker),
                ),
            ).rowcount
            if created:
                self._ensure_vector_table(conn, info.dim)
                return info
            row = conn.execute(
                "SELECT embedding_provider, embedding_model, dim, query_prefix, document_prefix,"
                " chunker FROM kbs WHERE name = %s",
                (ref.key,),
            ).fetchone()
        assert row is not None  # ON CONFLICT: it exists
        return KbInfo(ref, *row)

    def delete_kb(self, kb: KbRef) -> bool:
        with self._tx() as conn:
            # documents → chunks → postings and embeddings_<dim> all cascade from here
            return conn.execute("DELETE FROM kbs WHERE name = %s", (kb.key,)).rowcount > 0

    def stats(self, kb: KbRef) -> KbStats | None:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT k.n_chunks, k.total_length, k.embedding_provider, k.embedding_model, k.dim,"
                " (SELECT count(*) FROM documents d WHERE d.kb = k.name)"
                " FROM kbs k WHERE k.name = %s",
                (kb.key,),
            ).fetchone()
        if row is None:
            return None
        n_chunks, total, provider, model, dim, docs = row
        avg = total / n_chunks if n_chunks else 0.0
        return KbStats(kb.namespace, kb.name, docs, n_chunks, avg, f"{provider}/{model}", dim)

    # --- documents ---------------------------------------------------------------------------

    def document_hash(self, kb: KbRef, doc_id: str) -> str | None:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT content_hash FROM documents WHERE kb = %s AND doc_id = %s",
                (kb.key, doc_id),
            ).fetchone()
        return row[0] if row else None

    def upsert_document(
        self, kb: KbRef, doc: Document, content_hash: str, chunks: list[IndexedChunk]
    ) -> None:
        with self._tx() as conn:
            dim = self._lock_kb(conn, kb)
            for c in chunks:  # before anything is written
                if len(c.vector) != dim:
                    raise StoreError(
                        f"{c.chunk.chunk_id}: {len(c.vector)}-d vector in a {dim}-d kb"
                    )
            _, removed, removed_length = self._remove(conn, kb, doc.doc_id)
            conn.execute(
                "INSERT INTO documents (kb, doc_id, title, source, content_hash, chunk_count,"
                " metadata) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    kb.key,
                    doc.doc_id,
                    doc.title,
                    doc.source,
                    content_hash,
                    len(chunks),
                    Jsonb(doc.metadata),
                ),
            )
            self._copy_chunks(conn, kb, dim, chunks)
            self._adjust_stats(
                conn,
                kb,
                len(chunks) - removed,
                sum(len(c.terms) for c in chunks) - removed_length,
            )

    def delete_document(self, kb: KbRef, doc_id: str) -> bool:
        with self._tx() as conn:
            self._lock_kb(conn, kb)
            existed, removed, removed_length = self._remove(conn, kb, doc_id)
            self._adjust_stats(conn, kb, -removed, -removed_length)
            return existed

    # --- vector cache ------------------------------------------------------------------------

    def get(self, model_id: str, dim: int, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        with self._tx() as conn:
            self._ensure_cache_table(conn, dim)
            rows = conn.execute(
                sql.SQL(
                    "SELECT key, embedding::real[] FROM {} WHERE model_id = %s AND key = ANY(%s)"
                ).format(_table("embedding_cache", dim)),
                (model_id, keys),
            ).fetchall()
        return {key: list(vector) for key, vector in rows}

    def put(self, model_id: str, dim: int, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            return
        with self._tx() as conn, conn.cursor() as cur:
            self._ensure_cache_table(conn, dim)
            cur.executemany(  # ON CONFLICT rules out COPY here
                sql.SQL(
                    "INSERT INTO {} (model_id, key, embedding) VALUES (%s, %s, %s::{})"
                    " ON CONFLICT DO NOTHING"
                ).format(_table("embedding_cache", dim), vector_sql(dim)),
                [(model_id, key, _vector_text(v)) for key, v in vectors.items()],
            )

    # --- internals ---------------------------------------------------------------------------

    def _lock_kb(self, conn: psycopg.Connection, kb: KbRef) -> int:
        row = conn.execute("SELECT dim FROM kbs WHERE name = %s FOR UPDATE", (kb.key,)).fetchone()
        if row is None:
            raise StoreError(f"knowledge base {kb.key!r} does not exist")
        dim: int = row[0]
        return dim

    def _remove(self, conn: psycopg.Connection, kb: KbRef, doc_id: str) -> tuple[bool, int, int]:
        """Delete a document — its chunks, postings and vectors cascade. Returns whether it
        existed, and how many chunks and BM25 terms it took out of the kb."""
        row = conn.execute(
            "WITH gone AS (DELETE FROM documents WHERE kb = %s AND doc_id = %s RETURNING doc_id)"
            " SELECT (SELECT count(*) FROM gone),"
            " (SELECT count(*) FROM chunks WHERE kb = %s AND doc_id = %s),"
            " (SELECT coalesce(sum(length), 0) FROM chunks WHERE kb = %s AND doc_id = %s)",
            (kb.key, doc_id) * 3,
        ).fetchone()
        assert row is not None  # a SELECT of aggregates always returns a row
        existed, n_chunks, length = row  # the subqueries still see the rows being deleted
        return bool(existed), n_chunks, length

    def _adjust_stats(self, conn: psycopg.Connection, kb: KbRef, chunks: int, length: int) -> None:
        if chunks or length:
            conn.execute(
                "UPDATE kbs SET n_chunks = n_chunks + %s, total_length = total_length + %s"
                " WHERE name = %s",
                (chunks, length, kb.key),
            )

    def _copy_chunks(
        self, conn: psycopg.Connection, kb: KbRef, dim: int, chunks: list[IndexedChunk]
    ) -> None:
        if not chunks:
            return
        with conn.cursor() as cur:
            with cur.copy(
                "COPY chunks (kb, id, doc_id, n, text, start_offset, end_offset, tokens, length,"
                " heading_path) FROM STDIN"
            ) as copy:
                for c in chunks:
                    ch = c.chunk
                    copy.write_row(
                        (
                            kb.key,
                            ch.chunk_id,
                            ch.doc_id,
                            ch.n,
                            ch.text,
                            ch.start,
                            ch.end,
                            ch.tokens,
                            len(c.terms),
                            Jsonb(list(ch.heading_path)),
                        )
                    )
            with cur.copy(
                sql.SQL("COPY {} (kb, chunk_id, embedding) FROM STDIN").format(
                    _table("embeddings", dim)
                )
            ) as copy:
                for c in chunks:
                    copy.write_row((kb.key, c.chunk.chunk_id, _vector_text(c.vector)))
            with cur.copy("COPY postings (kb, term, chunk_id, tf) FROM STDIN") as copy:
                for c in chunks:
                    for term, tf in Counter(c.terms).items():
                        copy.write_row((kb.key, term, c.chunk.chunk_id, tf))

    def _ensure_vector_table(self, conn: psycopg.Connection, dim: int) -> None:
        if dim in self._vector_tables:
            return
        table, kind = _table("embeddings", dim), vector_sql(dim)
        ops = sql.SQL("vector_cosine_ops" if dim <= MAX_VECTOR_DIM else "halfvec_cosine_ops")
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (kb text NOT NULL, chunk_id text NOT NULL,"
                " embedding {} NOT NULL, PRIMARY KEY (kb, chunk_id),"
                " FOREIGN KEY (kb, chunk_id) REFERENCES chunks (kb, id) ON DELETE CASCADE)"
            ).format(table, kind)
        )
        conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding {})").format(
                sql.Identifier(f"ix_embeddings_{dim}_hnsw"), table, ops
            )
        )
        self._vector_tables.add(dim)

    def _ensure_cache_table(self, conn: psycopg.Connection, dim: int) -> None:
        if dim in self._cache_tables:
            return
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (model_id text NOT NULL, key text NOT NULL,"
                " embedding {} NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),"
                " PRIMARY KEY (model_id, key))"
            ).format(_table("embedding_cache", dim), vector_sql(dim))
        )
        self._cache_tables.add(dim)
