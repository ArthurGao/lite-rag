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
from typing import Any, Literal, overload

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from lite_rag.config import IDENTIFIER
from lite_rag.db import check_vector, connection_error, require_url
from lite_rag.errors import ConfigError, KbNotFound, StoreError
from lite_rag.keyword import Bm25Inputs, Posting, bm25
from lite_rag.stores.base import (
    IndexedChunk,
    KbInfo,
    KbRef,
    KbStats,
    StoredChunk,
    StoredParent,
)
from lite_rag.types import Chunk, Document

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


def _metadata_filter(
    kb: sql.Composable, doc_id: sql.Composable, where: dict[str, Any] | None
) -> tuple[sql.Composable, dict[str, Any]]:
    """`AND …` keeping rows whose document's metadata contains `where` (JSON containment: every
    key equals its value), and its parameters. Nothing when there is no filter."""
    if not where:
        return sql.SQL(""), {}
    clause = sql.SQL(
        " AND EXISTS (SELECT 1 FROM documents d WHERE d.kb = {kb} AND d.doc_id = {doc}"
        " AND d.metadata @> %(where)s)"
    ).format(kb=kb, doc=doc_id)
    return clause, {"where": Jsonb(where)}


# BM25 (keyword.py has the formula). One statistics block for both implementations: N and avgdl
# from `kbs`, n(t) counted over the whole kb BEFORE the metadata filter — a filter narrows which
# chunks may be returned, never the statistics.
_BM25_MATCHES = """
WITH q AS (SELECT DISTINCT unnest(%(terms)s::text[]) AS term),
stats AS (
    SELECT n_chunks::float8 AS n,
           CASE WHEN n_chunks > 0 THEN total_length::float8 / n_chunks ELSE 0 END AS avgdl
    FROM kbs WHERE name = %(kb)s),
matches AS (
    SELECT p.term, p.chunk_id, p.doc_id, p.tf, p.length,
           count(*) OVER (PARTITION BY p.term) AS df
    FROM postings p JOIN q USING (term) WHERE p.kb = %(kb)s)
"""
_BM25_SCORE = sql.SQL(
    _BM25_MATCHES
    + """
SELECT chunk_id, score FROM (
    SELECT m.chunk_id,
           sum(ln(1 + (s.n - m.df + 0.5) / (m.df + 0.5))                   -- idf
               * m.tf * (%(k1)s + 1)
               / (m.tf + %(k1)s * CASE WHEN s.avgdl > 0
                                       THEN 1 - %(b)s + %(b)s * m.length / s.avgdl
                                       ELSE 1 END)
               ORDER BY m.term) AS score                   -- fixed order, as keyword.bm25
    FROM matches m CROSS JOIN stats s
    WHERE TRUE {filter}
    GROUP BY m.chunk_id) scored
ORDER BY round(score::numeric, 9) DESC, chunk_id LIMIT %(n)s  -- ranking.PRECISION
"""
)
_BM25_INPUTS = sql.SQL(
    _BM25_MATCHES
    + """
SELECT m.term, m.chunk_id, m.tf, m.length, m.df, s.n, s.avgdl
FROM matches m CROSS JOIN stats s
WHERE TRUE {filter}
"""
)


def _bm25_inputs(rows: list[tuple[Any, ...]]) -> Bm25Inputs:
    if not rows:
        return Bm25Inputs(0, 0.0, {}, [])
    return Bm25Inputs(
        int(rows[0][5]),
        float(rows[0][6]),
        {term: int(df) for term, _, _, _, df, _, _ in rows},
        [Posting(term, chunk, tf, length) for term, chunk, tf, length, *_ in rows],
    )


class PgStore:
    """Implements both `Store` and `VectorCache`."""

    def __init__(
        self,
        database_url: str | None,
        schema: str,
        *,
        max_connections: int = 4,
        bm25_impl: Literal["sql", "python"] = "sql",
    ) -> None:
        """`bm25_impl`: score BM25 in SQL (fast) or in Python with `keyword.bm25` (readable);
        both use the same statistics and agree (tested)."""
        if not IDENTIFIER.match(schema):  # it goes into the connection's search_path option
            raise ConfigError(f"invalid schema {schema!r}: must be a lowercase SQL identifier")
        url = require_url(database_url)
        try:  # keep any options the caller put in the connection string, add the search_path
            own = conninfo_to_dict(url).get("options") or ""
        except psycopg.ProgrammingError as exc:
            raise StoreError(f"invalid database URL: {exc}") from exc
        options = f"{own} -c search_path={schema},public".strip()
        self._pool = ConnectionPool(
            url,
            min_size=1,
            max_size=max_connections,
            kwargs={"options": options},
            open=False,
            name="lite-rag",
        )
        try:
            self._pool.open(wait=True, timeout=10)
        except psycopg.Error as exc:
            raise connection_error(exc) from exc
        self._bm25_impl = bm25_impl
        self._vector_tables: set[int] = set()
        self._cache_tables: set[int] = set()
        with self._pool.connection() as conn:
            check_vector(conn)

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
            stored = self._get_kb(conn, ref)
        assert stored is not None  # ON CONFLICT: it exists
        return stored

    def get_kb(self, kb: KbRef) -> KbInfo | None:
        with self._tx() as conn:
            return self._get_kb(conn, kb)

    def _get_kb(self, conn: psycopg.Connection, kb: KbRef) -> KbInfo | None:
        row = conn.execute(
            "SELECT embedding_provider, embedding_model, dim, query_prefix, document_prefix,"
            " chunker FROM kbs WHERE name = %s",
            (kb.key,),
        ).fetchone()
        return KbInfo(kb, *row) if row else None

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

    # --- retrieval ---------------------------------------------------------------------------
    # The engine opens a kb (existence and identity) before any of these; they trust `kb`.

    def vector_query(
        self, kb: KbRef, vector: list[float], n: int, where: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        dim = len(vector)  # the engine checked it is the kb's dimension
        doc = sql.SQL("(SELECT c.doc_id FROM chunks c WHERE c.kb = e.kb AND c.id = e.chunk_id)")
        filter_sql, params = _metadata_filter(sql.SQL("e.kb"), doc, where)
        query = sql.SQL(
            "SELECT e.chunk_id, 1 - (e.embedding <=> %(q)s::{vt}) AS similarity"
            " FROM {table} e WHERE e.kb = %(kb)s {filter}"
            " ORDER BY e.embedding <=> %(q)s::{vt}, e.chunk_id LIMIT %(n)s"
        ).format(vt=vector_sql(dim), table=_table("embeddings", dim), filter=filter_sql)
        with self._tx() as conn:
            # The kb (and metadata) filter is applied after the HNSW scan; iterative scanning
            # keeps reading the index until `n` rows pass it (pgvector >= 0.8), in exact order.
            conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            rows = conn.execute(
                query, {"q": _vector_text(vector), "kb": kb.key, "n": n, **params}
            ).fetchall()
        return [(chunk, float(sim)) for chunk, sim in rows]

    def bm25_query(
        self,
        kb: KbRef,
        terms: list[str],
        n: int,
        k1: float,
        b: float,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if not terms:
            return []
        filter_sql, filter_params = _metadata_filter(sql.SQL("%(kb)s"), sql.SQL("m.doc_id"), where)
        params = {"terms": terms, "kb": kb.key, "k1": k1, "b": b, "n": n, **filter_params}
        with self._tx() as conn:
            if self._bm25_impl == "python":
                rows = conn.execute(_BM25_INPUTS.format(filter=filter_sql), params).fetchall()
                return bm25(_bm25_inputs(rows), n, k1, b)
            rows = conn.execute(_BM25_SCORE.format(filter=filter_sql), params).fetchall()
        return [(chunk, float(score)) for chunk, score in rows]

    def chunks(self, kb: KbRef, chunk_ids: list[str]) -> dict[str, StoredChunk]:
        if not chunk_ids:
            return {}
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT c.id, c.doc_id, d.title, c.text, c.start_offset, c.end_offset,"
                " c.heading_path, c.page,"
                " p.n, p.text, p.start_offset, p.end_offset, p.heading_path, p.page"
                " FROM chunks c"
                " JOIN documents d ON d.kb = c.kb AND d.doc_id = c.doc_id"
                " LEFT JOIN parents p ON p.kb = c.kb AND p.doc_id = c.doc_id AND p.n = c.parent_n"
                " WHERE c.kb = %s AND c.id = ANY(%s)",
                (kb.key, chunk_ids),
            ).fetchall()
        out = {}
        for chunk, doc, title, text, start, end, path, page, pn, pt, ps, pe, pp, ppage in rows:
            parent = StoredParent(pn, pt, ps, pe, tuple(pp), ppage) if pn is not None else None
            out[chunk] = StoredChunk(chunk, doc, title, text, start, end, tuple(path), page, parent)
        return out

    # --- documents ---------------------------------------------------------------------------

    def document_hash(self, kb: KbRef, doc_id: str) -> str | None:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT content_hash FROM documents WHERE kb = %s AND doc_id = %s",
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
            self._copy_parents(conn, kb, parents or [])  # first: chunks reference them
            self._copy_chunks(conn, kb, dim, chunks)
            self._adjust_stats(
                conn,
                kb,
                len(chunks) - removed,
                sum(len(c.terms) for c in chunks) - removed_length,
            )

    def delete_document(self, kb: KbRef, doc_id: str) -> bool:
        """False when the document — or the whole kb — does not exist."""
        with self._tx() as conn:
            if self._lock_kb(conn, kb, missing_ok=True) is None:
                return False
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

    @overload
    def _lock_kb(self, conn: psycopg.Connection, kb: KbRef) -> int: ...

    @overload
    def _lock_kb(self, conn: psycopg.Connection, kb: KbRef, *, missing_ok: bool) -> int | None: ...

    def _lock_kb(
        self, conn: psycopg.Connection, kb: KbRef, *, missing_ok: bool = False
    ) -> int | None:
        """Lock the kb's row (serialising writers and its BM25 statistics); returns its dim."""
        row = conn.execute("SELECT dim FROM kbs WHERE name = %s FOR UPDATE", (kb.key,)).fetchone()
        if row is None:
            if missing_ok:
                return None
            raise KbNotFound(f"knowledge base {kb.key!r} does not exist")
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

    def _copy_parents(self, conn: psycopg.Connection, kb: KbRef, parents: list[Chunk]) -> None:
        if not parents:
            return
        with (
            conn.cursor() as cur,
            cur.copy(
                "COPY parents (kb, doc_id, n, text, start_offset, end_offset, heading_path, page)"
                " FROM STDIN"
            ) as copy,
        ):
            for p in parents:
                copy.write_row(
                    (
                        kb.key,
                        p.doc_id,
                        p.n,
                        p.text,
                        p.start,
                        p.end,
                        Jsonb(list(p.heading_path)),
                        p.page,
                    )
                )

    def _copy_chunks(
        self, conn: psycopg.Connection, kb: KbRef, dim: int, chunks: list[IndexedChunk]
    ) -> None:
        if not chunks:
            return
        with conn.cursor() as cur:
            with cur.copy(
                "COPY chunks (kb, id, doc_id, n, text, start_offset, end_offset, tokens, length,"
                " heading_path, page, parent_n, context) FROM STDIN"
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
                            ch.page,
                            ch.parent,
                            c.context,
                        )
                    )
            with cur.copy(
                sql.SQL("COPY {} (kb, chunk_id, embedding) FROM STDIN").format(
                    _table("embeddings", dim)
                )
            ) as copy:
                for c in chunks:
                    copy.write_row((kb.key, c.chunk.chunk_id, _vector_text(c.vector)))
            with cur.copy(
                "COPY postings (kb, term, chunk_id, tf, length, doc_id) FROM STDIN"
            ) as copy:
                for c in chunks:
                    length = len(c.terms)
                    for term, tf in Counter(c.terms).items():
                        copy.write_row((kb.key, term, c.chunk.chunk_id, tf, length, c.chunk.doc_id))

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
