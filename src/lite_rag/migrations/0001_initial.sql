-- lite-rag schema (design §4). Runs with search_path = <schema>, public, so names are unqualified and
-- the `vector` type resolves from the extension. Vector tables (embeddings_<dim>) are created on first
-- use, one per dimension, because an HNSW index needs a fixed dimension.

CREATE TABLE kbs (
    name               text PRIMARY KEY,
    embedding_provider text NOT NULL,
    embedding_model    text NOT NULL,
    dim                integer NOT NULL CHECK (dim > 0),
    query_prefix       text NOT NULL DEFAULT '',
    chunker            jsonb NOT NULL,
    n_chunks           integer NOT NULL DEFAULT 0,  -- BM25: N
    total_length       bigint NOT NULL DEFAULT 0,   -- BM25: sum of chunk lengths, for avgdl
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE documents (
    kb           text NOT NULL REFERENCES kbs (name) ON DELETE CASCADE,
    doc_id       text NOT NULL,
    title        text NOT NULL,
    source       text,
    content_hash text NOT NULL,
    chunk_count  integer NOT NULL,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (kb, doc_id)
);

CREATE TABLE chunks (
    kb           text NOT NULL,
    id           text NOT NULL,             -- "{doc_id}#{n}"
    doc_id       text NOT NULL,
    n            integer NOT NULL,
    text         text NOT NULL,
    start_offset integer NOT NULL,
    end_offset   integer NOT NULL,
    tokens       integer NOT NULL,          -- embedding-model tokens (or estimated)
    length       integer NOT NULL,          -- BM25 terms
    heading_path jsonb NOT NULL DEFAULT '[]'::jsonb,
    PRIMARY KEY (kb, id),
    FOREIGN KEY (kb, doc_id) REFERENCES documents (kb, doc_id) ON DELETE CASCADE
);

CREATE TABLE postings (
    kb       text NOT NULL,
    term     text NOT NULL,
    chunk_id text NOT NULL,
    tf       integer NOT NULL CHECK (tf > 0),
    PRIMARY KEY (kb, term, chunk_id),
    FOREIGN KEY (kb, chunk_id) REFERENCES chunks (kb, id) ON DELETE CASCADE
);
CREATE INDEX ix_postings_chunk ON postings (kb, chunk_id);
