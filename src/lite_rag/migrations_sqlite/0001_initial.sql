-- lite-rag on SQLite, for the Chroma backend (design §4): the PostgreSQL tables except the
-- vectors, which live in Chroma (one collection per kb). JSON goes in TEXT columns. BM25 is
-- scored in Python (keyword.py), and N / avgdl are counted from `chunks`, not kept on `kbs`.
--
-- Vectors and rows share no transaction, so a document carries a status: written `pending`,
-- `ready` once its vectors are in Chroma, `deleting` while they are removed. Reads see `ready`
-- documents only; opening the store removes the others (design §4, write order and recovery).

CREATE TABLE kbs (
    name               text PRIMARY KEY,  -- "<namespace>:<kb>"
    namespace          text NOT NULL,
    kb                 text NOT NULL,
    embedding_provider text NOT NULL,
    embedding_model    text NOT NULL,
    dim                integer NOT NULL CHECK (dim > 0),
    query_prefix       text NOT NULL,
    document_prefix    text NOT NULL,
    chunker            text NOT NULL,
    created_at         text NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (name = namespace || ':' || kb),
    UNIQUE (namespace, kb)
);

CREATE TABLE documents (
    kb           text NOT NULL REFERENCES kbs (name) ON DELETE CASCADE,
    doc_id       text NOT NULL,
    title        text NOT NULL,
    source       text,
    content_hash text NOT NULL,
    chunk_count  integer NOT NULL,
    metadata     text NOT NULL DEFAULT '{}',
    text         text NOT NULL,
    pages        text,
    status       text NOT NULL CHECK (status IN ('pending', 'ready', 'deleting')),
    updated_at   text NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (kb, doc_id)
);
CREATE INDEX ix_documents_status ON documents (status) WHERE status <> 'ready';

CREATE TABLE parents (
    kb           text NOT NULL,
    doc_id       text NOT NULL,
    n            integer NOT NULL,
    text         text NOT NULL,
    start_offset integer NOT NULL,
    end_offset   integer NOT NULL,
    heading_path text NOT NULL DEFAULT '[]',
    page         integer,
    PRIMARY KEY (kb, doc_id, n),
    FOREIGN KEY (kb, doc_id) REFERENCES documents (kb, doc_id) ON DELETE CASCADE
);

CREATE TABLE chunks (
    kb           text NOT NULL,
    id           text NOT NULL,  -- "{doc_id}#{n}", also the Chroma record id
    doc_id       text NOT NULL,
    n            integer NOT NULL,
    text         text NOT NULL,
    start_offset integer NOT NULL,
    end_offset   integer NOT NULL,
    tokens       integer NOT NULL,
    length       integer NOT NULL,  -- BM25 terms
    heading_path text NOT NULL DEFAULT '[]',
    page         integer,
    parent_n     integer,
    context      text,
    PRIMARY KEY (kb, id),
    FOREIGN KEY (kb, doc_id) REFERENCES documents (kb, doc_id) ON DELETE CASCADE,
    FOREIGN KEY (kb, doc_id, parent_n) REFERENCES parents (kb, doc_id, n) ON DELETE CASCADE
);
CREATE INDEX ix_chunks_doc ON chunks (kb, doc_id);

CREATE TABLE postings (
    kb       text NOT NULL,
    term     text NOT NULL,
    chunk_id text NOT NULL,
    doc_id   text NOT NULL,
    tf       integer NOT NULL CHECK (tf > 0),
    length   integer NOT NULL,
    PRIMARY KEY (kb, term, chunk_id),
    FOREIGN KEY (kb, chunk_id) REFERENCES chunks (kb, id) ON DELETE CASCADE
);
CREATE INDEX ix_postings_chunk ON postings (kb, chunk_id);

CREATE TABLE embedding_cache (
    model_id   text NOT NULL,
    dim        integer NOT NULL,
    key        text NOT NULL,
    embedding  blob NOT NULL,  -- float32, as pgvector stores it
    created_at text NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, dim, key)
);
