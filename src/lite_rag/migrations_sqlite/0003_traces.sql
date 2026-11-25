-- Lineage design §6.3 (S25c): query traces, as in PostgreSQL migration 0008 (JSON and times
-- as text; times fixed-width ISO-8601 UTC, so text order is time order).
CREATE TABLE rag_query_traces (
    id          text PRIMARY KEY,
    namespace   text NOT NULL,
    kbs         text NOT NULL,
    operation   text NOT NULL CHECK (operation IN ('search', 'ask')),
    query       text,
    retrieval   text NOT NULL,
    model_id    text,
    usage       text,
    cached      integer,
    status      text,
    created_at  text NOT NULL
);
CREATE INDEX ix_rag_query_traces_created ON rag_query_traces (created_at);

CREATE TABLE rag_query_sources (
    trace_id     text NOT NULL REFERENCES rag_query_traces (id) ON DELETE CASCADE,
    artifact_ns  text NOT NULL,
    artifact_key text,                -- NULL: the chunk had no ledger key (unrecorded)
    chunk_id     text NOT NULL,
    stage        text NOT NULL CHECK (stage IN ('retrieved', 'reranked', 'in_context', 'cited')),
    rank         integer,
    score        real,
    PRIMARY KEY (trace_id, artifact_ns, chunk_id, stage)
);
