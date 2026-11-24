-- Lineage design §6.3 (S25c): query traces. A trace is one search or answer; its sources are
-- the chunks each stage used (retrieved / reranked / in_context / cited), keyed by the chunk's
-- ledger key as read at retrieval — no foreign key to the ledger: a chunk written before the
-- ledger existed is traced too, shown as unrecorded.
CREATE TABLE rag_query_traces (
    id          text PRIMARY KEY,
    namespace   text NOT NULL,
    kbs         jsonb NOT NULL,
    operation   text NOT NULL CHECK (operation IN ('search', 'ask')),
    query       text,
    retrieval   jsonb NOT NULL,
    model_id    text,
    usage       jsonb,
    cached      boolean,
    status      text,
    created_at  timestamptz NOT NULL
);
CREATE INDEX ix_rag_query_traces_created ON rag_query_traces (created_at);

CREATE TABLE rag_query_sources (
    trace_id     text NOT NULL REFERENCES rag_query_traces (id) ON DELETE CASCADE,
    artifact_ns  text NOT NULL,
    artifact_key text,                -- NULL: the chunk had no ledger key (unrecorded)
    chunk_id     text NOT NULL,
    stage        text NOT NULL CHECK (stage IN ('retrieved', 'reranked', 'in_context', 'cited')),
    rank         integer,
    score        double precision,
    PRIMARY KEY (trace_id, artifact_ns, chunk_id, stage)
);
