-- Design §4/§5 (2026-11-12, S11): graph retrieval, local mode. An LLM lists each chunk's entities
-- and relations at ingest; both are rows of the chunk, so they go and come with it (one document,
-- one transaction). `entity` is the normalised name (graph.entity_key) that merges mentions across
-- chunks and documents. The graph path walks co-mentions: the primary key finds a chunk's entities,
-- ix_graph_mentions_entity an entity's chunks. Relations are kept for later use (global mode, answer context). Empty for kbs without a graph.
CREATE TABLE graph_mentions (
    kb       text NOT NULL,
    chunk_id text NOT NULL,
    entity   text NOT NULL,
    name     text NOT NULL,
    type     text NOT NULL DEFAULT '',
    PRIMARY KEY (kb, chunk_id, entity),
    FOREIGN KEY (kb, chunk_id) REFERENCES chunks (kb, id) ON DELETE CASCADE
);
CREATE INDEX ix_graph_mentions_entity ON graph_mentions (kb, entity);

CREATE TABLE graph_relations (
    kb          text NOT NULL,
    chunk_id    text NOT NULL,
    source      text NOT NULL,
    target      text NOT NULL,
    description text NOT NULL DEFAULT '',
    PRIMARY KEY (kb, chunk_id, source, target),
    FOREIGN KEY (kb, chunk_id) REFERENCES chunks (kb, id) ON DELETE CASCADE
);
