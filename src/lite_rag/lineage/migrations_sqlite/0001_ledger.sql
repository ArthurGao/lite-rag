-- Lineage ledger (lite-rag lineage design §5.2), SQLite: JSON and ISO-8601 times as text.
CREATE TABLE lineage_artifacts (
    id text PRIMARY KEY,
    namespace text NOT NULL,
    kind text NOT NULL,
    key text NOT NULL,
    version integer NOT NULL,
    content_hash text NOT NULL,
    producer text,
    payload text,
    metadata text NOT NULL DEFAULT '{}',
    status text NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'deleted')),
    valid_from text NOT NULL,
    ended_at text,
    UNIQUE (namespace, kind, key, version)
);
CREATE UNIQUE INDEX lineage_artifacts_active ON lineage_artifacts (namespace, kind, key)
    WHERE status = 'active';

CREATE TABLE lineage_edges (
    parent_id text NOT NULL REFERENCES lineage_artifacts (id) ON DELETE CASCADE,
    child_id text NOT NULL REFERENCES lineage_artifacts (id) ON DELETE CASCADE,
    metadata text NOT NULL DEFAULT '{}',
    PRIMARY KEY (parent_id, child_id)
);
CREATE INDEX lineage_edges_child ON lineage_edges (child_id);
