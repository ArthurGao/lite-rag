-- Design §3.1 (2026-11-08, S5b): a chunk records the page it starts on (PDF), the parent chunk it
-- belongs to (strategy parent_child) and the context sentence an LLM wrote for it (contextual
-- enhancer). Parents are stored, never embedded or indexed: a search ranks children and returns
-- their parents. All three are NULL for kbs that do not use them.
CREATE TABLE parents (
    kb           text NOT NULL,
    doc_id       text NOT NULL,
    n            integer NOT NULL,
    text         text NOT NULL,
    start_offset integer NOT NULL,
    end_offset   integer NOT NULL,
    heading_path jsonb NOT NULL DEFAULT '[]'::jsonb,
    page         integer,
    PRIMARY KEY (kb, doc_id, n),
    FOREIGN KEY (kb, doc_id) REFERENCES documents (kb, doc_id) ON DELETE CASCADE
);

ALTER TABLE chunks
    ADD COLUMN page     integer,
    ADD COLUMN parent_n integer,
    ADD COLUMN context  text,
    ADD FOREIGN KEY (kb, doc_id, parent_n) REFERENCES parents (kb, doc_id, n) ON DELETE CASCADE;
