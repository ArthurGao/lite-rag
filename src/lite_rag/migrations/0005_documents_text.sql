-- Design §4 (2026-11-08, S6): a document keeps its original text and page starts, so a kb can be
-- rebuilt under new settings (`literag reindex`) without its source files. NULL for documents
-- ingested before this migration: reindex refuses those until they are ingested again.
ALTER TABLE documents ADD COLUMN text text, ADD COLUMN pages jsonb;
