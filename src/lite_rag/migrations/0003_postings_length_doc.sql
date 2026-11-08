-- Design §4/§5 (2026-11-08, S3): postings carry the chunk's BM25 length and its document id, so
-- scoring reads postings only (no probe into the wide chunks table per posting) and a metadata
-- filter goes straight to documents. Chunks are replaced, never updated, so neither can go stale.
ALTER TABLE postings ADD COLUMN length integer, ADD COLUMN doc_id text;

UPDATE postings p SET length = c.length, doc_id = c.doc_id
FROM chunks c WHERE c.kb = p.kb AND c.id = p.chunk_id;

ALTER TABLE postings ALTER COLUMN length SET NOT NULL, ALTER COLUMN doc_id SET NOT NULL;
