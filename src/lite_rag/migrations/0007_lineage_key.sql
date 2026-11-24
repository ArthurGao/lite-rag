-- Lineage design §6.3 (S25b): each chunk's ledger key ("{doc_id}/{text hash}/{k}"), written at
-- upsert and read with the chunk, so a query trace names the version it actually retrieved.
-- NULL for chunks written before this migration until `literag lineage backfill`.
ALTER TABLE chunks ADD COLUMN lineage_key text;
