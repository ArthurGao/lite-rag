-- Lineage design §6.3 (S25b): each chunk's ledger key, as in PostgreSQL migration 0007.
ALTER TABLE chunks ADD COLUMN lineage_key text;
