-- Design §3.2 / §4 (2026-11-08): every kb belongs to a namespace, and is bound to the document
-- prefix its vectors were embedded with (vectors embedded with another prefix are not comparable).
--
-- `kbs.name` stays the physical key the child tables reference; it is `<namespace>:<kb>`, which the
-- CHECK below guarantees, so the parts can never drift from the key.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM kbs) THEN
        RAISE EXCEPTION 'lite-rag 0002: kbs from before namespaces exist; drop and re-ingest them';
    END IF;
END $$;

ALTER TABLE kbs
    ADD COLUMN namespace text NOT NULL,
    ADD COLUMN kb text NOT NULL,
    ADD COLUMN document_prefix text NOT NULL DEFAULT '',
    ADD CONSTRAINT ck_kbs_key CHECK (name = namespace || ':' || kb),
    ADD CONSTRAINT uq_kbs_namespace_kb UNIQUE (namespace, kb);
