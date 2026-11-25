-- gc finds the ended artifacts that still hold their text: only those rows are indexed, so the
-- index shrinks as gc clears them and a run costs what is due, not the whole table.
CREATE INDEX lineage_artifacts_gc ON lineage_artifacts (ended_at) WHERE payload IS NOT NULL;
