-- A namespace's traces in time order: `trace purge` and listing (ids() reads it covered).
CREATE INDEX ix_rag_query_traces_namespace ON rag_query_traces (namespace, created_at, id);
