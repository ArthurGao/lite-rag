"""The HTTP service (design §7): `create_app` serves one `RagEngine` (and its per-namespace
copies) over FastAPI. Extra `[server]`."""

from lite_rag.server.app import check_bind, create_app, run

__all__ = ["check_bind", "create_app", "run"]
