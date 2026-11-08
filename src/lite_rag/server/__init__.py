"""The HTTP service (design §7): `create_app` serves one `RagRuntime` (an engine per
request, for the namespace it names) over FastAPI. Extra `[server]`."""

from lite_rag.server.app import check_bind, create_app, run

__all__ = ["check_bind", "create_app", "run"]
