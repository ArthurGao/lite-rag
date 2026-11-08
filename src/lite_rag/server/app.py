"""Routes, request models and RFC 9457 Problem Details (design §7)."""

import dataclasses
import logging
import re
import tempfile
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from anyio import to_thread
from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lite_rag.config import Mode, Settings
from lite_rag.engine import RagEngine, RagRuntime
from lite_rag.errors import (
    ConfigError,
    DocumentNotFound,
    InvalidRequest,
    KbNotFound,
    LiteRagError,
    TooLarge,
)
from lite_rag.loaders import SUFFIXES
from lite_rag.server.auth import Authorizer

log = logging.getLogger("lite_rag.server")
_CREDENTIALS = re.compile(r"(?<=://)[^\s/@]+@|(?<=password=)\S+", re.IGNORECASE)
MULTIPART_OVERHEAD = 8 * 1024  # boundaries and part headers on top of the file itself


def redact(text: str) -> str:
    """`text` with credentials removed: a URL's user:password, a conninfo's password."""
    return _CREDENTIALS.sub(lambda m: "***@" if m.group().endswith("@") else "***", text)


class _BodyLimit:
    """Refuses a request body over its limit before anything parses it — FastAPI reads a whole
    body (a multipart one onto disk) before the route, and before authentication, runs: by
    Content-Length up front, and by counting while it streams when the client sends none.
    The upload route (`/v1/kbs/{kb}/files`, matched exactly) has its own, larger limit."""

    def __init__(self, app: ASGIApp, upload_limit: int, body_limit: int) -> None:
        self._app = app
        self._upload = upload_limit + MULTIPART_OVERHEAD
        self._body = body_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        limit = self._upload if UPLOAD_PATH.fullmatch(scope["path"]) else self._body
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > limit:
            await problem(413, "TooLarge", TOO_LARGE)(scope, receive, send)
            return
        seen = 0

        async def counted() -> Message:
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > limit:
                raise TooLarge(TOO_LARGE)
            return message

        await self._app(scope, counted, send)


UPLOAD_PATH = re.compile(r"/v1/kbs/[^/]+/files")
TOO_LARGE = "the request body is larger than the server allows (server.max_upload_mb / max_body_mb)"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}  # bind addresses allowed without keys
LOCAL_HOSTS = sorted(f"[{h}]" if ":" in h else h for h in LOOPBACK)  # as Host headers spell them
NAMESPACE_HEADER = "X-LiteRag-Namespace"
PROBLEM = "application/problem+json"
TITLES = {401: "Unauthorized", 403: "Forbidden"}


class DocumentIn(BaseModel):
    title: str
    text: str
    metadata: dict[str, Any] = {}


class _Retrieval(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=200)
    mode: Mode | None = None
    rerank: bool | None = None
    where: dict[str, Any] | None = None


class SearchIn(_Retrieval):
    query: str = Field(min_length=1)


class AskIn(_Retrieval):
    question: str = Field(min_length=1)


def problem(status: int, title: str, detail: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {"type": "about:blank", "title": title, "status": status}
    if detail:
        body["detail"] = detail
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return JSONResponse(body, status_code=status, media_type=PROBLEM, headers=headers)


def create_app(settings: Settings, runtime: RagRuntime) -> FastAPI:
    """The app around `runtime`: each request gets an engine for the namespace it names."""
    auth = Authorizer(settings.server.keys)
    upload_limit = int(settings.server.max_upload_mb * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        to_thread.current_default_thread_limiter().total_tokens = settings.server.max_workers
        await to_thread.run_sync(runtime.warm_up)
        yield

    app = FastAPI(title="lite-rag", lifespan=lifespan)
    app.add_middleware(
        _BodyLimit,
        upload_limit=upload_limit,
        body_limit=int(settings.server.max_body_mb * 1024 * 1024),
    )
    if auth.open:
        # Open mode is safe only on loopback; a DNS-rebinding web page reaches loopback under
        # its own host name, so only local host names are answered.
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)

    @app.exception_handler(LiteRagError)
    async def _lite_rag(request: Request, exc: LiteRagError) -> JSONResponse:
        # Each error class carries its status and whether its message may be shown (errors.py).
        if not exc.expose:  # the detail stays here, credentials removed
            log.error(
                "%s %s: %s: %s",
                request.method,
                request.url.path,
                type(exc).__name__,
                redact(str(exc)),
            )
        title = TITLES.get(exc.status, type(exc).__name__)
        return problem(exc.status, title, str(exc) if exc.expose else None)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return problem(422, "Unprocessable Content", detail)

    def access(write: bool) -> Callable[..., Coroutine[Any, Any, RagEngine]]:
        """The engine for the request's namespace, once its key allows the request (async: no
        I/O, so it runs on the event loop instead of taking a worker thread)."""

        async def dependency(
            request: Request,
            authorization: Annotated[str | None, Header()] = None,
            namespace: Annotated[str | None, Header(alias=NAMESPACE_HEADER)] = None,
        ) -> RagEngine:
            if not namespace:
                raise InvalidRequest(f"the {NAMESPACE_HEADER} header is required")
            request.state.key = auth.check(
                authorization, namespace, request.path_params.get("kb"), write
            )
            # An engine per request: as cheap as a lookup, and no cache that callers can grow.
            return runtime.engine(namespace)

        return dependency

    Read = Annotated[RagEngine, Depends(access(write=False))]
    Write = Annotated[RagEngine, Depends(access(write=True))]

    # --- the one deliberately unguarded path ---------------------------------------------------

    @app.get("/health")
    def health(deep: bool = False) -> dict[str, str]:
        runtime.health(deep)
        return {"status": "ok"}

    # --- knowledge bases -----------------------------------------------------------------------

    @app.get("/v1/kbs")
    def list_kbs(request: Request, eng: Read) -> dict[str, Any]:
        kbs = Authorizer.visible(request.state.key, eng.list_kbs())
        return {"namespace": eng.namespace, "kbs": kbs}

    @app.put("/v1/kbs/{kb}")
    def create_kb(kb: str, eng: Write) -> dict[str, Any]:
        info = eng.create_kb(kb)
        return {k: v for k, v in dataclasses.asdict(info).items() if k != "ref"} | {"kb": kb}

    @app.get("/v1/kbs/{kb}")
    def kb_stats(kb: str, eng: Read) -> dict[str, Any]:
        stats = eng.stats(kb)
        if stats is None:
            raise KbNotFound.of(kb, eng.namespace)
        return dataclasses.asdict(stats)

    @app.delete("/v1/kbs/{kb}", status_code=204)
    def delete_kb(kb: str, eng: Write) -> None:
        if not eng.delete_kb(kb):
            raise KbNotFound.of(kb, eng.namespace)

    # --- documents -----------------------------------------------------------------------------

    @app.put("/v1/kbs/{kb}/documents/{doc_id}")
    def upsert(kb: str, doc_id: str, doc: DocumentIn, eng: Write) -> dict[str, Any]:
        return dataclasses.asdict(eng.upsert_text(kb, doc_id, doc.title, doc.text, doc.metadata))

    @app.delete("/v1/kbs/{kb}/documents/{doc_id}", status_code=204)
    def delete_document(kb: str, doc_id: str, eng: Write) -> None:
        if not eng.delete(kb, doc_id):
            raise DocumentNotFound(f"document {doc_id!r} does not exist in {kb!r}")

    @app.post("/v1/kbs/{kb}/files")
    def upload(
        kb: str,
        eng: Write,
        file: Annotated[UploadFile, File()],
        doc_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        name = file.filename or "upload"
        suffix = Path(name).suffix.lower()
        if suffix not in SUFFIXES:  # a fixed message: the client's file name is not echoed
            raise InvalidRequest(
                f"unsupported file type (supported: {', '.join(sorted(SUFFIXES))})"
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            written = 0
            with path.open("wb") as out:  # streamed and counted: never the whole file in memory
                while chunk := file.file.read(1 << 20):
                    written += len(chunk)
                    if written > upload_limit:
                        raise TooLarge(TOO_LARGE)
                    out.write(chunk)
            return dataclasses.asdict(eng.ingest(kb, path, doc_id=doc_id, source=name))

    # --- retrieval -----------------------------------------------------------------------------

    @app.post("/v1/kbs/{kb}/search")
    def search(kb: str, body: SearchIn, eng: Read) -> dict[str, Any]:
        hits = eng.search(kb, body.query, **body.model_dump(exclude={"query"}))
        return {"hits": [dataclasses.asdict(h) for h in hits]}

    @app.post("/v1/kbs/{kb}/ask")
    def ask(kb: str, body: AskIn, eng: Read) -> dict[str, Any]:
        return dataclasses.asdict(
            eng.ask(kb, body.question, **body.model_dump(exclude={"question"}))
        )

    return app


def check_bind(settings: Settings, host: str) -> None:
    """Without API keys the server is open, so it may only listen on a loopback address."""
    if not settings.server.keys and host not in LOOPBACK:
        raise ConfigError(
            f"no API keys configured: refusing to bind {host} (create one: literag key create)"
        )


def run(settings: Settings, runtime: RagRuntime, host: str, port: int) -> None:
    """Serve the app (uvicorn, extra `server`), after `check_bind` — wherever it is started."""
    check_bind(settings, host)
    from lite_rag.extras import require_extra

    require_extra("uvicorn", "server").run(create_app(settings, runtime), host=host, port=port)
