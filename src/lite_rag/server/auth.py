"""API keys and access (design §7; KB web-authz: default deny).

A key is `lrk_<prefix>_<secret>`; the server knows only its sha256. Every request except
`/health` (the one deliberately unguarded path) names a namespace in `X-LiteRag-Namespace` and
needs a key whose globs cover that namespace and the kb, with `write` mode for any change. No
route creates a key: `literag key create` prints one, once.
"""

import hashlib
import secrets
from fnmatch import fnmatchcase

from lite_rag.config import ApiKeyConfig
from lite_rag.errors import LiteRagError

PREFIX = "lrk"


def new_key() -> tuple[str, str]:
    """A fresh key and its sha256 (what the config holds)."""
    key = f"{PREFIX}_{secrets.token_hex(4)}_{secrets.token_urlsafe(32)}"
    return key, key_hash(key)


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Denied(LiteRagError):
    """401 (no valid key) or 403 (a valid key without this access)."""

    expose = True

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status  # type: ignore[misc]  # per instance: 401 or 403


class Authorizer:
    def __init__(self, keys: list[ApiKeyConfig]) -> None:
        self._by_hash = {k.sha256: k for k in keys}

    @property
    def open(self) -> bool:
        """No keys configured: every request is allowed (the server then binds loopback only)."""
        return not self._by_hash

    def check(
        self, authorization: str | None, namespace: str, kb: str | None, write: bool
    ) -> ApiKeyConfig | None:
        """The key, when it allows this (None: open mode); raises `Denied` otherwise. `kb` None:
        a namespace-level request, whose results the caller narrows with `visible`."""
        if self.open:
            return None
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise Denied(401, "an API key is required: Authorization: Bearer <key>")
        key = self._by_hash.get(key_hash(token.strip()))
        if key is None:
            raise Denied(401, "unknown API key")
        if not any(fnmatchcase(namespace, p) for p in key.namespaces):
            raise Denied(403, f"key {key.name!r} has no access to namespace {namespace!r}")
        if kb is not None:
            self.require_kbs(key, [kb])
        if write and key.mode != "write":
            raise Denied(403, f"key {key.name!r} is read-only")
        return key

    @staticmethod
    def require_kbs(key: ApiKeyConfig | None, kbs: list[str]) -> None:
        """Raise `Denied` (403) unless `key` covers every kb named (None: open mode). A request
        that names its kbs in the body (several at once) is checked here, like a path's kb."""
        if key is None:
            return
        for kb in kbs:
            if not any(fnmatchcase(kb, p) for p in key.kbs):
                raise Denied(403, f"key {key.name!r} has no access to knowledge base {kb!r}")

    @staticmethod
    def visible(key: ApiKeyConfig | None, kbs: list[str]) -> list[str]:
        """The kbs `key` may see: a listing never names kbs outside its globs."""
        if key is None:
            return kbs
        return [kb for kb in kbs if any(fnmatchcase(kb, p) for p in key.kbs)]
