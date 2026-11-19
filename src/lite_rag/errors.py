"""Errors the engine raises. A failure is never an empty result (design §3.2).

Each class says what it means to an HTTP client (`status`) and whether its message may be shown
(`expose`): upstream failures can carry endpoints or provider messages, so they are logged, not
returned (design §7)."""

import re
from typing import ClassVar

# (pattern, replacement), the same rules as lite-mem's `redact`: a URL's user:password@; a
# password-like URL parameter (it ends at the next parameter); a key=value password with any
# prefix (PGPASSWORD, sslpassword), spaces around '=', quoted or up to the next space.
_CREDENTIALS = (
    (re.compile(r"(://)[^\s/@]+@"), r"\1***@"),
    (re.compile(r"([?&]\w*password=)[^\s&]+", re.IGNORECASE), r"\1***"),
    (re.compile(r"((?<![?&\w])\w*password\s*=\s*)('[^']*'|\S+)", re.IGNORECASE), r"\1***"),
    # An HTTP credential, and an API key or token written as key=value / key: value.
    (re.compile(r"(\bbearer\s+)\S+", re.IGNORECASE), r"\1***"),
    (
        re.compile(
            r"(\b(?:api[_-]?key|(?:access|auth)?[_-]?token|secret)\s*[:=]\s*)\S+", re.IGNORECASE
        ),
        r"\1***",
    ),
)


def redact(text: str) -> str:
    """`text` with credentials removed — a URL's user:password, a conninfo's password — for
    logging an error whose message is not shown (the server, and in-process callers). A whole
    connection string is `db.redact_conninfo`'s job; this scrubs free text quoting one."""
    for pattern, replacement in _CREDENTIALS:
        text = pattern.sub(replacement, text)
    return text


class LiteRagError(Exception):
    """Base class for every lite-rag error."""

    status: ClassVar[int] = 500
    expose: ClassVar[bool] = False


class InvalidRequest(LiteRagError):
    """The caller asked for something invalid: a bad name, argument, file type or filter, or a
    kb that conflicts with the request. The caller can fix it, so the message is shown."""

    status = 400
    expose = True


class ConfigError(LiteRagError):
    """The service's own configuration is wrong or incomplete — an extra not installed, an API
    key not set, an unknown model. The operator's problem, not the caller's: over HTTP a 500
    whose message (naming env vars and extras) is logged, not returned. The CLI prints it."""


class KbMismatch(InvalidRequest):
    """A knowledge base exists but was built with other settings (model, prefixes, chunker)."""

    status = 409


class KbNotFound(LiteRagError):
    """The knowledge base does not exist. Reads never create one."""

    status = 404
    expose = True

    @classmethod
    def of(cls, kb: str, namespace: str) -> "KbNotFound":
        return cls(f"knowledge base {kb!r} does not exist in {namespace!r}")


class DocumentNotFound(LiteRagError):
    """The document does not exist in the knowledge base."""

    status = 404
    expose = True


class TooLarge(LiteRagError):
    """A request body over the server's limit (an upload)."""

    status = 413
    expose = True


class StoreError(LiteRagError):
    """The store could not be reached or rejected the operation."""

    status = 503
