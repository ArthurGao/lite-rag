"""Errors the engine raises. A failure is never an empty result (design §3.2).

Each class says what it means to an HTTP client (`status`) and whether its message may be shown
(`expose`): upstream failures can carry endpoints or provider messages, so they are logged, not
returned (design §7)."""

from typing import ClassVar


class LiteRagError(Exception):
    """Base class for every lite-rag error."""

    status: ClassVar[int] = 500
    expose: ClassVar[bool] = False


class ConfigError(LiteRagError):
    """The configuration is invalid or does not match what a knowledge base was built with."""

    status = 400
    expose = True


class KbMismatch(ConfigError):
    """A knowledge base exists but was built with other settings (model, prefixes, chunker)."""

    status = 409


class KbNotFound(LiteRagError):
    """The knowledge base (or a document in it) does not exist. Reads never create one."""

    status = 404
    expose = True

    @classmethod
    def of(cls, kb: str, namespace: str) -> "KbNotFound":
        return cls(f"knowledge base {kb!r} does not exist in {namespace!r}")


class TooLarge(LiteRagError):
    """A request body over the server's limit (an upload)."""

    status = 413
    expose = True


class StoreError(LiteRagError):
    """The store could not be reached or rejected the operation."""

    status = 503
