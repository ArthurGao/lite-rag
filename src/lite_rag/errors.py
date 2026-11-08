"""Errors the engine raises. A failure is never an empty result (design §3.2)."""


class LiteRagError(Exception):
    """Base class for every lite-rag error."""


class ConfigError(LiteRagError):
    """The configuration is invalid or does not match what a knowledge base was built with."""


class KbNotFound(LiteRagError):
    """The knowledge base does not exist (in this namespace). Reads never create one."""


class StoreError(LiteRagError):
    """The store could not be reached or rejected the operation."""
