"""Embedders (design §3.2). Documents and queries are embedded separately: many models want a query
prefix or an input type, so `embed_query` is never just `embed_documents([q])[0]` by assumption."""

from typing import Protocol


class Embedder(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
