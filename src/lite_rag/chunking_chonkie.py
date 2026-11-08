"""`semantic` chunking through chonkie (extra `chonkie`), to compare with lite-rag's own (S5b).

chonkie's SemanticChunker uses its own method (a fixed cosine-similarity threshold, a window of
sentences, Savitzky-Golay peak detection), with lite-rag's embedder and token counter. Around
it, lite-rag keeps its contract: chunks never cross a Markdown heading, offsets are exact, and
`size` is a hard limit (an oversize chunk is cut again with the recursive strategy).
"""

from typing import Any

from lite_rag.chunking import Embed, MarkdownChunker, Section
from lite_rag.config import ChunkingConfig
from lite_rag.errors import ConfigError
from lite_rag.extras import require_extra
from lite_rag.text import TokenCounter
from lite_rag.types import Chunk, Document

# chonkie's defaults plus Chinese end marks (its defaults are English only)
DELIMITERS = [". ", "! ", "? ", "\n", "。", "！", "？", "；"]


def _bridge(embed: Embed, counter: TokenCounter) -> Any:
    """lite-rag's embedder as a chonkie embeddings object."""
    np = require_extra("numpy", "chonkie")
    base = require_extra("chonkie.embeddings", "chonkie").BaseEmbeddings

    class Bridge(base):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            self._dim: int | None = None

        @property
        def dimension(self) -> int:
            if self._dim is None:
                self._dim = len(embed(["dimension probe"])[0])
            return self._dim

        def embed(self, text: str) -> Any:
            return np.asarray(embed([text])[0], dtype=np.float32)

        def embed_batch(self, texts: list[str]) -> list[Any]:
            return [np.asarray(v, dtype=np.float32) for v in embed(texts)] if texts else []

        def get_tokenizer(self) -> Any:
            return counter.count  # chonkie accepts a callable token counter

        def __repr__(self) -> str:
            return "LiteRagEmbeddings()"

    return Bridge()


class ChonkieSemanticChunker(MarkdownChunker):
    def __init__(self, config: ChunkingConfig, counter: TokenCounter, embed: Embed) -> None:
        super().__init__(config, counter)
        chonkie = require_extra("chonkie", "chonkie")
        self._chonkie = chonkie.SemanticChunker(
            embedding_model=_bridge(embed, counter),
            threshold=config.chonkie_threshold,
            chunk_size=config.size,
            delim=DELIMITERS,
        )

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text
        regions: list[Section] = []
        for (start, end), path in self._sections(text):
            for c in self._chonkie.chunk(text[start:end]):
                a, b = start + c.start_index, start + c.end_index
                if text[a:b] != c.text:
                    raise ConfigError(f"chonkie returned offsets that do not match ({a}, {b})")
                pieces = self._fit(text, (a, b), 0)  # the hard limit: cut again if needed
                regions.extend((r, path) for r in self._pack(pieces))
        return self._to_chunks(doc, regions)
