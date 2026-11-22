"""Named embedders (S22): `embeddings.<name>` besides `embedding` (the one named `default`)."""

import pytest

from lite_rag.config import load_settings


def test_named_embedders_sit_beside_the_default() -> None:
    s = load_settings(
        embeddings={"e5": {"provider": "fastembed", "model": "intfloat/multilingual-e5-large"}}
    )
    assert list(s.embeddings) == ["e5"] and s.embeddings["e5"].provider == "fastembed"
    assert s.embedding.model == "bge-m3:567m"  # the default is still `embedding`
    assert load_settings().embeddings == {}


@pytest.mark.parametrize("name", ["default", "Bad-Name", "", "1x"])
def test_an_embedder_name_is_a_lowercase_identifier_and_not_default(name: str) -> None:
    with pytest.raises(ValueError, match="embeddings"):
        load_settings(embeddings={name: {"model": "m"}})
