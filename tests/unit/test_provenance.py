import hashlib

from lite_rag.provenance import chunk_keys, text_hash


def test_text_hash_is_sha256_hex() -> None:
    assert text_hash("签证费 100 元") == hashlib.sha256("签证费 100 元".encode()).hexdigest()


def test_chunk_key_names_the_document_the_text_and_its_occurrence() -> None:
    h = text_hash("a")[:16]
    assert chunk_keys("doc/1", ["a"]) == [f"doc/1/{h}/0"]


def test_chunk_keys_count_repeated_texts() -> None:
    keys = chunk_keys("d", ["a", "b", "a"])
    assert keys[0].endswith("/0") and keys[2].endswith("/1")
    assert keys[0][:-2] == keys[2][:-2]
    assert keys[1].endswith("/0") and keys[1] != keys[0]


def test_chunk_keys_do_not_depend_on_position() -> None:
    assert chunk_keys("d", ["x", "a"])[1] == chunk_keys("d", ["a"])[0]
