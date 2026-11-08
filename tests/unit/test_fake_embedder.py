import math

from lite_rag.testing import FakeEmbedder


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_deterministic_normalised_and_sized() -> None:
    e = FakeEmbedder(dim=32)
    [v1, v2] = e.embed_documents(["退款 需要 7 天", "退款 需要 7 天"])
    assert v1 == v2
    assert len(v1) == 32
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0)
    assert e.calls == [["退款 需要 7 天", "退款 需要 7 天"]]


def test_shared_words_are_closer_than_unrelated_text() -> None:
    e = FakeEmbedder(dim=256)
    q = e.embed_query("refund policy")
    related, unrelated = e.embed_documents(["our refund policy is 7 days", "the weather is sunny"])
    assert _cos(q, related) > _cos(q, unrelated)


def test_chinese_is_compared_per_character() -> None:
    e = FakeEmbedder(dim=256)
    q = e.embed_query("北京")
    near, far = e.embed_documents(["我住在北京", "天气很好"])
    assert _cos(q, near) > _cos(q, far)
