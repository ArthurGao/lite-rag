import pytest

from lite_rag.chunking import STRATEGIES, build_chunker, embedding_text
from lite_rag.config import ChunkingConfig
from lite_rag.text import CharCounter, EstimateCounter, TokenCounter
from lite_rag.types import Chunk, Document

EN = (
    "Refunds are processed within seven days. Contact support if it takes longer.\n\n"
    "Shipping is free above fifty dollars. Orders ship on weekdays only; weekends are closed.\n\n"
    "Returns must be unused and in the original packaging."
)
ZH = (
    "退款会在七天内处理。如果超过七天，请联系客服！\n\n"
    "满五十元包邮。订单只在工作日发货；周末不发货。\n\n"
    "退货必须未使用，并保留原包装。"
)
MD = (
    "Intro line.\n\n"
    "# Policies\n\n## Refunds\n\nRefunds take seven days.\n\n"
    "## Shipping\n\nFree above fifty dollars.\n\n# FAQ\n\nAsk support."
)


def _split(text: str, counter: TokenCounter | None = None, **cfg: object) -> list[Chunk]:
    config = ChunkingConfig(
        **{"strategy": "recursive", "size": 40, "overlap": 0, "min_size": 0, **cfg}
    )
    return build_chunker(config, counter or CharCounter()).split(Document("d", "t", text))


@pytest.mark.parametrize("min_size", [0, 15])
@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
@pytest.mark.parametrize("text", [EN, ZH, MD])
def test_every_chunk_is_an_exact_slice_within_size(strategy: str, text: str, min_size: int) -> None:
    chunks = _split(text, strategy=strategy, size=40, overlap=10, min_size=min_size)
    assert chunks, "no chunks"
    for i, c in enumerate(chunks):
        assert text[c.start : c.end] == c.text  # offsets point at the chunk
        assert c.chunk_id == f"d#{i}" and c.n == i
        assert c.text == c.text.strip() and c.text
        assert c.tokens == len(c.text) <= 40
    # nothing is lost: every non-space character is in some chunk
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


def test_fixed_windows_have_exact_size_and_overlap() -> None:
    text = "abcdefghij" * 3  # 30 chars
    chunks = _split(text, strategy="fixed", size=10, overlap=4)
    assert [(c.start, c.end) for c in chunks] == [(0, 10), (6, 16), (12, 22), (18, 28), (24, 30)]


def test_recursive_prefers_paragraphs_then_sentences() -> None:
    chunks = _split(EN, size=80)
    assert (
        chunks[0].text
        == "Refunds are processed within seven days. Contact support if it takes longer."
    )
    # the second paragraph does not fit with the first: it is cut at sentence ends, never mid-word
    assert chunks[1].text == "Shipping is free above fifty dollars. Orders ship on weekdays only;"
    assert all(c.text[-1] in ".;" for c in chunks)


def test_sentence_chunks_cut_at_chinese_end_marks() -> None:
    chunks = _split(ZH, strategy="sentence", size=16)
    texts = [c.text for c in chunks]
    assert "退款会在七天内处理。" in texts
    assert all(t[-1] in "。！；" for t in texts)


def test_overlap_repeats_whole_pieces_from_the_previous_chunk() -> None:
    text = "One. Two. Three. Four. Five. Six."
    chunks = _split(text, strategy="sentence", size=14, overlap=7)
    assert [c.text for c in chunks] == [
        "One. Two.",
        "Two. Three.",
        "Three. Four.",
        "Four. Five.",
        "Five. Six.",
    ]


def test_paragraph_packs_short_paragraphs_together() -> None:
    text = "A a.\n\nB b.\n\nC c.\n\n" + "D " * 30
    chunks = _split(text, strategy="paragraph", size=20)
    assert chunks[0].text == "A a.\n\nB b.\n\nC c."
    # the long paragraph is cut into chunks of its own, each within size
    rest = chunks[1:]
    assert len(rest) >= 3
    assert all(set(c.text) <= {"D", " "} and c.tokens <= 20 for c in rest)


def test_markdown_keeps_heading_paths_and_never_crosses_a_heading() -> None:
    chunks = _split(MD, strategy="markdown", size=200)
    by_path = {c.heading_path: c.text for c in chunks}
    assert by_path[()] == "Intro line."
    assert by_path[("Policies", "Refunds")] == "## Refunds\n\nRefunds take seven days."
    assert by_path[("Policies", "Shipping")] == "## Shipping\n\nFree above fifty dollars."
    assert by_path[("FAQ",)] == "# FAQ\n\nAsk support."
    refunds = next(c for c in chunks if c.heading_path == ("Policies", "Refunds"))
    assert embedding_text(refunds).startswith("Policies > Refunds\n")


def test_a_short_tail_is_rebalanced_and_size_stays_a_hard_limit() -> None:
    text = "aaaa bbbb cc"  # pieces of 5, 5 and 2 units
    assert [c.text for c in _split(text, size=10, min_size=0)] == ["aaaa bbbb", "cc"]
    rebalanced = _split(text, size=10, min_size=5)
    assert [c.text for c in rebalanced] == ["aaaa", "bbbb cc"]
    assert all(c.tokens <= 10 for c in rebalanced)


def test_a_short_tail_stays_when_no_even_split_fits() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta. Eta."  # whole sentences of 18, 20 and 4
    chunks = _split(text, strategy="sentence", size=20, min_size=6)
    assert (
        chunks[-1].text == "Eta."
    )  # merging would exceed size; splitting a sentence is not allowed
    assert all(c.tokens <= 20 for c in chunks)


def test_a_word_longer_than_size_is_cut_at_token_boundaries() -> None:
    chunks = _split("x" * 25, size=10)
    assert [c.text for c in chunks] == ["x" * 10, "x" * 10, "x" * 5]


def test_estimated_tokens_count_cjk_per_character() -> None:
    counter = EstimateCounter()
    assert counter.count("我住北京") == 4
    assert counter.count("I live in Beijing!") == 6  # "Beijing" is 7 letters: 2 estimated tokens
    assert counter.identity == "estimate"


def test_a_short_tail_never_pushes_a_chunk_over_size() -> None:
    # previous chunk 9 units, tail 2 units: merging would make 11 > size 10
    text = "aaaa bbbb cc"
    chunks = _split(text, strategy="recursive", size=10, min_size=5)
    assert all(c.tokens <= 10 for c in chunks)
    assert [c.text for c in chunks] == ["aaaa", "bbbb cc"]


def test_estimated_tokens_split_a_long_unbroken_run() -> None:
    counter = EstimateCounter()
    assert counter.count("x" * 60) == 10  # six characters per estimated token
    chunks = _split("y" * 100, counter, strategy="recursive", size=4)
    assert all(len(c.text) <= 24 for c in chunks)


def test_rebalancing_never_produces_an_oversize_chunk() -> None:
    text = "a bbbbbbb ccc"  # pieces of 2, 8 and 3 units; the only even split would make 11 > 10
    chunks = _split(text, size=10, min_size=4)
    assert [c.text for c in chunks] == ["a bbbbbbb", "ccc"]
    assert all(c.tokens <= 10 for c in chunks)


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_each_chunk_records_the_page_it_starts_on(strategy: str) -> None:
    text = "Page one text.\n\nPage two text.\n\nPage three text."
    pages = (0, text.index("Page two"), text.index("Page three"))
    config = ChunkingConfig(strategy=strategy, unit="chars", size=16, overlap=0, min_size=0)
    chunks = build_chunker(config, CharCounter()).split(Document("d", "t", text, pages=pages))
    assert [(c.text, c.page) for c in chunks] == [
        ("Page one text.", 1),
        ("Page two text.", 2),
        ("Page three text.", 3),
    ]
    no_pages = build_chunker(config, CharCounter()).split(Document("d", "t", text))
    assert {c.page for c in no_pages} == {None}  # never an invented page


def _parent_child(text: str, **cfg: object) -> tuple[list[Chunk], list[Chunk]]:
    config = ChunkingConfig(
        **{
            "strategy": "parent_child",
            "unit": "chars",
            "size": 20,
            "overlap": 0,
            "min_size": 0,
            "parent_size": 60,
            **cfg,
        }
    )
    return build_chunker(config, CharCounter()).split_parents(Document("d", "t", text))


def test_parent_child_children_sit_inside_their_parent() -> None:
    parents, children = _parent_child(EN)
    assert len(parents) > 1 and len(children) > len(parents)
    for p in parents:
        assert EN[p.start : p.end] == p.text and p.tokens <= 60 and p.parent is None
    for c in children:
        p = parents[c.parent]  # type: ignore[index]
        assert p.start <= c.start and c.end <= p.end  # a child never crosses its parent
        assert EN[c.start : c.end] == c.text and c.tokens <= 20
    assert [c.n for c in children] == list(range(len(children)))
    assert {c.parent for c in children} == set(range(len(parents)))  # every parent has children


def test_parent_child_parents_keep_heading_paths() -> None:
    parents, children = _parent_child(MD, parent_size=40)
    refunds = next(c for c in children if "seven days" in c.text)
    assert refunds.heading_path == ("Policies", "Refunds")
    assert parents[refunds.parent].heading_path == ("Policies", "Refunds")  # type: ignore[index]


def test_other_strategies_have_no_parents() -> None:
    config = ChunkingConfig(strategy="recursive", unit="chars", size=40, overlap=0, min_size=0)
    chunker = build_chunker(config, CharCounter())
    parents, children = chunker.split_parents(Document("d", "t", EN))
    assert parents == [] and children == chunker.split(Document("d", "t", EN))


TOPICS = (
    "Coffee needs fresh beans. Grind the coffee just before brewing. Brew coffee at 93 degrees. "
    "Taxes are due in April. File your tax return online. Late tax filing costs a fee."
)


def _topic_vectors(texts: list[str]) -> list[list[float]]:
    """Coffee sentences point one way, tax sentences another."""
    return [[1.0, 0.0] if "offee" in t else [0.0, 1.0] for t in texts]


def _semantic(text: str, **cfg: object) -> list[Chunk]:
    config = ChunkingConfig(
        **{"strategy": "semantic", "unit": "chars", "size": 400, "overlap": 0, "min_size": 0, **cfg}
    )
    return build_chunker(config, CharCounter(), embed=_topic_vectors).split(
        Document("d", "t", text)
    )


def test_semantic_cuts_where_the_topic_changes() -> None:
    chunks = _semantic(TOPICS)
    assert [c.text for c in chunks] == [
        "Coffee needs fresh beans. Grind the coffee just before brewing. "
        "Brew coffee at 93 degrees.",
        "Taxes are due in April. File your tax return online. Late tax filing costs a fee.",
    ]


def test_semantic_does_not_cut_a_single_topic() -> None:
    one_topic = TOPICS.split(" Taxes")[0]
    assert len(_semantic(one_topic)) == 1  # all distances equal: no break


def test_semantic_still_respects_size() -> None:
    chunks = _semantic(TOPICS, size=60)
    assert all(c.tokens <= 60 for c in chunks)
    assert not any("offee" in c.text and "ax" in c.text for c in chunks)  # never mixes topics
    assert all(TOPICS[c.start : c.end] == c.text for c in chunks)


def test_semantic_needs_an_embedder() -> None:
    from lite_rag.errors import ConfigError

    config = ChunkingConfig(strategy="semantic", unit="chars", size=400, overlap=0)
    with pytest.raises(ConfigError, match="embed"):
        build_chunker(config, CharCounter())


def test_chonkie_semantic_keeps_lite_rags_contract() -> None:
    """Exact offsets, `size` as a hard limit, never across a heading. Where chonkie cuts is its
    own method (it needs ~8 sentences before it cuts by meaning at all): judged by the eval."""
    coffee = " ".join(f"Coffee fact number {i} is worth knowing well." for i in range(10))
    tax = " ".join(f"Tax rule number {i} is worth knowing well too." for i in range(10))
    long_sentence = "Tax law " + "word " * 60  # one sentence over size: chonkie keeps it whole
    text = f"# Coffee\n\n{coffee}\n\n# Tax\n\n{tax} {long_sentence}"
    chunks = _semantic(text, semantic_impl="chonkie", size=120)
    assert len(chunks) > 2 and all(text[c.start : c.end] == c.text for c in chunks)
    assert all(c.tokens <= 120 for c in chunks)
    assert {c.heading_path for c in chunks} == {("Coffee",), ("Tax",)}
    assert not any("offee" in c.text and "Tax" in c.text for c in chunks)  # the heading cuts


def test_semantic_distance_ignores_vector_length() -> None:
    def scaled(texts: list[str]) -> list[list[float]]:
        # short coffee vectors, long tax vectors 53 degrees away: a raw dot product would put
        # the largest "distance" between two coffee sentences, not at the topic change
        return [[0.1, 0.0] if "offee" in t else [1.8, 2.4] for t in texts]

    config = ChunkingConfig(strategy="semantic", unit="chars", size=400, overlap=0, min_size=0)
    doc = Document("d", "t", TOPICS)
    by_length = build_chunker(config, CharCounter(), embed=scaled).split(doc)
    assert [c.text for c in by_length] == [c.text for c in _semantic(TOPICS)]


def test_chonkie_keeps_the_size_limit_in_characters_and_for_unbroken_runs() -> None:
    coffee = " ".join(f"Coffee fact number {i} is worth knowing well." for i in range(10))
    text = coffee + " " + "x" * 300  # one run with no break at all
    chunks = _semantic(text, semantic_impl="chonkie", unit="chars", size=80)
    assert all(len(c.text) <= 80 for c in chunks)
    assert all(text[c.start : c.end] == c.text for c in chunks)
