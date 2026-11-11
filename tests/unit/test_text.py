import pytest

from lite_rag.text import CharCounter, EstimateCounter, HFTokenCounter, terms


def test_terms_segment_chinese_into_words_and_drop_stopwords_and_punctuation() -> None:
    assert terms("我们的退款政策：七天内退款！") == ["退款", "政策", "七天", "内", "退款"]


def test_terms_lowercase_english_and_keep_numbers() -> None:
    assert terms("Refunds take 7 days, the FAQ says.") == [
        "refunds",
        "take",
        "7",
        "days",
        "faq",
        "says",
    ]


def test_estimate_spans_point_at_the_tokens() -> None:
    text = "我住 Beijing!"
    spans = EstimateCounter().spans(text)
    assert [text[a:b] for a, b in spans] == ["我", "住", "Beijin", "g", "!"]  # runs of up to 6


def test_char_counter_counts_characters() -> None:
    assert CharCounter().count("退款 ok") == 5


def test_a_tokenizer_shipped_with_truncation_still_covers_the_whole_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # paraphrase-multilingual-MiniLM-L12-v2's tokenizer.json enables truncation at 128 tokens;
    # sizing chunks with it must not drop the text past that point.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    def truncating(name: str) -> Tokenizer:
        tok = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        tok.enable_truncation(max_length=3)
        return tok

    monkeypatch.setattr(Tokenizer, "from_pretrained", staticmethod(truncating))
    counter = HFTokenCounter("some/model")
    text = "one two three four five"
    assert counter.count(text) == 5
    assert [text[s:e] for s, e in counter.spans(text)] == text.split()
