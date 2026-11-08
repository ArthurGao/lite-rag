from lite_rag.text import CharCounter, EstimateCounter, terms


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
