from pathlib import Path

import pytest

from lite_rag import loaders
from lite_rag.errors import InvalidRequest
from lite_rag.loaders import join_pages, load_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_pages_are_joined_and_each_start_is_recorded() -> None:
    text, starts = join_pages([" one \n", "two", "three"])
    assert text == "one\n\ntwo\n\nthree"
    assert [text[s : s + 3] for s in starts] == ["one", "two", "thr"]


def test_pypdf_reads_each_page_and_where_it_starts() -> None:
    loaded = load_file(FIXTURES / "pages.pdf")
    assert loaded.title == "Refund Handbook"  # the PDF's own title
    assert len(loaded.pages) == 3
    page_texts = [
        loaded.text[a:b]
        for a, b in zip(loaded.pages, [*loaded.pages[1:], len(loaded.text)], strict=True)
    ]
    assert "seven days" in page_texts[0]
    assert page_texts[1].startswith("退货规则") and "十五天" in page_texts[1]
    assert page_texts[2].startswith("Shipping")


def test_docling_output_keeps_headings_and_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    markdown = "## Handbook\n\nIntro.\n\n<!-- lite-rag:page-break -->\n\n## Shipping\n\nFree."
    seen: list[Path] = []
    monkeypatch.setattr(loaders, "docling_markdown", lambda p: seen.append(p) or markdown)
    doc = tmp_path / "policy.docx"
    loaded = load_file(doc, pdf="pypdf")  # docx always goes through docling
    assert seen == [doc]
    assert loaded.title == "Handbook"  # docling writes every heading as ##
    assert loaded.text == "## Handbook\n\nIntro.\n\n## Shipping\n\nFree."
    assert loaded.text[loaded.pages[1] :].startswith("## Shipping")


def test_markdown_title_is_the_first_h1_and_text_files_have_no_pages(tmp_path: Path) -> None:
    md = tmp_path / "a.md"
    md.write_text("## Not the title\n\n# Title\n\nBody.")
    assert (load_file(md).title, load_file(md).pages) == ("Title", ())
    with pytest.raises(InvalidRequest, match="unsupported file type"):
        load_file(tmp_path / "a.rtf")


def test_pages_join_the_content_hash_only_when_present() -> None:
    import hashlib
    import json

    from lite_rag.engine import content_hash

    old = hashlib.sha256(json.dumps(["t", "x", {}], ensure_ascii=False).encode()).hexdigest()
    assert content_hash("t", "x", {}) == old  # documents without pages keep their hash
    assert content_hash("t", "x", {}, (0, 5)) not in (old, content_hash("t", "x", {}, (0, 6)))


def test_an_empty_page_keeps_later_pages_on_their_own_number() -> None:
    from bisect import bisect_right

    text, starts = join_pages(["one", "", "three"])
    assert bisect_right(starts, text.index("three")) == 3  # not page 2
