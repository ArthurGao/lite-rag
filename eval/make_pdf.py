"""Render a small Markdown file as a structured PDF, for loader tests and the PDF eval corpus.

Not a dependency: run it on demand (the PDFs it makes are committed):

    uv run --with reportlab==5.1.0 python eval/make_pdf.py IN.md OUT.pdf

Supported Markdown: `#`..`###` headings, paragraphs (blank-line separated) and a line holding
only `<!-- pagebreak -->`. Every page break is explicit, so tests know which page holds what.
One CID font (STSong-Light, built into reportlab) covers Chinese and Latin text.
"""

import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate

FONT = "STSong-Light"
STYLES = {
    0: ParagraphStyle("body", fontName=FONT, fontSize=11, leading=17, spaceAfter=8),
    1: ParagraphStyle("h1", fontName=FONT, fontSize=20, leading=26, spaceAfter=14),
    2: ParagraphStyle("h2", fontName=FONT, fontSize=15, leading=21, spaceBefore=8, spaceAfter=10),
    3: ParagraphStyle("h3", fontName=FONT, fontSize=12.5, leading=18, spaceBefore=6, spaceAfter=8),
}


def flowables(markdown: str) -> list[Flowable]:
    out: list[Flowable] = []
    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block == "<!-- pagebreak -->":
            out.append(PageBreak())
            continue
        level = len(block) - len(block.lstrip("#")) if block.startswith("#") else 0
        text = block[level:].strip() if level else " ".join(block.split("\n"))
        out.append(Paragraph(escape(text), STYLES[min(level, 3)]))
    return out


def main(source: str, target: str) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    markdown = Path(source).read_text("utf-8")
    title = next((line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")), "")
    doc = SimpleDocTemplate(target, pagesize=A4, title=title)
    doc.build(flowables(markdown))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
