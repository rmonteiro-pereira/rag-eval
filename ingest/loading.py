"""PDF -> text.

Copom minutes are text-based PDFs (not scans), so `pypdf` is enough and keeps
the dependency footprint small. Page boundaries are preserved because the gold
eval set (M2) needs to point at a *span* inside a specific page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# BACEN PDFs carry a running header/footer on every page; stripping it stops the
# same boilerplate from polluting every chunk (and every retrieval result).
_NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),  # bare page numbers
    re.compile(r"Banco Central do Brasil\s*$", re.IGNORECASE),
]


@dataclass
class Page:
    page_number: int  # 1-indexed, as printed
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    url: str
    reference_date: str
    pages: list[Page]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def _clean(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.replace("\xa0", " ").rstrip()
        if any(p.search(line) for p in _NOISE_PATTERNS):
            continue
        lines.append(line)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def load_pdf(path: Path, doc_id: str, title: str, url: str, reference_date: str) -> Document:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append(Page(page_number=i, text=text))
    return Document(
        doc_id=doc_id,
        title=title,
        url=url,
        reference_date=reference_date,
        pages=pages,
    )
