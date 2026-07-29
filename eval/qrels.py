"""Turning gold rows into complete relevance judgements over the corpus.

Recall needs a denominator, and the only honest denominator is *every* relevant
chunk in the collection — not just the ones that happened to come back. So the
runner scrolls all 636 chunks out of Qdrant once and asks, per gold row, which
of them are relevant. At this corpus size that is cheap and exact; there is no
reason to approximate it.

Two grades:

* **2 — span chunk.** The chunk actually contains the gold `source_span`.
  This is the chunk the generator needs in its context to answer at all.
* **1 — page chunk.** Same document, same printed page, but the span is not in
  it. Retrieving it is genuinely useful (right document, right page) yet it may
  not carry the answer, so it should not count the same as a span chunk.

Matching is done on a normalised form — accents stripped, case folded, and all
non-alphanumeric characters (including whitespace) removed. That is not
cosmetic: `pypdf` extraction of these PDFs produces `eleva r`, `0, 25` and
`202 6`, and the gold file stores unaccented ASCII. Normalising both sides is
what makes "is this span in this chunk" answerable at all.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: A span part may be prefixed with the page it came from, e.g. "(p.5) ...",
#: which is how multi-hop rows record their second and later spans.
_PAGE_MARKER = re.compile(r"^\(\s*p\.?\s*(\d+)\s*\)\s*")

_NON_ALNUM = re.compile(r"[^0-9a-z]+")

#: Below this many normalised characters a span part is too generic to identify
#: a chunk; treat it as unusable rather than let it match half the corpus.
MIN_SPAN_CHARS = 20

SPAN_GAIN = 2
PAGE_GAIN = 1


@dataclass(frozen=True)
class ChunkRef:
    """A chunk as stored in the vector store, reduced to what qrels need."""

    doc_id: str
    chunk_index: int
    page_number: int
    text: str

    @property
    def key(self) -> str:
        return chunk_key(self.doc_id, self.chunk_index)


def chunk_key(doc_id: str, chunk_index: int) -> str:
    """Stable identity of a chunk. `chunk_index` is unique within a document."""
    return f"{doc_id}#{chunk_index}"


def normalize(text: str) -> str:
    """Fold to a form where PDF extraction noise cannot cause a false miss."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM.sub("", ascii_only.lower())


def split_span(source_span: str) -> list[tuple[int | None, str]]:
    """Split a `|`-separated multi-hop span into (page_hint, text) parts."""
    parts: list[tuple[int | None, str]] = []
    for raw in source_span.split("|"):
        chunk = raw.strip()
        if not chunk:
            continue
        page_hint: int | None = None
        marker = _PAGE_MARKER.match(chunk)
        if marker:
            page_hint = int(marker.group(1))
            chunk = chunk[marker.end() :].strip()
        if chunk:
            parts.append((page_hint, chunk))
    return parts


@dataclass(frozen=True)
class RowQrels:
    """Relevance judgements for one gold row, plus why they came out that way."""

    qrels: dict[str, int]
    span_parts: int
    span_parts_matched: int

    @property
    def span_matched(self) -> bool:
        """True when every span part was located in at least one chunk.

        A False here is a finding about the *gold row*, not about retrieval:
        either the span was mistyped or the chunker split it. The runner
        surfaces these in the report instead of hiding them.
        """
        return self.span_parts > 0 and self.span_parts_matched == self.span_parts


def build_row_qrels(
    source_doc_id: str,
    source_page: int | None,
    source_span: str | None,
    chunks: Iterable[ChunkRef],
) -> RowQrels:
    """Judge every chunk in the corpus against one answerable gold row."""
    doc_chunks = [chunk for chunk in chunks if chunk.doc_id == source_doc_id]
    parts = split_span(source_span) if source_span else []

    qrels: dict[str, int] = {}
    matched = 0

    pages = {source_page} if source_page is not None else set()

    for page_hint, part in parts:
        # The page a span part came from is useful context even when the part
        # itself straddles a chunk boundary, or is too short to identify one.
        if page_hint is not None:
            pages.add(page_hint)
        needle = normalize(part)
        if len(needle) < MIN_SPAN_CHARS:
            continue
        hits = [chunk for chunk in doc_chunks if needle in normalize(chunk.text)]
        if hits:
            matched += 1
            for chunk in hits:
                qrels[chunk.key] = SPAN_GAIN

    for chunk in doc_chunks:
        if chunk.page_number in pages:
            qrels.setdefault(chunk.key, PAGE_GAIN)

    return RowQrels(qrels=qrels, span_parts=len(parts), span_parts_matched=matched)


def gains_for_ranking(ranking: Sequence[str], qrels: Mapping[str, int]) -> list[int]:
    """Gain of each retrieved chunk, in rank order — for report readability."""
    return [qrels.get(key, 0) for key in ranking]
