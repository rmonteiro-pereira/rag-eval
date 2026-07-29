"""BM25 lexical retrieval over the indexed corpus.

Written out rather than pulled from `rank_bm25`, for two reasons: the whole
thing is forty lines of arithmetic, and the ablation is more defensible when the
scoring function it compares against is visible in the repo instead of behind a
dependency pin.

This is the Robertson/Sparck-Jones BM25 with the standard `k1=1.5, b=0.75` and
the **non-negative** IDF variant

    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

The `1 +` matters: the textbook form goes negative for terms present in more
than half the documents, and in a corpus of 636 near-identical chunks of central
bank minutes that describes words like `copom`, `inflacao` and `selic` — exactly
the words in every question. Negative IDF would actively *penalise* documents
for containing the topic.

The index is built from what is already in Qdrant, so dense and sparse always
see byte-identical text and the ablation compares retrieval strategies rather
than two different corpora.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from retrieval.store import Retrieved
from retrieval.text import tokenize

K1 = 1.5
B = 0.75


def chunk_key(doc_id: str, chunk_index: int) -> str:
    """Stable identity of a chunk — mirrors `eval.qrels.chunk_key`."""
    return f"{doc_id}#{chunk_index}"


@dataclass
class BM25Index:
    """An inverted index over `Retrieved` records, scored with BM25.

    `records` is the corpus in a fixed order; every other structure indexes into
    it positionally.
    """

    records: list[Retrieved]
    k1: float = K1
    b: float = B
    _postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _lengths: list[int] = field(default_factory=list, repr=False)
    _avgdl: float = 0.0

    @classmethod
    def build(cls, records: Iterable[Retrieved], k1: float = K1, b: float = B) -> BM25Index:
        index = cls(records=list(records), k1=k1, b=b)
        index._index()
        return index

    def _index(self) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, record in enumerate(self.records):
            tokens = tokenize(record.text)
            self._lengths.append(len(tokens))
            frequencies: dict[str, int] = defaultdict(int)
            for token in tokens:
                frequencies[token] += 1
            for token, count in frequencies.items():
                postings[token].append((position, count))

        n = len(self.records) or 1
        self._postings = dict(postings)
        self._avgdl = (sum(self._lengths) / n) if self._lengths else 0.0
        self._idf = {
            token: math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for token, plist in self._postings.items()
        }

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def score(self, query: str, allowed_doc_ids: Sequence[str] | None = None) -> dict[int, float]:
        """BM25 score per corpus position; positions scoring 0 are omitted.

        `allowed_doc_ids` restricts scoring to a document subset — the same
        restriction the dense side expresses as a Qdrant payload filter, so
        metadata filtering means the same thing on both arms of the hybrid.
        """
        allowed = set(allowed_doc_ids) if allowed_doc_ids is not None else None
        scores: dict[int, float] = defaultdict(float)

        for token in tokenize(query):
            plist = self._postings.get(token)
            if not plist:
                continue
            idf = self._idf[token]
            for position, tf in plist:
                if allowed is not None and self.records[position].doc_id not in allowed:
                    continue
                length_norm = 1 - self.b + self.b * (self._lengths[position] / (self._avgdl or 1))
                scores[position] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)

        return dict(scores)

    def search(
        self,
        query: str,
        top_k: int,
        allowed_doc_ids: Sequence[str] | None = None,
    ) -> list[Retrieved]:
        """Top-`k` chunks by BM25, as `Retrieved` carrying the BM25 score."""
        scores = self.score(query, allowed_doc_ids)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        hits: list[Retrieved] = []
        for position, score in ranked:
            record = self.records[position]
            hits.append(
                Retrieved(
                    score=score,
                    doc_id=record.doc_id,
                    title=record.title,
                    url=record.url,
                    reference_date=record.reference_date,
                    chunk_index=record.chunk_index,
                    page_number=record.page_number,
                    text=record.text,
                    signals={"bm25": score},
                )
            )
        return hits
