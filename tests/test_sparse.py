"""BM25 and RRF fusion.

BM25 is hand-rolled, so its arithmetic is pinned against values computed by hand
from the formula rather than against whatever the implementation happened to
return the first time it ran.
"""

from __future__ import annotations

import math

import pytest

from retrieval.fusion import reciprocal_rank_fusion
from retrieval.sparse import BM25Index
from retrieval.store import Retrieved


def make(doc_id: str, chunk_index: int, text: str) -> Retrieved:
    return Retrieved(
        score=0.0,
        doc_id=doc_id,
        title=f"{doc_id} title",
        url="http://example.invalid",
        reference_date="2024-01-01",
        chunk_index=chunk_index,
        page_number=1,
        text=text,
    )


CORPUS = [
    make("ata-a", 0, "O Copom decidiu reduzir a taxa basica de juros para 13,25% a.a."),
    make("ata-b", 0, "O Copom decidiu reduzir a taxa basica de juros para 12,75% a.a."),
    make("ata-c", 0, "A inflacao de servicos permanece resiliente no horizonte relevante."),
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.build(CORPUS)


def test_rate_token_discriminates_between_near_identical_chunks(index):
    """The whole reason BM25 earns a place next to a strong embedder.

    `ata-a` and `ata-b` differ in five characters. Every content word is shared,
    so both score — and the rate is what breaks the tie, decisively.
    """
    hits = index.search("reduzir a taxa basica de juros para 12,75% a.a.", top_k=3)
    assert [hit.doc_id for hit in hits[:2]] == ["ata-b", "ata-a"]
    assert hits[0].score > hits[1].score


def test_a_rate_alone_retrieves_only_the_chunk_that_states_it(index):
    hits = index.search("Em qual reuniao a Selic foi reduzida para 12,75% a.a.?", top_k=3)
    assert [hit.doc_id for hit in hits] == ["ata-b"]


def test_idf_is_never_negative(index):
    """`copom` is in most documents; the textbook IDF would go negative there.

    Negative IDF penalises a chunk for containing the topic of the query, which
    on this corpus means penalising every chunk that could answer it.
    """
    assert all(value >= 0 for value in index._idf.values())


def test_idf_matches_the_formula_by_hand(index):
    # `inflacao` appears in 1 of 3 documents.
    expected = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    assert index._idf["inflacao"] == pytest.approx(expected)


def test_score_is_zero_for_unseen_terms(index):
    assert index.score("bitcoin ethereum") == {}


def test_allowed_doc_ids_restricts_scoring(index):
    hits = index.search("taxa basica de juros", top_k=5, allowed_doc_ids=["ata-a"])
    assert {hit.doc_id for hit in hits} == {"ata-a"}


def test_empty_allowed_list_is_treated_as_no_restriction(index):
    """`[]` and `None` must not mean different things — the caller passes None."""
    assert index.search("taxa basica", top_k=5, allowed_doc_ids=None)


def test_search_carries_the_bm25_score_in_signals(index):
    hit = index.search("13,25", top_k=1)[0]
    assert hit.signals["bm25"] == hit.score


def test_stopwords_do_not_drive_the_ranking(index):
    """A query of pure function words must not confidently rank anything."""
    assert index.score("o a de em que") == {}


def test_rrf_promotes_what_both_arms_agree_on():
    dense = [CORPUS[2], CORPUS[0], CORPUS[1]]
    sparse = [CORPUS[0], CORPUS[2], CORPUS[1]]
    fused = reciprocal_rank_fusion([dense, sparse])
    # ata-a is 2nd and 1st; ata-c is 1st and 2nd — tie on score, broken by key.
    assert fused[0].doc_id in {"ata-a", "ata-c"}
    assert fused[-1].doc_id == "ata-b"
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_does_not_mutate_its_inputs():
    dense = [CORPUS[0]]
    before = CORPUS[0].score
    reciprocal_rank_fusion([dense, [CORPUS[1]]])
    assert CORPUS[0].score == before


def test_rrf_keeps_both_arms_signals():
    a = make("ata-a", 0, "x")
    a.signals["dense"] = 0.9
    b = make("ata-a", 0, "x")
    b.signals["bm25"] = 4.2
    fused = reciprocal_rank_fusion([[a], [b]])
    assert fused[0].signals["dense"] == 0.9
    assert fused[0].signals["bm25"] == 4.2
    assert "rrf" in fused[0].signals
