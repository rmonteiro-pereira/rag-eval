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


# --------------------------------------------------------------------------
# Written to kill mutation survivors. Each names the mutant it was added for —
# `uv run mutmut run` on retrieval/fusion.py found the fusion tests above
# asserted ordering and nothing else, so fusion could have dropped every
# document's title and text and stayed green.
# --------------------------------------------------------------------------


def test_ties_are_broken_deterministically_by_key():
    """Kills `sorted(..., key=(-score, item[0]))` -> `(-score, item[1])`.

    The test above deliberately accepts either of two tied documents, so nothing
    pinned the tie-break — and a tie-break that varies is not a cosmetic defect
    here. This repo's headline reproducibility claim is that the whole ablation
    regenerates to +-0.0000 from an empty index; that holds only if equal scores
    resolve the same way every run.
    """
    a, b = make("ata-a", 0, "x"), make("ata-b", 0, "x")
    # `b` is seen FIRST, so insertion order and key order disagree. That detail is
    # the whole test: fed a-then-b, Python's stable sort preserves insertion order
    # and a broken tie-break returns the right answer by accident. The first
    # version of this test did exactly that and the mutant survived it.
    fused = reciprocal_rank_fusion([[b, a], [a, b]])
    assert fused[0].score == pytest.approx(fused[1].score)
    assert [hit.key for hit in fused] == ["ata-a#0", "ata-b#0"]
    # And it is stable: same input, same order, however many times it runs.
    for _ in range(5):
        assert [h.key for h in reciprocal_rank_fusion([[b, a], [a, b]])] == [h.key for h in fused]


def test_top_k_actually_truncates():
    """Kills `ordered[: top_k or len(ordered)]` -> `ordered[: top_k and len(...)]`.

    With `top_k=None` the mutant is equivalent, which is why no existing test
    caught it — none passed a `top_k` at all, so the parameter was never
    exercised despite every caller in the repo using it.
    """
    hits = [make(f"ata-{i}", 0, "x") for i in range(5)]
    assert len(reciprocal_rank_fusion([hits], top_k=2)) == 2
    assert len(reciprocal_rank_fusion([hits], top_k=None)) == 5
    assert len(reciprocal_rank_fusion([hits])) == 5


def test_fusion_carries_the_document_through_intact():
    """Kills the whole `title=None` / `text=None` / `url=None` family.

    Fusion rebuilds each record rather than reusing it, so every field is a place
    a document's identity can be dropped. These fields are what the answer cites;
    losing them would be invisible to an ordering assertion and very visible to a
    reader.
    """
    original = make("ata-a", 3, "o Copom decidiu manter a taxa")
    fused = reciprocal_rank_fusion([[original]])[0]
    for field in ("doc_id", "title", "url", "reference_date", "chunk_index", "page_number", "text"):
        assert getattr(fused, field) == getattr(original, field), field


def test_the_rrf_signal_records_the_score_it_ranked_by():
    """Kills `record.signals["rrf"] = None`.

    `signals` exists so a report can say *why* something ranked where it did. The
    existing test asserted the key was present, not that it held anything.
    """
    fused = reciprocal_rank_fusion([[make("ata-a", 0, "x")]])
    assert fused[0].signals["rrf"] == pytest.approx(fused[0].score)
    assert fused[0].signals["rrf"] == pytest.approx(1 / 61)
