"""Unit tests for the retrieval metric math.

Fixtures are hand-computed, not produced by the code under test. The nDCG case
in `test_dcg_matches_textbook_example` is the standard worked example with
relevance grades [3, 2, 3, 0, 1, 2], whose DCG (6.861) and nDCG (0.961) are
published values — so a silent change to the discount or the gain formulation
breaks the test rather than quietly moving every number in every report.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics.retrieval import (
    aggregate,
    dcg,
    first_relevant_rank,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_ranking,
)

# Three relevant chunks: one span chunk (gain 2) and two page chunks (gain 1).
QRELS = {"a": 2, "b": 1, "c": 1}

# a is at rank 2, c at rank 4, b at rank 6. Everything else is irrelevant.
RANKING = ["x", "a", "y", "c", "z", "b", "p", "q", "r", "s"]


class TestRecall:
    @pytest.mark.parametrize(
        ("k", "expected"),
        [(1, 0.0), (2, 1 / 3), (3, 1 / 3), (4, 2 / 3), (5, 2 / 3), (6, 1.0), (10, 1.0)],
    )
    def test_recall_at_k(self, k, expected):
        assert recall_at_k(RANKING, QRELS, k) == pytest.approx(expected)

    def test_recall_denominator_is_the_whole_relevant_set(self):
        # Only one of three relevant chunks was retrieved at all -> 1/3, not 1/1.
        assert recall_at_k(["a", "x", "y"], QRELS, 10) == pytest.approx(1 / 3)

    def test_recall_at_1_is_capped_by_the_size_of_the_relevant_set(self):
        assert recall_at_k(["a", "b", "c"], QRELS, 1) == pytest.approx(1 / 3)

    def test_perfect_ranking_reaches_one(self):
        assert recall_at_k(["a", "b", "c", "x"], QRELS, 3) == pytest.approx(1.0)

    def test_non_positive_k_is_rejected(self):
        with pytest.raises(ValueError):
            recall_at_k(RANKING, QRELS, 0)


class TestHitRate:
    @pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (3, 1.0), (10, 1.0)])
    def test_hit_rate_at_k(self, k, expected):
        assert hit_rate_at_k(RANKING, QRELS, k) == expected

    def test_hit_rate_is_zero_when_nothing_relevant_was_retrieved(self):
        assert hit_rate_at_k(["x", "y", "z"], QRELS, 3) == 0.0

    def test_hit_rate_ignores_how_many_relevant_chunks_were_found(self):
        assert hit_rate_at_k(["a", "b", "c"], QRELS, 3) == hit_rate_at_k(["a", "x", "y"], QRELS, 3)


class TestReciprocalRank:
    def test_reciprocal_rank_uses_the_first_relevant_hit(self):
        assert reciprocal_rank(RANKING, QRELS) == pytest.approx(0.5)

    def test_rank_one_gives_one(self):
        assert reciprocal_rank(["b", "x", "a"], QRELS) == pytest.approx(1.0)

    def test_no_relevant_hit_gives_zero(self):
        assert reciprocal_rank(["x", "y", "z"], QRELS) == 0.0

    def test_first_relevant_rank_is_one_indexed(self):
        assert first_relevant_rank(RANKING, QRELS) == 2
        assert first_relevant_rank(["x", "y"], QRELS) is None

    def test_grade_does_not_affect_reciprocal_rank(self):
        # b has gain 1, a has gain 2; MRR binarises, so b at rank 1 wins.
        assert reciprocal_rank(["b", "a"], QRELS) == pytest.approx(1.0)


class TestDcgAndNdcg:
    def test_dcg_matches_textbook_example(self):
        assert dcg([3, 2, 3, 0, 1, 2]) == pytest.approx(6.861126688593502)

    def test_dcg_first_position_is_undiscounted(self):
        assert dcg([5]) == pytest.approx(5.0)

    def test_dcg_second_position_is_halved(self):
        assert dcg([0, 4]) == pytest.approx(4 / math.log2(3))

    def test_ndcg_matches_textbook_example(self):
        qrels = {"d1": 3, "d2": 2, "d3": 3, "d4": 0, "d5": 1, "d6": 2}
        ranking = ["d1", "d2", "d3", "d4", "d5", "d6"]
        assert ndcg_at_k(ranking, qrels, 6) == pytest.approx(0.9608081943360617)

    @pytest.mark.parametrize(
        ("k", "expected"),
        [
            (1, 0.0),
            (3, 0.4030302838010049),
            (5, 0.5405857679450102),
            (10, 0.6543561860457984),
        ],
    )
    def test_ndcg_at_k(self, k, expected):
        assert ndcg_at_k(RANKING, QRELS, k) == pytest.approx(expected)

    def test_ideal_ranking_scores_one(self):
        assert ndcg_at_k(["a", "b", "c"], QRELS, 3) == pytest.approx(1.0)

    def test_ndcg_rewards_putting_the_span_chunk_first(self):
        span_first = ndcg_at_k(["a", "b", "c"], QRELS, 3)
        page_first = ndcg_at_k(["b", "a", "c"], QRELS, 3)
        assert span_first > page_first

    def test_ideal_is_truncated_at_k(self):
        # Only one slot, so the ideal is the single best gain -> retrieving it scores 1.
        assert ndcg_at_k(["a"], QRELS, 1) == pytest.approx(1.0)


class TestEmptyRelevantSet:
    """Scoring a query with no relevant chunk is a category error, not a 0.0."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda q: recall_at_k(RANKING, q, 5),
            lambda q: hit_rate_at_k(RANKING, q, 5),
            lambda q: reciprocal_rank(RANKING, q),
            lambda q: ndcg_at_k(RANKING, q, 5),
            lambda q: score_ranking(RANKING, q),
        ],
    )
    def test_raises_on_empty_qrels(self, call):
        with pytest.raises(ValueError):
            call({})

    def test_raises_when_all_gains_are_zero(self):
        with pytest.raises(ValueError):
            recall_at_k(RANKING, {"a": 0}, 5)


class TestScoreRanking:
    def test_bundles_the_hand_computed_values(self):
        scores = score_ranking(RANKING, QRELS, ks=(1, 3, 5, 10))
        assert scores.recall[3] == pytest.approx(1 / 3)
        assert scores.recall[10] == pytest.approx(1.0)
        assert scores.hit_rate[1] == 0.0
        assert scores.hit_rate[3] == 1.0
        assert scores.ndcg[5] == pytest.approx(0.5405857679450102)
        assert scores.mrr == pytest.approx(0.5)
        assert scores.first_relevant_rank == 2
        assert scores.n_relevant_total == 3
        assert scores.n_retrieved == 10

    def test_to_json_is_flat_and_serialisable(self):
        payload = score_ranking(RANKING, QRELS, ks=(1, 5)).to_json()
        assert payload["recall@1"] == 0.0
        assert payload["hit_rate@5"] == 1.0
        assert payload["mrr"] == pytest.approx(0.5)
        assert set(payload) == {
            "recall@1",
            "recall@5",
            "hit_rate@1",
            "hit_rate@5",
            "ndcg@1",
            "ndcg@5",
            "mrr",
            "first_relevant_rank",
            "n_relevant_total",
            "n_retrieved",
        }


class TestAggregate:
    def test_macro_average_weighs_every_query_equally(self):
        perfect = score_ranking(["a", "b", "c"], QRELS, ks=(1, 3))
        miss = score_ranking(["x", "y", "z"], QRELS, ks=(1, 3))
        agg = aggregate([perfect, miss])
        assert agg["recall@3"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert agg["hit_rate@1"] == pytest.approx(0.5)
        assert agg["mrr"] == pytest.approx(0.5)
        assert agg["n_queries"] == 2

    def test_a_query_with_many_relevant_chunks_does_not_dominate(self):
        big = score_ranking(["m"], {f"m{i}": 1 for i in range(20)} | {"m": 1}, ks=(1,))
        small = score_ranking(["s"], {"s": 1}, ks=(1,))
        # Micro-averaging would give 2/22; macro gives (1/21 + 1) / 2.
        assert aggregate([big, small])["recall@1"] == pytest.approx((1 / 21 + 1.0) / 2)

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError):
            aggregate([])
