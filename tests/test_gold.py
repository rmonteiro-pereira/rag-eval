"""Tests for the gold-set loader, including the guarantees the project depends on.

The last class is the important one: it asserts, against the real committed
file, that nothing is marked `validated`. Validation is Rodrigo's human pass and
the scientific asset of this project — an agent flipping that flag would be the
single most damaging silent change possible here, so it fails a test rather
than a review.
"""

from __future__ import annotations

import json

import pytest

from eval.gold import DEFAULT_GOLD_PATH, GoldSetError, load_gold, status_counts

ANSWERABLE = {
    "id": "t-001",
    "status": "draft",
    "question": "pergunta?",
    "answer": "resposta",
    "answer_type": "extractive",
    "source_doc_id": "doc-a",
    "source_title": "Doc A",
    "source_page": 3,
    "source_span": "um trecho",
    "difficulty": "easy",
    "capability": "single-hop lookup",
    "notes": "",
}

NEGATIVE = {
    **ANSWERABLE,
    "id": "t-002",
    "answer_type": "abstention",
    "source_doc_id": None,
    "source_title": None,
    "source_page": None,
    "source_span": None,
}


def write_jsonl(tmp_path, records, comment=True):
    path = tmp_path / "gold.jsonl"
    lines = [json.dumps({"_comment": "header"})] if comment else []
    lines += [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestLoading:
    def test_comment_record_is_not_a_test_case(self, tmp_path):
        rows = load_gold(write_jsonl(tmp_path, [ANSWERABLE]))
        assert [r.id for r in rows] == ["t-001"]

    def test_blank_lines_are_skipped(self, tmp_path):
        path = write_jsonl(tmp_path, [ANSWERABLE])
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(load_gold(path)) == 1

    def test_abstention_rows_are_flagged(self, tmp_path):
        rows = load_gold(write_jsonl(tmp_path, [ANSWERABLE, NEGATIVE]))
        assert [r.is_abstention for r in rows] == [False, True]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(GoldSetError):
            load_gold(tmp_path / "nope.jsonl")


class TestStatusFilter:
    def test_draft_keeps_everything_scorable(self, tmp_path):
        path = write_jsonl(
            tmp_path, [ANSWERABLE, {**ANSWERABLE, "id": "t-9", "status": "validated"}]
        )
        assert {r.id for r in load_gold(path, min_status="draft")} == {"t-001", "t-9"}

    def test_validated_keeps_only_validated_rows(self, tmp_path):
        path = write_jsonl(
            tmp_path, [ANSWERABLE, {**ANSWERABLE, "id": "t-9", "status": "validated"}]
        )
        assert [r.id for r in load_gold(path, min_status="validated")] == ["t-9"]

    def test_rejected_rows_are_never_scored(self, tmp_path):
        path = write_jsonl(tmp_path, [{**ANSWERABLE, "status": "rejected"}])
        assert load_gold(path, min_status="draft") == []

    def test_unknown_min_status_raises(self, tmp_path):
        with pytest.raises(GoldSetError):
            load_gold(write_jsonl(tmp_path, [ANSWERABLE]), min_status="pending")


class TestSchemaValidation:
    def test_missing_field_raises(self, tmp_path):
        broken = {k: v for k, v in ANSWERABLE.items() if k != "source_span"}
        with pytest.raises(GoldSetError, match="missing field"):
            load_gold(write_jsonl(tmp_path, [broken]))

    def test_unknown_status_raises(self, tmp_path):
        with pytest.raises(GoldSetError, match="unknown status"):
            load_gold(write_jsonl(tmp_path, [{**ANSWERABLE, "status": "maybe"}]))

    def test_unknown_answer_type_raises(self, tmp_path):
        with pytest.raises(GoldSetError, match="unknown answer_type"):
            load_gold(write_jsonl(tmp_path, [{**ANSWERABLE, "answer_type": "guess"}]))

    def test_answerable_row_without_a_span_raises(self, tmp_path):
        with pytest.raises(GoldSetError, match="no source_span"):
            load_gold(write_jsonl(tmp_path, [{**ANSWERABLE, "source_span": None}]))

    def test_answerable_row_without_a_doc_raises(self, tmp_path):
        with pytest.raises(GoldSetError, match="no source_doc_id"):
            load_gold(write_jsonl(tmp_path, [{**ANSWERABLE, "source_doc_id": None}]))

    def test_abstention_row_naming_a_source_raises(self, tmp_path):
        with pytest.raises(GoldSetError, match="names a source doc"):
            load_gold(write_jsonl(tmp_path, [{**NEGATIVE, "source_doc_id": "doc-a"}]))

    def test_duplicate_id_raises(self, tmp_path):
        with pytest.raises(GoldSetError, match="duplicate id"):
            load_gold(write_jsonl(tmp_path, [ANSWERABLE, ANSWERABLE]))

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "gold.jsonl"
        path.write_text('{"id": "t-1"\n', encoding="utf-8")
        with pytest.raises(GoldSetError, match="not valid JSON"):
            load_gold(path)


class TestCommittedGoldSet:
    """Guarantees about the file that actually ships."""

    def test_it_loads(self):
        assert load_gold(DEFAULT_GOLD_PATH, min_status="draft")

    def test_nothing_is_marked_validated(self):
        # Human gate. An agent must never flip this flag; see eval/datasets/README.md.
        assert status_counts(DEFAULT_GOLD_PATH).get("validated", 0) == 0

    def test_it_has_at_least_fifty_rows(self):
        assert len(load_gold(DEFAULT_GOLD_PATH, min_status="draft")) >= 50

    def test_it_carries_enough_abstention_negatives(self):
        rows = load_gold(DEFAULT_GOLD_PATH, min_status="draft")
        assert sum(1 for r in rows if r.is_abstention) >= 5

    def test_it_carries_enough_reverse_lookup_probes(self):
        rows = load_gold(DEFAULT_GOLD_PATH, min_status="draft")
        assert sum(1 for r in rows if "reverse lookup" in r.capability) >= 5

    def test_every_answerable_row_has_a_span(self):
        rows = load_gold(DEFAULT_GOLD_PATH, min_status="draft")
        assert all(r.source_span for r in rows if not r.is_abstention)

    def test_every_source_doc_id_exists_in_the_manifest(self):
        from rag.config import REPO_ROOT

        manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
        known = {d["doc_id"] for d in manifest["documents"]}
        rows = load_gold(DEFAULT_GOLD_PATH, min_status="draft")
        assert {r.source_doc_id for r in rows if r.source_doc_id} <= known

    def test_validated_filter_currently_selects_nothing(self):
        # The corollary of the human gate: until Rodrigo validates rows, the
        # "number that counts" is deliberately unavailable.
        assert load_gold(DEFAULT_GOLD_PATH, min_status="validated") == []
