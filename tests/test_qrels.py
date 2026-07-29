"""Unit tests for turning gold rows into relevance judgements.

The normalisation tests are not cosmetic: `pypdf` really does emit `eleva r`,
`0, 25` and `202 6` for these PDFs, and several gold spans exist specifically
to exercise those cases. If normalisation regresses, spans stop matching, every
row silently drops to page-level relevance and every metric moves without any
retrieval change.
"""

from __future__ import annotations

from eval.qrels import (
    MIN_SPAN_CHARS,
    PAGE_GAIN,
    SPAN_GAIN,
    ChunkRef,
    build_row_qrels,
    chunk_key,
    normalize,
    split_span,
)

DOC = "2026-06-17-279a-reuniao"
OTHER = "2025-06-18-271a-reuniao"

CHUNKS = [
    ChunkRef(DOC, 0, 1, "Ata da Reuniao do Comite de Politica Monetaria - Copom"),
    ChunkRef(
        DOC, 5, 6, "O Copom decidiu reduzir a taxa basica de juros para 14,25% a.a., e entende"
    ),
    ChunkRef(DOC, 6, 6, "que essa decisao e compativel com a estrategia de convergencia"),
    ChunkRef(DOC, 7, 7, "Votaram por essa decisao os seguintes membros do Comite"),
    ChunkRef(OTHER, 5, 6, "O Copom decidiu elevar a taxa basica de juros para 15,00% a.a."),
]


class TestNormalize:
    def test_strips_accents(self):
        assert normalize("decisão") == normalize("decisao")

    def test_survives_a_word_split_by_pdf_extraction(self):
        assert normalize("eleva r a taxa") == normalize("elevar a taxa")

    def test_survives_a_number_split_by_pdf_extraction(self):
        assert normalize("0, 25 ponto") == normalize("0,25 ponto")
        assert normalize("para 202 6") == normalize("para 2026")

    def test_is_case_insensitive(self):
        assert normalize("O COPOM") == normalize("o copom")

    def test_drops_punctuation_and_whitespace(self):
        assert normalize("14,25% a.a.") == "1425aa"

    def test_keeps_digits_distinguishable(self):
        assert normalize("14,25%") != normalize("14,75%")


class TestSplitSpan:
    def test_single_span_has_no_page_hint(self):
        assert split_span("uma frase qualquer") == [(None, "uma frase qualquer")]

    def test_pipe_separates_multi_hop_parts(self):
        assert split_span("primeira | segunda") == [(None, "primeira"), (None, "segunda")]

    def test_page_marker_is_parsed_and_removed(self):
        assert split_span("primeira | (p.5) segunda") == [(None, "primeira"), (5, "segunda")]

    def test_tolerates_marker_spacing_variants(self):
        assert split_span("(p. 12) texto") == [(12, "texto")]

    def test_empty_parts_are_dropped(self):
        assert split_span("a |  | b") == [(None, "a"), (None, "b")]


class TestBuildRowQrels:
    def test_span_chunk_outranks_page_chunk(self):
        result = build_row_qrels(
            DOC, 6, "decidiu reduzir a taxa basica de juros para 14,25%", CHUNKS
        )
        assert result.qrels[chunk_key(DOC, 5)] == SPAN_GAIN
        assert result.qrels[chunk_key(DOC, 6)] == PAGE_GAIN

    def test_other_documents_are_never_relevant(self):
        result = build_row_qrels(
            DOC, 6, "decidiu reduzir a taxa basica de juros para 14,25%", CHUNKS
        )
        assert chunk_key(OTHER, 5) not in result.qrels

    def test_other_pages_of_the_same_document_are_not_relevant(self):
        result = build_row_qrels(
            DOC, 6, "decidiu reduzir a taxa basica de juros para 14,25%", CHUNKS
        )
        assert chunk_key(DOC, 0) not in result.qrels
        assert chunk_key(DOC, 7) not in result.qrels

    def test_span_matches_through_pdf_spacing_noise(self):
        result = build_row_qrels(DOC, 6, "decidiu redu zir a taxa basica de ju ros", CHUNKS)
        assert result.qrels[chunk_key(DOC, 5)] == SPAN_GAIN
        assert result.span_matched

    def test_unmatched_span_falls_back_to_page_relevance(self):
        result = build_row_qrels(
            DOC, 6, "uma frase que nao existe em lugar nenhum do corpus", CHUNKS
        )
        assert not result.span_matched
        assert set(result.qrels) == {chunk_key(DOC, 5), chunk_key(DOC, 6)}
        assert set(result.qrels.values()) == {PAGE_GAIN}

    def test_multi_hop_span_marks_both_pages(self):
        span = "decidiu reduzir a taxa basica de juros para 14,25% | (p.7) Votaram por essa decisao"
        result = build_row_qrels(DOC, 6, span, CHUNKS)
        assert result.span_parts == 2
        assert result.span_parts_matched == 2
        assert result.span_matched
        assert result.qrels[chunk_key(DOC, 5)] == SPAN_GAIN
        assert result.qrels[chunk_key(DOC, 7)] == SPAN_GAIN
        assert result.qrels[chunk_key(DOC, 6)] == PAGE_GAIN

    def test_partially_matched_multi_hop_span_is_reported_as_unmatched(self):
        span = "decidiu reduzir a taxa basica de juros para 14,25% | (p.7) frase inexistente aqui"
        result = build_row_qrels(DOC, 6, span, CHUNKS)
        assert result.span_parts_matched == 1
        assert not result.span_matched

    def test_a_span_shorter_than_the_floor_is_ignored(self):
        short = "a" * (MIN_SPAN_CHARS - 1)
        result = build_row_qrels(DOC, 6, short, CHUNKS)
        assert result.span_parts_matched == 0
        assert set(result.qrels.values()) == {PAGE_GAIN}

    def test_a_document_with_no_chunks_yields_no_judgements(self):
        result = build_row_qrels("nao-existe", 3, "qualquer coisa suficientemente longa", CHUNKS)
        assert result.qrels == {}
