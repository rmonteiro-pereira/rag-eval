"""Generation metrics — deterministic first, judge second.

The order is the point. An LLM judge is the flexible instrument and the
untrustworthy one: on this project it is a 3B/8B model running locally, grading
answers written by a 3B/8B model running locally, with nobody having checked
either. Reporting a judge score as *the* generation number would be measuring
one unvalidated model with another and calling the agreement a result.

So everything that can be computed without a model, is:

* **`numeric_recall`** — of the numbers in the reference answer, how many appear
  in the generated one. On a corpus about interest rates this is close to a
  correctness oracle: `14,25` is either there or it is not, and the reference
  answer's numbers came from the source document.
* **`unsupported_numbers`** — numbers in the answer that appear in *neither* the
  retrieved context nor the question. This is the hallucination detector that
  matters here. A fabricated policy rate is the failure mode with consequences,
  and unlike "is this claim supported", it is exactly decidable.
* **`lexical_groundedness`** — share of the answer's content words present in the
  retrieved context. Crude, and deliberately so: it cannot detect a fluent
  misreading, but it cannot be talked into anything either.
* **`abstained`** — did the answer refuse. Scored against the seven negatives,
  where refusing is the correct behaviour, and against answerable rows, where it
  is a false refusal.
* **`cited_gold_doc`** — did the answer's citations include the document the gold
  row names.

The judge (`generation/judge.py`) then adds faithfulness and answer relevance on
top, and the report keeps the two families separate so that a reader can see
where they disagree. Where they *do* disagree is the most informative cell in
the report, and it is the reason the calibration sheet exists.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from retrieval.text import STOPWORDS_PT, normalise, tokenize

#: Numbers, after `normalise` has welded pypdf-split decimals back together.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

#: Phrasings that count as a refusal. The system prompt mandates one exact
#: sentence, but a 3B model paraphrases it, so this stays a small explicit list
#: rather than an exact-match check — and it is a heuristic, which is why the
#: calibration sheet carries the abstention items for human review too.
_REFUSAL_PATTERNS: tuple[str, ...] = (
    "naoencontreiessainformacao",
    "naoencontreiinformacao",
    "naoencontreidocumentosrelevantes",
    "naohainformacao",
    "naocontemessainformacao",
    "naoepossivelresponder",
    "naoepossiveldeterminar",
    "ostrechosnaocontem",
    "naofoipossivelencontrar",
)


def numbers_in(text: str) -> set[str]:
    """Numeric tokens, normalised so `13.25` and `13,25` are the same number.

    Bare years are kept: `2026` in an answer about forecast horizons is a claim
    like any other, and dropping it would hide a model that swapped 2026 for
    2027.
    """
    return {match.group(0).replace(".", ",") for match in _NUMBER.finditer(normalise(text))}


def numeric_recall(reference: str, generated: str) -> float | None:
    """Share of the reference answer's numbers present in the generated answer.

    `None` when the reference states no numbers — an abstention row, or a
    question whose answer is a list of names. Returning 0.0 there would report a
    failure that was never possible.
    """
    expected = numbers_in(reference)
    if not expected:
        return None
    return len(expected & numbers_in(generated)) / len(expected)


#: Citation markers the prompt asks for: `[2]`, `[2, 19]`, `[1, Tabela 1 e 17]`.
#: Stripped before numbers are extracted, because their digits are *pointers to
#: passages*, not claims about the world. Missing this produced a measured
#: hallucination that was not one: `[2, 19]` normalises to the decimal `2,19`
#: (see `_SPLIT_DECIMAL` in `retrieval/text.py`), which then appears in no
#: passage and is indistinguishable from an invented rate. One row of one arm,
#: enough to move `hallucinated_number_rate` from 0.000 to 0.020 and to put a
#: false claim in the README.
_CITATION = re.compile(r"\[[^\]\n]{0,80}\]")


def strip_citations(text: str) -> str:
    """Remove `[...]` citation markers so their digits are not read as claims."""
    return _CITATION.sub(" ", text)


def unsupported_numbers(generated: str, context: str, question: str = "") -> list[str]:
    """Numbers asserted by the answer that nothing in its evidence supports.

    The question is part of the allowed set because echoing back "na 279a
    reuniao" is quoting the user, not inventing a fact. Citation markers are
    removed from the answer first — see `_CITATION`.
    """
    allowed = numbers_in(context) | numbers_in(question)
    return sorted(numbers_in(strip_citations(generated)) - allowed)


def _content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS_PT]


def lexical_groundedness(generated: str, context: str) -> float | None:
    """Share of the answer's content words that occur in the retrieved context.

    A verbatim quote scores 1.0, which is why the extractive backend is the
    floor this metric is calibrated against rather than a competitor.
    """
    tokens = _content_tokens(generated)
    if not tokens:
        return None
    context_tokens = set(_content_tokens(context))
    return sum(1 for token in tokens if token in context_tokens) / len(tokens)


def abstained(generated: str) -> bool:
    """Whether the answer refused to answer."""
    folded = normalise(generated).replace(" ", "")
    return any(pattern in folded for pattern in _REFUSAL_PATTERNS)


def cited_doc_ids(passages: Iterable) -> list[str]:
    return [passage.doc_id for passage in passages]


@dataclass(frozen=True)
class GenerationScores:
    """Deterministic scores for one generated answer."""

    numeric_recall: float | None
    unsupported_numbers: tuple[str, ...]
    lexical_groundedness: float | None
    abstained: bool
    cited_gold_doc: bool
    n_citations: int

    @property
    def has_unsupported_numbers(self) -> bool:
        return bool(self.unsupported_numbers)

    def to_json(self) -> dict:
        return {
            "numeric_recall": self.numeric_recall,
            "unsupported_numbers": list(self.unsupported_numbers),
            "has_unsupported_numbers": self.has_unsupported_numbers,
            "lexical_groundedness": self.lexical_groundedness,
            "abstained": self.abstained,
            "cited_gold_doc": self.cited_gold_doc,
            "n_citations": self.n_citations,
        }


def score_generation(
    generated: str,
    reference: str,
    context: str,
    question: str,
    retrieved_doc_ids: Sequence[str],
    gold_doc_id: str | None,
) -> GenerationScores:
    return GenerationScores(
        numeric_recall=numeric_recall(reference, generated),
        unsupported_numbers=tuple(unsupported_numbers(generated, context, question)),
        lexical_groundedness=lexical_groundedness(generated, context),
        abstained=abstained(generated),
        cited_gold_doc=gold_doc_id is not None and gold_doc_id in set(retrieved_doc_ids),
        n_citations=len(retrieved_doc_ids),
    )


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept else None


def aggregate_generation(
    scores: Sequence[GenerationScores],
    is_abstention: Sequence[bool],
) -> dict:
    """Macro-average, with abstention split out because it inverts.

    On a negative row, `abstained=True` is correct. On an answerable row it is a
    false refusal. Averaging them together would produce a number that goes up
    for two opposite reasons.
    """
    if len(scores) != len(is_abstention):
        raise ValueError("scores and is_abstention must be the same length")

    answerable = [s for s, neg in zip(scores, is_abstention, strict=True) if not neg]
    negatives = [s for s, neg in zip(scores, is_abstention, strict=True) if neg]

    payload: dict = {
        "n": len(scores),
        "n_answerable": len(answerable),
        "n_negative": len(negatives),
        "numeric_recall": _mean(s.numeric_recall for s in answerable),
        "lexical_groundedness": _mean(s.lexical_groundedness for s in answerable),
        "hallucinated_number_rate": (
            sum(s.has_unsupported_numbers for s in answerable) / len(answerable)
            if answerable
            else None
        ),
        "citation_correctness": (
            sum(s.cited_gold_doc for s in answerable) / len(answerable) if answerable else None
        ),
        "false_refusal_rate": (
            sum(s.abstained for s in answerable) / len(answerable) if answerable else None
        ),
        "abstention_correctness": (
            sum(s.abstained for s in negatives) / len(negatives) if negatives else None
        ),
    }
    return payload
