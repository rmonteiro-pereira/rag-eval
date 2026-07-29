"""Portuguese text normalisation shared by BM25 and the metadata parser.

Two corpus-specific quirks drive everything here:

1. **`pypdf` splits numbers.** Extraction of these PDFs yields `0, 25` and
   `14 , 25` where the page reads `0,25` and `14,25`. Since half the gold set is
   numeric extraction ("para que nivel a taxa foi levada?"), a tokenizer that
   turns `14,25` into `14` and `25` throws away the single most discriminative
   token in the query. So a digit pair split by a decimal mark and stray spaces
   is welded back together *before* tokenising.

   The weld is deliberately narrow — at most two digits on each side, with no
   digit adjoining on either flank. That is what keeps `expectativas para 2022,
   2023 e 2024` from collapsing into a single nonsense token: `22` is preceded
   by a digit and `20` is followed by one, so neither side qualifies.

2. **Queries are unaccented, documents are not.** The gold file stores ASCII
   (`marco de 2025`) while the PDFs say `março`. Folding accents on both sides
   is what makes them meet.

Deliberately no stemming. Portuguese stemming needs a real stemmer (RSLP) and
this corpus is small and formulaic enough that the win would be noise; adding
one is a future ablation arm, not a silent default.
"""

from __future__ import annotations

import re
import unicodedata

_SPLIT_DECIMAL = re.compile(r"(?<!\d)(\d{1,2})\s*([.,])\s*(\d{1,2})(?!\d)")

#: Words plus numbers-with-decimals. `14,25` survives as one token; `%` and
#: punctuation are dropped.
_TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")

#: Function words that appear in nearly every question and nearly every ata.
#: BM25's IDF already discounts them, but dropping them keeps the sparse
#: candidate list from being ranked by query length rather than query content.
#: Kept short and boring on purpose.
STOPWORDS_PT: frozenset[str] = frozenset(
    """
    a as ao aos o os um uma uns umas de do da dos das em no na nos nas
    e ou que qual quais quando quanto quanta quantos quantas como onde
    por para pelo pela pelos pelas com sem sobre entre ate apos
    foi foram era eram ser sao esta estao seu sua seus suas
    se ja nao mais menos muito pouco entao
    """.split()
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise(text: str) -> str:
    """Lowercase, unaccent, and repair PDF-split decimals."""
    folded = strip_accents(text).lower()
    return _SPLIT_DECIMAL.sub(r"\1\2\3", folded)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Tokens for BM25. Numeric tokens keep their decimal mark, comma-style."""
    tokens: list[str] = []
    for raw in _TOKEN.findall(normalise(text)):
        # `14.25` and `14,25` are the same number written two ways; collapse to
        # the Portuguese form so a query and a document can match on it.
        token = raw.replace(".", ",") if raw[0].isdigit() else raw
        if drop_stopwords and token in STOPWORDS_PT:
            continue
        tokens.append(token)
    return tokens
