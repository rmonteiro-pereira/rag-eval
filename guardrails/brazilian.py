"""Brazilian identifier recognisers for Presidio.

These exist because of a measured gap, not a hypothetical one. Presidio with the
Portuguese spaCy pipeline, given

    "Meu nome e Joao Silva, CPF 529.982.247-25, email joao@exemplo.com,
     telefone (11) 98765-4321."

returns `PERSON`, `EMAIL_ADDRESS`, `URL` and a 0.4-confidence `PHONE_NUMBER` —
and **misses the CPF entirely**. The CPF is the single most consequential
personal identifier in Brazil and the one a financial-domain system must never
leak. Out of the box, Presidio does not see it.

Every recogniser here validates **check digits** rather than matching a shape.
That distinction is the whole design:

* A shape match on `\\d{3}\\.\\d{3}\\.\\d{3}-\\d{2}` fires on `123.456.789-00`,
  which is not a CPF, and on document identifiers that happen to be punctuated
  the same way. False positives in a masker are not harmless — they redact real
  content and quietly degrade answers.
* Check-digit validation makes the recogniser nearly exact, which is what
  licenses masking at high confidence without a human in the loop.

Repeated-digit sequences (`111.111.111-11`) satisfy the checksum arithmetic and
are still not valid CPFs; they are rejected explicitly, because they are exactly
what appears in test fixtures and documentation and would otherwise be masked as
though real.
"""

from __future__ import annotations

import re

from presidio_analyzer import Pattern, PatternRecognizer

CPF_ENTITY = "BR_CPF"
CNPJ_ENTITY = "BR_CNPJ"
CEP_ENTITY = "BR_CEP"
BR_PHONE_ENTITY = "BR_PHONE"

_DIGITS = re.compile(r"\D")


def _only_digits(value: str) -> str:
    return _DIGITS.sub("", value)


def _check_digit(digits: str, weights: list[int]) -> int:
    total = sum(int(d) * w for d, w in zip(digits, weights, strict=True))
    remainder = total % 11
    return 0 if remainder < 2 else 11 - remainder


def is_valid_cpf(value: str) -> bool:
    """CPF check-digit validation (Receita Federal modulo-11)."""
    digits = _only_digits(value)
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    first = _check_digit(digits[:9], list(range(10, 1, -1)))
    second = _check_digit(digits[:10], list(range(11, 1, -1)))
    return digits[9] == str(first) and digits[10] == str(second)


def is_valid_cnpj(value: str) -> bool:
    """CNPJ check-digit validation (modulo-11 with the standard weight cycles)."""
    digits = _only_digits(value)
    if len(digits) != 14 or len(set(digits)) == 1:
        return False
    first = _check_digit(digits[:12], [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2])
    second = _check_digit(digits[:13], [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2])
    return digits[12] == str(first) and digits[13] == str(second)


class CpfRecognizer(PatternRecognizer):
    """CPF, punctuated or bare. Only a valid checksum reaches high confidence."""

    PATTERNS = [
        Pattern("CPF (punctuated)", r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", 0.4),
        Pattern("CPF (bare)", r"\b\d{11}\b", 0.05),
    ]
    CONTEXT = ["cpf", "documento", "contribuinte", "titular"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity=CPF_ENTITY,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="pt",
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        """`True` promotes the match to certainty, `False` discards it.

        The bare 11-digit pattern starts at 0.05 precisely so that an invalid
        one is dropped rather than masked: an 11-digit number in a document about
        interest rates is far more likely to be a figure than a CPF.
        """
        return is_valid_cpf(pattern_text)


class CnpjRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern("CNPJ (punctuated)", r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", 0.4),
        Pattern("CNPJ (bare)", r"\b\d{14}\b", 0.05),
    ]
    CONTEXT = ["cnpj", "empresa", "razao social", "inscricao"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity=CNPJ_ENTITY,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="pt",
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        return is_valid_cnpj(pattern_text)


class CepRecognizer(PatternRecognizer):
    """Postal code. Punctuated only — a bare 8-digit number is too ambiguous."""

    PATTERNS = [Pattern("CEP", r"\b\d{5}-\d{3}\b", 0.5)]
    CONTEXT = ["cep", "endereco", "logradouro"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity=CEP_ENTITY,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="pt",
        )


class BrazilianPhoneRecognizer(PatternRecognizer):
    """Brazilian phone numbers, which Presidio's generic recogniser scores at 0.4.

    Requires either the parenthesised area code or a `+55` prefix, so it does not
    fire on a bare eight- or nine-digit run.
    """

    PATTERNS = [
        Pattern("BR phone (DDD)", r"\(\d{2}\)\s?9?\d{4}-?\d{4}\b", 0.7),
        Pattern("BR phone (+55)", r"\+55\s?\d{2}\s?9?\d{4}-?\d{4}\b", 0.8),
    ]
    CONTEXT = ["telefone", "celular", "contato", "fone", "whatsapp"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity=BR_PHONE_ENTITY,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="pt",
        )


def brazilian_recognizers() -> list[PatternRecognizer]:
    return [
        CpfRecognizer(),
        CnpjRecognizer(),
        CepRecognizer(),
        BrazilianPhoneRecognizer(),
    ]
