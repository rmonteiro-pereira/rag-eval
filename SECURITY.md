# Security Policy

## Reporting a vulnerability

Please report security issues **privately** rather than opening a public issue.

- Use GitHub's [private vulnerability reporting](https://github.com/rmonteiro-pereira/rag-eval/security/advisories/new) — preferred.
- Or email **rmonteiropereira1@gmail.com** with `SECURITY` in the subject.

Include the commit, the command you ran, and what you observed. Expect an acknowledgement
within **7 days**; this is a personal project, so treat that as best effort.

## What this project handles

A retrieval-evaluation harness over **public** Banco Central do Brasil Copom minutes. The
corpus is not redistributed — only a manifest of URLs, titles, dates and SHA-256 digests is
committed.

**No credentials belong in this repository.** Qdrant and any model backend are configured by
environment variable. If you find anything credential-shaped committed, report it privately
rather than opening an issue.

## Areas worth reporting

This project is partly *about* adversarial input, so the line between a finding and a
measured result matters:

- **A guardrail bypass that the harness does not detect.** Prompt-injection and PII cases
  that succeed *and* are scored as safe are genuine findings — the suite exists to measure
  exactly this, so a gap in the measurement is worth more than a gap in the defence.
- **Document-level ACL leakage** — any path where a filtered retrieval returns a restricted
  document.
- **PII reaching an artifact.** Masking runs before persistence; anything that writes
  unmasked entities to a report, a log or the vector store is a finding.
- **Deserialisation or path traversal** in the ingestion and reporting paths.
- **Dependency vulnerabilities** reachable from the CLI entry points.

## Out of scope

- Injection attempts that the suite **already detects and reports** — those are measured
  results, and the measured attack-success rate is published rather than hidden.
- The behaviour of third-party model backends run locally.
- Availability of the upstream BACEN site.

## A note on the numbers

Metrics here are computed against a gold set whose pairs are **draft and not human-validated**;
`--min-status validated` deliberately returns nothing today. That is a stated epistemic limit,
not a security issue.
