# Security policy

## Reporting a vulnerability

Please report security issues **privately** rather than opening a public issue.

- Use GitHub's [private vulnerability reporting](https://github.com/rmonteiro-pereira/rag-eval/security/advisories/new)
  — preferred.
- Or email **rmonteiropereira1@gmail.com** with `SECURITY` in the subject.

Include the commit, the command you ran, and what you observed. Expect an
acknowledgement within **7 days**. This is a portfolio and research project
maintained by one person in his own time — that is the honest expectation to set,
rather than an SLA nobody is on call for.

## What this project is, and what that means for its threat model

`rag-eval` is a retrieval-evaluation harness over **public** Banco Central do
Brasil Copom minutes. The corpus is not redistributed — only a manifest of URLs,
titles, dates and SHA-256 digests is committed.

It is a **local, single-user research harness**. Every service it talks to is on
`localhost`: Qdrant, Postgres, Langfuse, Ollama. There is no authentication
layer, no multi-tenancy, and no network exposure by default. It is not hardened
for deployment and it does not claim to be.

Two consequences worth stating plainly, because both look like vulnerabilities and
neither is a secret:

1. **The `/ask` endpoint takes its ACL subject from the request body.** A caller
   picks who they are. This is a demonstration of the enforcement mechanism, not
   an access-control system — an ACL whose subject is chosen by the caller is not
   an ACL. `serving/api.py` says so in the endpoint docstring and repeats it in
   every response. In production the subject comes from an authenticated session.
2. **`docker-compose.yml` contains credentials, deliberately.** They are published
   constants for containers that hold no real data and never leave the machine:
   a Postgres password, a Langfuse init password, a `NEXTAUTH_SECRET`, and a
   64-hex `ENCRYPTION_KEY` that is patterned rather than random precisely so it
   cannot be mistaken for a leak. Each is commented in place. If you point this
   stack at anything you care about, generate real ones (`openssl rand -hex 32`)
   and move them to `.env`, which is gitignored.

Beyond those, **no credentials belong in this repository.** If you find anything
credential-shaped committed, report it privately rather than opening an issue.

## Areas worth reporting

This project is partly *about* adversarial input, so the line between a finding
and a measured result matters:

- **A guardrail bypass that the harness does not detect.** Prompt-injection and
  PII cases that succeed *and* are scored as safe are genuine findings — the suite
  exists to measure exactly this, so a gap in the measurement is worth more than a
  gap in the defence.
- **Document-level ACL leakage** — any path where a filtered retrieval returns a
  restricted document.
- **PII reaching an artifact.** Masking runs before persistence; anything that
  writes unmasked entities to a report, a log or the vector store is a finding.
- **Deserialisation or path traversal** in the ingestion and reporting paths.
- **Dependency vulnerabilities** reachable from the CLI entry points.

## Out of scope

- Injection attempts the suite **already detects and reports** — those are
  measured results, and the measured attack-success rate is published rather than
  hidden.
- The behaviour of third-party model backends run locally.
- Availability of the upstream BACEN site.

## Security-relevant behaviour that *is* implemented and measured

Each control is measured against an ungoverned arm running the identical inputs —
see [`docs/governance.md`](docs/governance.md) and
[`eval/reports/adversarial.json`](eval/reports/adversarial.json).

| Control | Where | Measured |
|---|---|---|
| Prompt-injection detection, direct and indirect | `guardrails/injection.py` | 8.3% attack success governed vs 16.7% ungoverned; **0.0% vs 16.7%** on the indirect surface |
| PII masking, input and output | `guardrails/pii.py`, `guardrails/brazilian.py` | 0.0% output leak vs **100.0%** ungoverned |
| Document-level ACL as a Qdrant pre-filter | `governance/acl.py` | **0** restricted chunks reached an uncleared user, at top-200 |
| Append-only audit log | `governance/audit.py` | 109 events per suite run |
| Read-only SQL with layered validation + HITL gate | `agent/tools.py`, `agent/hitl.py` | 6 of 24 statements refused in the recorded demo |

**The residual is stated rather than rounded away.** Two of twenty-four injection
attacks defeat the stack (`inj-012`, `inj-023`), both named in
`docs/governance.md` with the structural reason the rule missed them. A security
section that opens with 0% is either testing weak attacks or not being straight.

Note also the gap between *detecting* an attack (58.3%) and *stopping* one
(91.7%): eight attacks evaded the detector and failed anyway. Detection is not
defence, and neither number substitutes for the other.

## Data handling

- **The audit log stores the masked query and a SHA-256 of the raw one** — never
  the raw query, the answer, or a matched PII substring. A log that stores those
  is a second copy of exactly what the masker exists to contain, with broader
  read access. `PiiFinding.to_json()` deliberately omits the matched text.
- **No corpus, no weights, no database** is committed. `data/`, `.venv/`,
  `qdrant_storage/`, `*.pdf` and the model-weight extensions are gitignored, and
  `.env` has been ignored since the first commit.
- **No telemetry leaves the machine.** `QDRANT__TELEMETRY_DISABLED` and
  `TELEMETRY_ENABLED: "false"` are set in `docker-compose.yml`; there is no paid
  API and no API key anywhere in the repository.

## Supply chain

- Dependencies are pinned in a committed `uv.lock`. The spaCy Portuguese model is
  pinned by release-wheel URL, because `guardrails/pii.py` **silently degrades to
  a regex backend** without it rather than failing — `docs/REPRODUCE.md` step 2b
  asserts the loaded backend before anything measures with it.
- `.github/workflows/eval.yml` requests `permissions: contents: read`.
- **Dependabot security alerts are currently OFF.** `GET
  /repos/rmonteiro-pereira/rag-eval/vulnerability-alerts` returns 404 and the
  alerts API returns *"Dependabot alerts are disabled for this repository"*. That
  is a repository setting, not something a file here can change. Stated rather
  than left for a reader to discover: nothing is watching this dependency tree
  for published CVEs today. `.github/dependabot.yml` configures *version* updates
  only, which is a different thing.
- Dependency **version** updates are monthly, grouped and capped at 3 open PRs,
  with `torch`, `qdrant-client` and the spaCy model excluded by argument rather
  than by oversight — each is pinned for a reason recorded in
  `.github/dependabot.yml`.
- **Actions are referenced by major-version tag, not pinned to a commit SHA**, and
  that is a weaker guarantee rather than a pin: a tag is mutable, so
  `actions/checkout@v4` can be repointed by its maintainer. The trade is
  deliberate — SHA pinning without Dependabot to move the pins produces actions
  that silently rot, and this repository has one maintainer. Anyone forking it
  into an environment where a compromised action would matter should replace the
  tags with 40-character SHAs.
- The full pre-publication audit of this repository — secrets, entire git
  history, blob sizes, and what was remediated — is
  [`docs/PUBLICATION-SCAN.md`](docs/PUBLICATION-SCAN.md).

## A note on the numbers

Metrics here are computed against a gold set whose pairs are **draft and not
human-validated**; `--min-status validated` deliberately returns nothing today.
That is a stated epistemic limit, not a security issue.
