# Security policy

## Reporting a vulnerability

Open a [private security advisory](https://github.com/rmonteiro-pereira/rag-eval/security/advisories/new).
Please do not open a public issue for anything exploitable.

I will acknowledge within **7 days**. This is a portfolio and research project
maintained by one person in his own time — that is the honest expectation to set,
rather than an SLA nobody is on call for.

## What this project is, and what that means for its threat model

`rag-eval` is a **local, single-user research harness**. Every service it talks to
is on `localhost`: Qdrant, Postgres, Langfuse, Ollama. There is no
authentication layer, no multi-tenancy, and no network exposure by default. It is
not hardened for deployment and it does not claim to be.

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

## Security-relevant behaviour that *is* implemented and measured

These are controls, and each is measured against an ungoverned arm running the
identical inputs — see [`docs/governance.md`](docs/governance.md) and
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
- `.github/workflows/eval.yml` requests `permissions: contents: read` and pins
  actions to major-version tags.
- The full pre-publication audit of this repository — secrets, entire git
  history, blob sizes, and what was remediated — is
  [`docs/PUBLICATION-SCAN.md`](docs/PUBLICATION-SCAN.md).
