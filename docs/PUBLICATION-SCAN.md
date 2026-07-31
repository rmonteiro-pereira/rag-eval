# Publication safety scan

Run before this repository was made public. Every section below is the output of
a command, reproduced verbatim, with the command shown so you can re-run it.

- **Scanned:** the full working tree **and all 18 commits of history**, on `main` at
  `7c11937`, before the remediation commit described in §7.
- **Tooling:** `gitleaks` (dev build), plus hand-written pattern sweeps for the
  things gitleaks does not model — internal hostnames, RFC-1918 and Tailscale
  addresses, absolute local paths, and personal identifiers.
- **Redaction policy:** no live secret value is printed in this file. Where a
  value had to be identified it is named by file and line only, and quoted only
  after being replaced with a non-secret.

**Verdict is at the bottom (§8).**

---

## 1. Secrets — automated scan

```console
$ gitleaks git --no-banner --redact --exit-code 0 .
INF 18 commits scanned.
INF scanned ~4904820 bytes (4.90 MB) in 992ms
WRN leaks found: 1

Finding:     ENCRYPTION_KEY: "REDACTED"
RuleID:      generic-api-key
Entropy:     3.982913
File:        docker-compose.yml
Line:        67
Commit:      0af910e603786fe3535bbf837e5ca0c7c6280af9
Fingerprint: 0af910e603786fe3535bbf837e5ca0c7c6280af9:docker-compose.yml:generic-api-key:67
```

Working tree, including untracked files:

```console
$ gitleaks dir --no-banner --redact --exit-code 0 -f json -r gl-scan.json .
total findings in worktree : 77
inside .venv/ (gitignored) : 76
OUTSIDE .venv/             : 1
    docker-compose.yml line 69 | generic-api-key | entropy 3.98
```

The 76 `.venv/` hits are constants inside `pywin32` (`CMSG_KEY_AGREE_VERSION` and
friends) in a directory that is gitignored and has never been committed — see §4.
**One** finding is in a publishable file. It is `F-01` in §6.

## 2. Secrets — manual pattern sweeps

Automated scanners miss the things that are specific to a person's machine. These
sweeps cover them.

### 2.1 Secret-shaped tokens, tracked tree

```console
$ git grep -nIE 'sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{30,}|BEGIN [A-Z ]*PRIVATE KEY|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.' -- .
  exit=1 (no match)
```

### 2.2 The same sweep over **every blob in every commit**

Not `git log -p` — that only shows diffs of text files git chose to diff. This
walks every object in the object database, so a file added and removed in the same
branch is still read.

```console
$ git rev-list --objects --all \
    | git cat-file --batch-check='%(objecttype) %(objectname) %(rest)' \
    | awk '$1=="blob" {print $2, $3}' \
    | while read -r sha path; do
        git cat-file blob "$sha" \
          | grep -nIE 'sk-[A-Za-z0-9]{16,}|ghp_|github_pat_|xox[baprs]-|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{30,}|BEGIN [A-Z ]*PRIVATE KEY|vanir|\.dev\.br|192\.168\.|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.|\.ts\.net|minioadmin' \
          | sed "s|^|$path:|"
      done | sort -u
  (no output)
```

**Nothing.** No key, no internal hostname, no `vanir.dev.br`, no RFC-1918 address,
no Tailscale `100.64/10` address, no `.ts.net` name, no MinIO credential — in any
version of any file this repository has ever contained.

### 2.3 Internal hostnames and private IPs, tracked tree

```console
$ git grep -nIE 'vanir|\.dev\.br|192\.168\.[0-9]+\.[0-9]+|10\.[0-9]+\.[0-9]+\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+|\.ts\.net' -- .
uv.lock:831:version = "10.4.0.35"
uv.lock:834:    { url = ".../nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_aarch64.whl", ... }
uv.lock:835:    { url = ".../nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl", ... }
```

Three matches, all the same false positive: the `10.x.x.x` branch of the RFC-1918
pattern matching the *version number* of `nvidia-curand` in the lockfile. Not an
address. **No true positive.**

### 2.4 Absolute local paths and machine names

A leaked `C:\Users\...` path tells a reader the account name and the directory
layout of the author's machine, and is the most common thing left behind in a
portfolio repo.

```console
$ git grep -nIE '([A-Za-z]:\\|[A-Za-z]:/)[A-Za-z_]|/home/[a-z]+/|/Users/[A-Za-z]+/' -- . ':!uv.lock'
  exit=1 (no match)

$ rg '([A-Za-z]:\\|[A-Za-z]:/)[A-Za-z_]|/home/[a-z]+/|/Users/[A-Za-z]+/|DESKTOP-|LAPTOP-' eval/reports/
No matches found
```

Clean, including inside the 3.4 MB of committed JSON reports. Every path in the
repository is relative to the repository root. The one sibling-project dependency,
the OFL lakehouse DuckDB used by agent mode, is referenced through a configurable
setting and documented as an external artifact — `agent/tools.py:21`.

### 2.5 `.env`

```console
$ git ls-files --error-unmatch .env
  .env is NOT tracked
$ test -f .env
  no .env on disk
```

`.env` is gitignored (`.gitignore:2`) and does not exist. `.env.example` is
committed and contains only local defaults — `localhost` URLs, a HuggingFace model
name, chunk sizes, and the two auto-provisioned Langfuse demo keys discussed in §6.

### 2.6 Commit messages

```console
$ git log --all --format='%H%n%B' | grep -nIE 'sk-[A-Za-z0-9]{16,}|ghp_|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|password[= ]|senha[= ]|vanir|192\.168\.'
  exit=1 (no match)
```

## 3. Size

### 3.1 What a reviewer downloads

```console
$ git count-objects -vH
in-pack: 231
size-pack: 772.18 KiB
```

**772 KiB.** A `git clone` of this repository is under a megabyte.

### 3.2 Largest blobs in the whole history

```console
$ git rev-list --objects --all \
    | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
    | awk '$1=="blob"' | sort -k3 -n -r | head -10
      1.70 MB  eval/reports/ablation.json
      1.60 MB  eval/reports/generation.json
      0.29 MB  eval/datasets/judge_calibration_sheet.jsonl
      0.27 MB  uv.lock
      0.24 MB  uv.lock
      0.23 MB  uv.lock
      0.19 MB  eval/reports/baseline_dense.json
      0.17 MB  uv.lock
      0.17 MB  uv.lock
      0.15 MB  eval/reports/baseline_dense.json
```

**No blob in history exceeds 5 MB.** The largest ever committed is 1.70 MB. The
repeated `uv.lock` entries are successive versions of the lockfile, which is
expected and is why they are listed separately.

### 3.3 Worktree

```console
$ git ls-files -z | xargs -0 ls -l | awk '$5 > 1048576 {printf "%10.2f MB  %s\n", $5/1048576, $9}'
      1.75 MB  eval/reports/ablation.json
      1.61 MB  eval/reports/generation.json

$ git ls-files -z | xargs -0 ls -l | awk '{s+=$5} END {printf "%.2f MB across %d files\n", s/1048576, NR}'
4.73 MB across 94 files
```

Two tracked files exceed 1 MB; neither approaches 5 MB. Both are evaluation
reports committed **on purpose** and with a stated reason — they carry the
per-query audit trail behind every number in `docs/`, and `.gitignore` allowlists
exactly four of them by name rather than ignoring the directory. A number you
cannot drill into is a number the reader has to take on faith.

## 4. Junk

```console
$ for d in .venv .pytest_cache .ruff_cache data/raw data/audit qdrant_storage langfuse_data models .docker-data; do ... git check-ignore -q "$d" ...; done
.venv                1.2G  [ignored OK]
.pytest_cache        31K   [ignored OK]
.ruff_cache          23K   [ignored OK]
data/raw             9.5M  [ignored OK]
data/audit           208K  [ignored OK]
```

Everything present on disk that must not ship is ignored, and the rest does not
exist. Checked and clean:

| Class | Status |
|---|---|
| `.venv/` (1.2 GB) | ignored, never committed |
| `__pycache__/` (14 dirs) | ignored |
| Raw BACEN corpus, `data/raw/` — 30 PDFs, 9.5 MB | ignored; only `data/manifest.json` ships |
| `data/audit/` — JSONL audit log, 208 KB | ignored |
| Model weights (`*.safetensors`, `*.gguf`, `*.bin`, `*.onnx`) | ignored by pattern; none on disk |
| Qdrant / Langfuse state | Docker named volumes, outside the tree |
| `mlruns/`, `*.db`, `*.duckdb` | none exist anywhere under the repo |
| Notebooks | **none** — this repo contains no `.ipynb`, so no notebook outputs to strip |

```console
$ find . -path ./.venv -prune -o -path ./.git -prune -o \
    \( -name '*.ipynb' -o -name '*.db' -o -name '*.duckdb' -o -name '*.safetensors' \
       -o -name '*.gguf' -o -name '*.bin' \) -print
  (nothing)
```

## 5. What history actually contains

```console
$ git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(rest)' \
    | awk '$1=="blob" && NF>1 {$1=""; print substr($0,2)}' | sort -u | wc -l
95
$ comm -23 <all historical paths> <paths tracked today>
retrieval/dense.py
```

95 distinct paths have ever existed in this repository; 94 are tracked today. The
single removal is `retrieval/dense.py`, a source file superseded by
`retrieval/configs.py` in M4. **No file was ever committed and then deleted to hide
it** — the usual way a secret survives in history.

## 6. Findings

Every finding carries a file and a line.

### F-01 — `docker-compose.yml:69` — high-entropy Langfuse `ENCRYPTION_KEY` — **LOW, fixed forward**

A 64-character random hex string was committed as the `ENCRYPTION_KEY` for the
self-hosted Langfuse container. This is the only automated finding in a
publishable file.

*Why it is LOW and not HIGH:* it is not a credential to any third-party service.
Langfuse v2 uses it to encrypt rows in a Postgres that runs in a container on
`localhost`, holds no real user data, and is created empty by `docker compose up`.
Possessing it grants access to nothing that a reader cannot create themselves in
thirty seconds.

*Why it was still fixed:* a 64-hex random string in a public repo is
indistinguishable from a real leak at a glance, and "it's fine, trust me" is the
wrong default. Replaced with a **published constant** —
`0123456789abcdef` repeated four times — carrying a comment that says exactly what
it is and how to generate a real one (`openssl rand -hex 32`) if the stack is ever
pointed at data that matters. A patterned value cannot be mistaken for a secret,
and it keeps the quickstart a single command.

*Residual:* the original random value **remains in history** at commit
`0af910e`. Removing it requires rewriting history, which is **not** this scan's
call — see §7. The recommendation is to leave it: it protects nothing.

### F-02 — `eval/datasets/adversarial.jsonl:21` — fabricated identity at a real government domain — **LOW, fixed**

The indirect-injection fixture `inj-020` used `joao.silva@bcb.gov.br` as the PII
canary an attacker tries to exfiltrate. The CPF beside it (`529.982.247-25`) is
the canonical checksum-valid test value and the name is the Brazilian equivalent
of "John Doe" — both are unambiguously synthetic. The **domain**, however, is the
Banco Central do Brasil's real domain, so the string is a plausible-looking
contact record for a person who might exist.

Nothing in the pipeline ever sent that address anywhere, and the file's purpose
makes the context obvious. It is still not a thing to publish. Changed to
`joao.silva@exemplo.com.br`, matching the reserved-example domain the sibling
fixture `pii-002` already used, and the row's `notes` now state that every
identifier in it is synthetic.

Because the string is an attack **canary**, changing it changes what the attack
tries to steal, so `eval/reports/adversarial.json` was **re-measured, not edited**.
The re-run is recorded in §7.

### F-03 — `docker-compose.yml:40,65,66,82`, `.env.example:27`, `README.md:105` — local demo credentials — **INFORMATIONAL, kept**

`POSTGRES_PASSWORD: langfuse`, `NEXTAUTH_SECRET: local-dev-nextauth-secret-change-me`,
`SALT: local-dev-salt-change-me`, `LANGFUSE_INIT_USER_PASSWORD: ragevallocal123`,
and the auto-provisioned Langfuse keys `pk-lf-rag-eval-local` / `sk-lf-rag-eval-local`.

These are **kept deliberately**. Each is self-describing, scoped to a container on
`localhost`, and reproduced identically by anyone who runs `docker compose up`.
Moving them to a `.env` the reader must populate would add a setup step and
protect nothing. The compose file says so at line 63; the README prints the demo
login at line 105 so the quickstart works.

### F-04 — commit metadata — two author identities — **INFORMATIONAL, no action**

```console
$ git log --all --format='%an <%ae>' | sort -u
Rodrigo Monteiro Pereira <rodrigomonteiropereira1@gmail.com>
rmonteiro-pereira <rmonteiropereira1@gmail.com>
```

Two spellings of the author's own name and two of his own addresses (note
`rodrigomonteiropereira1@` vs `rmonteiropereira1@`). Deliberate publication of
his own name on his own portfolio — not a leak. Flagged only because a reviewer
reading `git log` will see split authorship on a single-author repo. Fixing it
means rewriting history; not worth it. Both are Rodrigo's; no third party's
address appears in any commit.

Commits also carry `Co-Authored-By: Claude Opus 5` trailers. **Keep them.** The
repository was built with an AI agent and says so; removing the trailers before
showing it to an employer would be the dishonest choice.

### F-05 — no `LICENSE` file — **fixed in PUB1-3**

An unlicensed public repo is legally "all rights reserved", which is the opposite
of what a portfolio wants. MIT `LICENSE` added.

### Not findings, checked anyway

- **The Copom minutes are public.** BACEN publishes them; `data/manifest.json`
  carries only URLs to `bcb.gov.br`. The corpus itself is gitignored regardless.
- **The synthetic CPFs and CNPJs** in `guardrails/`, `tests/` and the adversarial
  fixtures are checksum-valid test values chosen precisely so the modulo-11
  validators can be tested. They belong to nobody.
- **The document ACL classification is synthetic** and labelled as such in
  `governance/acl.py` and in the UI — the atas are public documents; the five
  newest stand in for an embargo.
- **`data/audit/*.jsonl`** stores SHA-256 query fingerprints rather than query
  text, and is gitignored either way.

## 7. Remediation applied

Two changes, both forward-only. **No history was rewritten.**

| # | Change | File |
|---|---|---|
| 1 | Random `ENCRYPTION_KEY` → published patterned constant + explanatory comment | `docker-compose.yml:63-69` |
| 2 | `joao.silva@bcb.gov.br` → `joao.silva@exemplo.com.br`, `notes` extended | `eval/datasets/adversarial.jsonl:21` |

Change 2 alters an attack canary, so the adversarial report was regenerated
rather than hand-edited:

```console
$ docker compose ps
SERVICE    STATUS
langfuse   Up (healthy)
postgres   Up (healthy)
qdrant     Up (healthy)

$ uv run python -m eval.run_eval --suite adversarial --out eval/reports/adversarial.json
```

The re-measured numbers are in `eval/reports/adversarial.json` and are what the
README and `docs/writeup.md` quote. See §9.

### Escalated to Rodrigo — not an agent's decision

**The `ENCRYPTION_KEY` from F-01 stays in git history** at `0af910e`, and this scan
did not remove it. History rewriting on a repository about to be published is the
author's call, not an automation's.

The recommendation is **do nothing**: the key encrypts a throwaway local
container's own rows, granting an attacker nothing, and `git filter-repo` would
rewrite all 18 commit hashes to remove a value that protects nothing. If you
disagree, rewrite *before* the first push — after publication the old objects can
be fetched from GitHub's reflog for a long time, and the rewrite stops being
effective. Also logged in `_openwiki/program/blockers.md`.

## 8. Verdict

```
SAFE TO PUBLISH: yes
```

Conditional on nothing. The two remediations in §7 are already applied and
committed. Specifically:

- **No credential to any third-party service** exists in the tree or in any of the
  18 commits — no API key, no token, no private key, no connection string to
  anything off this machine.
- **No internal hostname, RFC-1918 address, Tailscale address or MinIO credential**
  appears anywhere in history.
- **No absolute local path** leaks the author's machine layout.
- **No blob over 5 MB**, ever; a full clone is **772 KiB**.
- **No corpus, no weights, no virtualenv, no notebook, no database** is tracked or
  reachable — all ignored, verified with `git check-ignore`.
- The one automated finding and the one manual finding are **fixed**, and the one
  residual is **local-only, documented, and escalated** rather than quietly left.

## 9. Reproducing this scan

```bash
gitleaks git  --no-banner --redact --exit-code 0 .
gitleaks dir  --no-banner --redact --exit-code 0 .   # 76 of 77 hits are in .venv/
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob"' | sort -k3 -n -r | head
git count-objects -vH
git ls-files -z | xargs -0 ls -l | awk '$5 > 5242880'
docker compose up -d && uv run python -m eval.run_eval --suite adversarial
```
