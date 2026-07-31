# 6. The ACL is a query pre-filter, never a post-filter

**Status:** Accepted · M5

## Context

Documents carry a `classification` payload (`public` / `restricted`) and users
carry a clearance set. An uncleared user must not receive restricted content.

There are two places to enforce that: inside the vector query, or on the result
list after it comes back.

## Decision

Compile clearance to a `qmodels.Filter` and pass it as `query_filter` on every
search. `access_filter()` **always returns a filter, never `None`** — there is no
code path where an empty clearance silently means "no restriction".

## Alternative rejected

**Post-filtering: retrieve top-k, drop restricted hits.** It is easier, it needs
no payload index, and it is what most tutorials do. Rejected for three reasons,
in increasing order of seriousness:

1. **It changes k.** Ask for 5, get 5, drop 2, return 3 — the uncleared user
   silently gets a worse answer, and nothing says why.
2. **The process has already read the document.** The text is in memory, in logs,
   in a trace, in an exception. "We filtered it out afterwards" is not a defence
   in a review.
3. **It leaks through the result count.** An uncleared user who asks about an
   embargoed meeting and gets 3 results instead of 5 has learned that two
   restricted documents match — a side channel that returns real information
   about content they cannot see.

## Consequences

- Requires a keyword payload index on `classification`, created at ingest.
- Proven against **live Qdrant** in `tests/test_acl.py`, marked `integration`,
  under three escalating checks: a broad query at top-200, a query aimed by name
  at a restricted meeting, and a raw search bypassing the pipeline. Measured
  result: **0** restricted chunks reached an uncleared user.
- The classification itself is **synthetic** — every ata is public; the five most
  recent stand in for a publication embargo. Labelled as such in the report, the
  UI and `NOTICE`. The enforcement is real; the policy is a fixture.
- The `/ask` endpoint takes its subject from the request body **for the demo
  only**, and says so in the docstring, in every response, and in `SECURITY.md`.
  An ACL whose subject is chosen by the caller is not an ACL.

## Reverses if

Never for correctness. If a store without server-side filtering were adopted, the
honest move is to state that enforcement moved to the client — not to post-filter
and keep the same claim.
