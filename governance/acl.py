"""Document-level access control, enforced inside the vector search.

## The one design decision that matters

The ACL compiles to a **Qdrant payload filter that is part of the query**, not a
post-filter over results. That is the whole point, and it is where most RAG
access control goes wrong:

* A post-filter has already read the restricted document, ranked it, and put it
  in the process's memory. Whether it is then shown to the user is a rendering
  decision.
* A post-filter silently shortens the result list. Ask for `top_k=5`, get 2, and
  the *length* of the response leaks how many restricted documents matched — a
  classic side channel. A pre-filter returns 5 documents the user may actually
  see.
* A post-filter is one forgotten `if` away from leaking. A pre-filter cannot leak
  what the database never returned.

`tests/test_acl.py` asserts the first property directly against a live Qdrant: a
user without clearance retrieves **zero** chunks from restricted documents, even
when the query is written specifically to target one.

## The classification is synthetic, and that is stated everywhere

These are public BACEN documents. Nothing in this corpus is actually restricted,
so a real ACL cannot be demonstrated on it without inventing one.

The invention is chosen to be *plausible* rather than arbitrary: Copom minutes
are published on a delay, so the **most recent N meetings** are labelled
`restricted`, standing in for "released internally, not yet public". Every report
that uses this carries `synthetic: true`. It demonstrates the mechanism; it does
not describe a real classification of BACEN data.

Classifications are written to the existing collection with `set_payload`, which
does not touch vectors — so applying, changing or removing them costs no
re-embedding.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag.config import settings

PUBLIC = "public"
RESTRICTED = "restricted"
CLASSIFICATIONS: tuple[str, ...] = (PUBLIC, RESTRICTED)

PAYLOAD_FIELD = "classification"

#: How many of the most recent meetings are treated as embargoed.
DEFAULT_RESTRICTED_COUNT = 5


@dataclass(frozen=True)
class User:
    """Who is asking, and what they may see."""

    user_id: str
    clearances: frozenset[str] = field(default=frozenset({PUBLIC}))

    @property
    def visible(self) -> frozenset[str]:
        return self.clearances

    def may_see(self, classification: str) -> bool:
        return classification in self.clearances

    def to_json(self) -> dict:
        return {"user_id": self.user_id, "clearances": sorted(self.clearances)}


#: The two roles the adversarial suite exercises.
ANALYST = User(user_id="analyst", clearances=frozenset({PUBLIC}))
SUPERVISOR = User(user_id="supervisor", clearances=frozenset({PUBLIC, RESTRICTED}))


def classify_documents(
    documents: Sequence,
    restricted_count: int = DEFAULT_RESTRICTED_COUNT,
) -> dict[str, str]:
    """Assign a classification per `doc_id`. Deterministic — newest are restricted.

    `documents` needs `doc_id` and `reference_date`. Sorting by reference date
    (descending, `doc_id` breaking ties) makes the assignment stable across runs
    and independent of the order Qdrant happens to scroll in.
    """
    ordered = sorted(documents, key=lambda d: (d.reference_date, d.doc_id), reverse=True)
    restricted = {doc.doc_id for doc in ordered[:restricted_count]}
    return {doc.doc_id: (RESTRICTED if doc.doc_id in restricted else PUBLIC) for doc in ordered}


def access_filter(user: User) -> qmodels.Filter:
    """The payload filter expressing what this user may retrieve.

    Always returns a filter, never `None`. An ACL whose "allow everything" case
    is expressed by returning no filter is one refactor away from a leak — the
    call site would have to remember that `None` means unrestricted rather than
    unauthorised.
    """
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=PAYLOAD_FIELD,
                match=qmodels.MatchAny(any=sorted(user.visible)),
            )
        ]
    )


def combine(*filters: qmodels.Filter | None) -> qmodels.Filter | None:
    """AND several filters together, ignoring `None`s.

    The ACL and the M4 meeting filter are both payload filters and must both
    apply; whichever narrows more, neither may be dropped.
    """
    conditions: list = []
    for f in filters:
        if f is not None and f.must:
            conditions.extend(f.must)
    return qmodels.Filter(must=conditions) if conditions else None


def ensure_payload_index(client: QdrantClient, collection: str | None = None) -> None:
    collection = collection or settings.qdrant_collection
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=PAYLOAD_FIELD,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
    except Exception:  # noqa: BLE001 - already indexed is the common, fine case
        pass


def apply_classifications(
    client: QdrantClient,
    assignments: dict[str, str],
    collection: str | None = None,
) -> dict[str, int]:
    """Write classifications onto the existing points. Vectors are untouched.

    Idempotent: re-running with the same assignments is a no-op in effect, so it
    is safe to call at the start of every governed run.
    """
    collection = collection or settings.qdrant_collection
    ensure_payload_index(client, collection)

    counts: dict[str, int] = {}
    for classification in CLASSIFICATIONS:
        doc_ids = sorted(doc for doc, value in assignments.items() if value == classification)
        if not doc_ids:
            continue
        client.set_payload(
            collection_name=collection,
            payload={PAYLOAD_FIELD: classification},
            points=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id", match=qmodels.MatchAny(any=doc_ids)
                    )
                ]
            ),
            wait=True,
        )
        counts[classification] = len(doc_ids)
    return counts


def restricted_doc_ids(assignments: dict[str, str]) -> set[str]:
    return {doc for doc, value in assignments.items() if value == RESTRICTED}


def leaked(hits: Iterable, assignments: dict[str, str], user: User) -> list[str]:
    """Retrieved chunks this user was not cleared to see. Must always be empty."""
    return sorted(
        {
            hit.doc_id
            for hit in hits
            if not user.may_see(assignments.get(hit.doc_id, PUBLIC))
        }
    )
