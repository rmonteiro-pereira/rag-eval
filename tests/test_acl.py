"""Document-level access control.

The claim being tested is narrow and absolute: **a user without clearance
retrieves zero chunks from restricted documents.** Not "few", not "they are
filtered from the display" — zero, because the filter is inside the query and the
database never returns them.

Two layers of test, and both are needed:

* Unit tests on filter construction, which run anywhere and pin the semantics.
* An integration test against a live Qdrant, which is the only thing that proves
  the filter is actually applied by the server rather than merely constructed.
  It skips when Qdrant is unreachable — a skip is honest, a fake pass is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from qdrant_client.http import models as qmodels

from governance.acl import (
    ANALYST,
    PUBLIC,
    RESTRICTED,
    SUPERVISOR,
    User,
    access_filter,
    classify_documents,
    combine,
    leaked,
    restricted_doc_ids,
)


@dataclass
class Doc:
    doc_id: str
    reference_date: str


@dataclass
class Hit:
    doc_id: str


DOCS = [
    Doc("ata-2022-10", "2022-10-26"),
    Doc("ata-2024-05", "2024-05-08"),
    Doc("ata-2025-12", "2025-12-10"),
    Doc("ata-2026-06", "2026-06-17"),
]


def test_the_newest_meetings_are_the_restricted_ones():
    assignments = classify_documents(DOCS, restricted_count=2)
    assert assignments["ata-2026-06"] == RESTRICTED
    assert assignments["ata-2025-12"] == RESTRICTED
    assert assignments["ata-2024-05"] == PUBLIC
    assert assignments["ata-2022-10"] == PUBLIC


def test_classification_is_deterministic_and_order_independent():
    """Qdrant scrolls in an arbitrary order; the ACL must not depend on it."""
    forwards = classify_documents(DOCS, restricted_count=2)
    backwards = classify_documents(list(reversed(DOCS)), restricted_count=2)
    assert forwards == backwards


def test_restricted_count_zero_restricts_nothing():
    assert restricted_doc_ids(classify_documents(DOCS, restricted_count=0)) == set()


def test_an_analyst_filter_admits_only_public():
    condition = access_filter(ANALYST).must[0]
    assert condition.key == "classification"
    assert condition.match.any == [PUBLIC]


def test_a_supervisor_filter_admits_both():
    assert access_filter(SUPERVISOR).must[0].match.any == [PUBLIC, RESTRICTED]


def test_a_user_with_no_clearance_gets_a_filter_that_matches_nothing():
    """The dangerous case. An empty clearance set must produce a filter that
    admits nothing — never an absent filter, which would admit everything."""
    nobody = User(user_id="nobody", clearances=frozenset())
    condition = access_filter(nobody).must[0]
    assert condition.match.any == []


def test_combine_ands_the_acl_with_the_meeting_filter():
    """Both are payload filters and both must apply; neither may be dropped."""
    meeting = qmodels.Filter(
        must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchAny(any=["ata-2026-06"]))]
    )
    combined = combine(access_filter(ANALYST), meeting)
    keys = {condition.key for condition in combined.must}
    assert keys == {"classification", "doc_id"}


def test_combine_ignores_none_but_keeps_the_rest():
    combined = combine(None, access_filter(ANALYST), None)
    assert len(combined.must) == 1


def test_combine_of_nothing_is_none():
    assert combine(None, None) is None


def test_leaked_names_documents_the_user_may_not_see():
    assignments = classify_documents(DOCS, restricted_count=2)
    hits = [Hit("ata-2024-05"), Hit("ata-2026-06")]
    assert leaked(hits, assignments, ANALYST) == ["ata-2026-06"]
    assert leaked(hits, assignments, SUPERVISOR) == []


def test_an_unknown_document_defaults_to_public_not_to_denied():
    """Stated explicitly because it is a real trade-off, not an oversight.

    Fail-open on classification keeps an un-ingested document from becoming
    invisible to everyone. It is only safe because `apply_classifications` runs
    at pipeline construction over every document in the collection, so 'unknown'
    cannot occur for indexed content.
    """
    assert leaked([Hit("never-seen")], {}, ANALYST) == []


# --------------------------------------------------------------------------
# Integration: the filter has to be enforced by the server, not just built.
# --------------------------------------------------------------------------


def _qdrant_or_skip():
    try:
        from retrieval.store import collection_size, get_client

        client = get_client()
        if collection_size(client) == 0:
            pytest.skip("Qdrant collection is empty — run `python -m ingest.pipeline`")
        return client
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qdrant unreachable: {exc}")


@pytest.mark.integration
def test_an_uncleared_user_retrieves_zero_restricted_chunks_from_a_live_qdrant():
    """The claim, proven against the real store.

    Deliberately hostile to the ACL: it asks for 200 results (a third of the
    whole collection) with no meeting filter, so nothing but the payload filter
    stands between the analyst and the restricted documents.
    """
    client = _qdrant_or_skip()
    from governance.acl import apply_classifications
    from ingest.embedding import Embedder
    from retrieval.metadata import document_meta
    from retrieval.store import scroll_all, search

    corpus = scroll_all(client)
    assignments = classify_documents(document_meta(corpus), restricted_count=5)
    apply_classifications(client, assignments)
    restricted = restricted_doc_ids(assignments)
    assert restricted, "the fixture needs at least one restricted document"

    vector = Embedder().embed_query("Qual foi a decisao do Copom sobre a taxa Selic?")

    analyst_hits = search(client, vector, top_k=200, query_filter=access_filter(ANALYST))
    assert analyst_hits, "the filter must not empty the collection for a cleared-for-public user"
    assert [h.doc_id for h in analyst_hits if h.doc_id in restricted] == []

    supervisor_hits = search(client, vector, top_k=200, query_filter=access_filter(SUPERVISOR))
    assert [h.doc_id for h in supervisor_hits if h.doc_id in restricted], (
        "the supervisor must actually see restricted documents, or the test proves nothing"
    )


@pytest.mark.integration
def test_the_acl_beats_a_query_aimed_straight_at_a_restricted_document():
    """The metadata filter from M4 actively steers toward the June 2026 meeting,
    which is restricted. The ACL has to win that conflict."""
    client = _qdrant_or_skip()
    from governance.acl import apply_classifications
    from ingest.embedding import Embedder
    from retrieval.metadata import document_meta
    from retrieval.store import doc_id_filter, scroll_all, search

    corpus = scroll_all(client)
    assignments = classify_documents(document_meta(corpus), restricted_count=5)
    apply_classifications(client, assignments)
    restricted = sorted(restricted_doc_ids(assignments))

    vector = Embedder().embed_query("Qual foi a decisao do Copom na 279a reuniao?")
    aimed = combine(access_filter(ANALYST), doc_id_filter(restricted))
    assert search(client, vector, top_k=200, query_filter=aimed) == []
