"""Governance: access control at retrieval time, and an audit trail.

Both are about the same principle — a RAG system's security boundary is the
*retrieval query*, not the answer. Filtering results after the vector search has
already returned them means the restricted content was read, ranked, and held in
memory; whether it is then shown is a UI decision, not a control.
"""
