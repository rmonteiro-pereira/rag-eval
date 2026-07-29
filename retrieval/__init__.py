"""Retrieval layer.

Every strategy is a flag on one `RetrievalConfig` and runs through one code path
(`retrieval/configs.py`), so an ablation delta measures the component under test
rather than two different implementations of retrieval.

The M1 dense-only retriever used to live in `retrieval/dense.py`. It was deleted
in M4 rather than kept alongside: it is now the `dense` arm, built from the same
composition as every other arm, which is the only way `dense` vs `hybrid` is a
controlled comparison.
"""
