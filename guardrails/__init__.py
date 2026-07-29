"""Input/output guardrails: PII detection and masking, prompt-injection defence.

Everything here runs on both sides of the model. Masking only the input protects
the model from the user; masking only the output protects the user from the
model. A RAG system needs both, because the corpus is a third party that neither
of them controls.
"""
