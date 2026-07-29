# Metrics

Empty on purpose. The metric implementations land in **M3**, after the gold set in
`../datasets/` has been validated by a human — measuring against an unvalidated
reference set produces numbers that look rigorous and mean nothing.

Planned split (kept deliberately separate, because mixing them is what makes most
RAG projects unable to diagnose their own failures):

- **Retrieval** — recall@k, precision@k, MRR, nDCG. Answers: *was the right span
  retrieved at all?*
- **Generation** — faithfulness/groundedness, answer relevance, context
  precision/recall (Ragas, pointed at the local model). Answers: *given the right
  span, was the answer right?*
- **End-to-end** — task success, citation correctness.
- **LLM-as-judge** — rubric plus calibration against ~30 human labels, reporting
  agreement rather than trusting the judge blindly.
