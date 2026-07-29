"""Agent mode: tool use over two very different sources.

`rag_search` answers *what the Copom said and why* — unstructured Portuguese
prose. `sql_query` answers *what the numbers actually did* — structured monthly
series from the Open-Finance-LakeHouse gold marts.

Neither can answer the other's questions, which is the point. "The Copom cut to
14,25% in June 2026 — what had 12-month IPCA done by then?" needs both, and
deciding which tool answers which half is the only thing the agent is for.
"""
