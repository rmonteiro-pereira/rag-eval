"""Local embeddings with bge-m3 (multilingual, strong on Portuguese).

Runs through `sentence-transformers` on CPU. No API key, no network at query
time — the model is downloaded once from HuggingFace and cached in
`~/.cache/huggingface`.
"""

from __future__ import annotations

from functools import lru_cache

from rag.config import settings


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str):
    # Imported lazily: `sentence_transformers` pulls in torch, which costs
    # several seconds of import time we do not want in unrelated commands.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class Embedder:
    """Thin wrapper so the rest of the code never imports torch directly."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name, self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size or settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        # bge-m3 is trained without an instruction prefix, so queries and
        # documents go through the exact same encoder path.
        vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return vector.tolist()
