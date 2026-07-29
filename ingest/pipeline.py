"""End-to-end ingest: manifest -> PDF text -> chunks -> bge-m3 -> Qdrant.

    uv run python -m ingest.pipeline --download 30
    uv run python -m ingest.pipeline            # reuse whatever is in data/raw
"""

from __future__ import annotations

import argparse
import sys
import time

from ingest.chunking import fixed_size_chunks
from ingest.corpus import download_corpus, load_manifest
from ingest.embedding import Embedder
from ingest.loading import load_pdf
from rag.config import settings
from retrieval.store import (
    collection_size,
    get_client,
    recreate_collection,
    upsert_chunks,
)


def run(download: int | None = None, batch_size: int | None = None) -> int:
    if download:
        print(f"Downloading up to {download} Copom minutes from bcb.gov.br ...")
        download_corpus(download)

    manifest = load_manifest()
    print(f"\nManifest: {len(manifest)} documents")

    # 1. load + chunk
    all_chunks = []
    for entry in manifest:
        path = settings.raw_dir / entry.filename
        if not path.exists():
            print(f"  ! missing {entry.filename} — run with --download")
            continue
        doc = load_pdf(
            path,
            doc_id=entry.doc_id,
            title=entry.title,
            url=entry.source_page or entry.pdf_url,
            reference_date=entry.reference_date,
        )
        chunks = fixed_size_chunks(doc, settings.chunk_size, settings.chunk_overlap)
        all_chunks.extend(chunks)
        print(f"  {entry.title[:48]:<48} {len(doc.pages):>3} pages -> {len(chunks):>4} chunks")

    if not all_chunks:
        print("\nNo chunks produced. Did the corpus download?")
        return 1

    print(f"\nTotal: {len(all_chunks)} chunks")

    # 2. embed (local, CPU)
    print(f"Loading embedding model {settings.embedding_model} on {settings.embedding_device} ...")
    embedder = Embedder()
    dim = embedder.dimension
    print(f"  dimension = {dim}")

    started = time.perf_counter()
    vectors = embedder.embed_documents(
        [c.text for c in all_chunks],
        batch_size=batch_size or settings.embedding_batch_size,
    )
    elapsed = time.perf_counter() - started
    print(f"  embedded {len(vectors)} chunks in {elapsed:.0f}s "
          f"({len(vectors) / max(elapsed, 1e-6):.1f} chunks/s)")

    # 3. upsert
    client = get_client()
    print(f"\nRecreating collection '{settings.qdrant_collection}' at {settings.qdrant_url} ...")
    recreate_collection(client, dim)

    step = 256
    written = 0
    for i in range(0, len(all_chunks), step):
        written += upsert_chunks(
            client,
            all_chunks[i : i + step],
            vectors[i : i + step],
            start_id=i,
        )
        print(f"  upserted {written}/{len(all_chunks)}")

    total = collection_size(client)
    print(f"\nDone. Collection '{settings.qdrant_collection}' holds {total} points.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the BACEN corpus into Qdrant.")
    parser.add_argument(
        "-d",
        "--download",
        type=int,
        nargs="?",
        const=30,
        default=None,
        help="download N documents before ingesting (default 30)",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=None)
    args = parser.parse_args(argv)
    return run(download=args.download, batch_size=args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
