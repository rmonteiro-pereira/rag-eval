"""Corpus acquisition: BACEN / Copom meeting minutes ("atas do Copom").

Public, free data from bcb.gov.br. The site exposes a small JSON service that
lists the latest minutes together with the PDF path and the human-facing page:

    GET /api/servico/sitebcb/atascopom/ultimas?quantidade=N&filtro=

    {"conteudo": [
        {"DataReferencia": "2026-06-17T03:00:00Z",
         "Titulo": "279a Reuniao - 16-17 junho, 2026",
         "Url": "/content/copom/atascopom/Copom279-not20260617279.pdf",
         "LinkPagina": "/publicacoes/atascopom/17062026"}, ...]}

PDFs land in `data/raw/` (gitignored). Only `data/manifest.json` is committed,
so the corpus is reproducible without shipping binaries into git.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from rag.config import settings

BCB_BASE = "https://www.bcb.gov.br"
ATAS_ENDPOINT = "/api/servico/sitebcb/atascopom/ultimas"
USER_AGENT = "rag-eval/0.1 (portfolio research project; contact: local)"


@dataclass
class ManifestEntry:
    doc_id: str
    title: str
    reference_date: str
    pdf_url: str
    source_page: str
    filename: str
    sha256: str
    bytes: int


def _slugify(title: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in title]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:60]


def list_minutes(quantity: int, client: httpx.Client) -> list[dict]:
    """Return the raw catalogue entries for the `quantity` most recent minutes."""
    resp = client.get(
        f"{BCB_BASE}{ATAS_ENDPOINT}",
        params={"quantidade": quantity, "filtro": ""},
    )
    resp.raise_for_status()
    return resp.json().get("conteudo", [])


def download_corpus(quantity: int = 30, raw_dir: Path | None = None) -> list[ManifestEntry]:
    """Download up to `quantity` Copom minutes and write the committed manifest.

    Already-downloaded PDFs are skipped, so re-running is cheap and idempotent.
    """
    raw_dir = raw_dir or settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    entries: list[ManifestEntry] = []
    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(timeout=120.0, follow_redirects=True, headers=headers) as client:
        catalogue = list_minutes(quantity, client)
        if not catalogue:
            raise RuntimeError("BCB returned an empty catalogue of Copom minutes")

        for item in catalogue:
            title = item["Titulo"].strip()
            pdf_path = item["Url"]
            pdf_url = pdf_path if pdf_path.startswith("http") else f"{BCB_BASE}{pdf_path}"
            reference_date = item.get("DataReferencia", "")[:10]
            doc_id = _slugify(f"{reference_date}-{title}")
            filename = f"{doc_id}.pdf"
            target = raw_dir / filename

            if not target.exists():
                pdf = client.get(pdf_url)
                pdf.raise_for_status()
                if not pdf.content.startswith(b"%PDF"):
                    print(f"  ! skipping {title}: response is not a PDF")
                    continue
                target.write_bytes(pdf.content)
                print(f"  + {filename} ({len(pdf.content) / 1024:.0f} KB)")
            else:
                print(f"  = {filename} (cached)")

            raw = target.read_bytes()
            entries.append(
                ManifestEntry(
                    doc_id=doc_id,
                    title=title,
                    reference_date=reference_date,
                    pdf_url=pdf_url,
                    source_page=f"{BCB_BASE}{item.get('LinkPagina', '')}",
                    filename=filename,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    bytes=len(raw),
                )
            )

    write_manifest(entries)
    return entries


def write_manifest(entries: list[ManifestEntry], path: Path | None = None) -> Path:
    path = path or settings.manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Banco Central do Brasil - Atas do Copom",
        "license": "Public information published by BACEN (dados abertos / public domain).",
        "endpoint": f"{BCB_BASE}{ATAS_ENDPOINT}",
        "document_count": len(entries),
        "documents": [asdict(e) for e in entries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(path: Path | None = None) -> list[ManifestEntry]:
    path = path or settings.manifest_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ManifestEntry(**d) for d in payload["documents"]]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download the BACEN Copom minutes corpus.")
    parser.add_argument("-n", "--quantity", type=int, default=30)
    args = parser.parse_args()

    entries = download_corpus(args.quantity)
    print(f"\n{len(entries)} documents in {settings.raw_dir}")
    print(f"manifest -> {settings.manifest_path}")


if __name__ == "__main__":
    main()
