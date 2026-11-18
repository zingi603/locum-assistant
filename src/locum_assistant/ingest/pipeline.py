"""Download -> parse -> chunk -> embed -> store, one source at a time."""

import logging
from pathlib import Path

import httpx

from locum_assistant.ingest.chunk import chunk_document
from locum_assistant.ingest.parse import parse
from locum_assistant.ingest.sources import Source
from locum_assistant.retrieval.embed import Embedder
from locum_assistant.retrieval.store import Store

log = logging.getLogger(__name__)
USER_AGENT = "locum-assistant/0.1 (portfolio project; github.com/zingi603)"


def download(source: Source, raw_dir: Path, refresh: bool = False) -> bytes:
    """Fetch once and cache on disk, so re-running ingest doesn't hit the publisher again."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{source.id}.{'pdf' if source.kind == 'pdf' else 'html'}"
    if path.exists() and not refresh:
        return path.read_bytes()
    response = httpx.get(source.url, headers={"User-Agent": USER_AGENT}, timeout=60,
                         follow_redirects=True)
    response.raise_for_status()
    path.write_bytes(response.content)
    return response.content


def ingest_source(source: Source, store: Store, embedder: Embedder, raw_dir: Path,
                  refresh: bool = False) -> int:
    document = parse(source, download(source, raw_dir, refresh))
    chunks = chunk_document(document.sections)
    if not chunks:
        raise ValueError(f"{source.id}: no text extracted")
    embeddings = embedder.embed_documents([f"{c.heading}\n{c.text}" for c in chunks])
    store.replace_document(source, chunks, embeddings)
    log.info("%s: %d sections, %d chunks", source.id, len(document.sections), len(chunks))
    return len(chunks)
