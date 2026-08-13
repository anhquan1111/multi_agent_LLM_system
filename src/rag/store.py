"""Chroma vector store: build the index, and search it.

The API key is passed explicitly rather than left to the SDK's environment lookup —
pydantic-settings never populates `os.environ` (see AGENTS.md pattern #6).
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings
from src.rag.chunking import Chunk

COLLECTION = "policies"


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not settings.has_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set — check .env or the host's env vars.")
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.embedding_model}",
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dim,
    )


def get_store() -> Chroma:
    """Open the persisted collection. Read path — does not embed anything."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def build_index(chunks: list[Chunk]) -> Chroma:
    """Embed every chunk and persist the collection, replacing whatever was there."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    store = get_store()

    existing = store.get(include=[])["ids"]
    if existing:
        store.delete(ids=existing)

    store.add_documents(
        [
            Document(
                page_content=c.text,
                metadata={"source": c.source, "section": c.section, "chunk_id": c.chunk_id},
                id=c.chunk_id,
            )
            for c in chunks
        ]
    )
    return store


def is_empty() -> bool:
    return not get_store().get(include=[])["ids"]


def ensure_index(policies_dir: Path) -> int:
    """Build the index if it is missing. Returns the number of chunks now indexed.

    Called at application startup rather than at Docker build time: building during the
    image build would require the API key as a build argument, which would bake a secret
    into the image layers. The corpus is small enough that embedding at boot is cheap.
    """
    from src.rag.chunking import load_policy_chunks

    if not is_empty():
        return len(get_store().get(include=[])["ids"])

    chunks = load_policy_chunks(policies_dir)
    if not chunks:
        return 0
    build_index(chunks)
    return len(chunks)


def search(query: str, k: int = 4) -> list[Chunk]:
    """Return the k most similar chunks. Empty list when the index has not been built."""
    docs = get_store().similarity_search(query, k=k)
    return [
        Chunk(
            text=d.page_content,
            source=d.metadata.get("source", "unknown"),
            section=d.metadata.get("section", ""),
            chunk_id=d.metadata.get("chunk_id", ""),
        )
        for d in docs
    ]
