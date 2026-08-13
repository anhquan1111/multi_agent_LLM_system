"""Two-stage splitting of the policy documents.

Stage 1 splits on markdown headings so a chunk never merges text from two different
sections — "Returns" and "Warranty" contain near-duplicate passages and heading metadata
is what keeps them apart at retrieval time. Stage 2 caps the length of long sections.

No LLM and no network here, so this is fully unit-testable.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
HEADERS = [("#", "doc_title"), ("##", "section")]


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # "returns.md"
    section: str  # "Damaged or defective merchandise"
    chunk_id: str  # "returns.md#3"

    def as_citation(self) -> dict[str, str]:
        return {"source": self.source, "chunk_id": self.chunk_id}


def split_document(text: str, source: str) -> list[Chunk]:
    """Split one markdown document into retrievable chunks."""
    by_heading = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(text)
    by_size = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(by_heading)

    return [
        Chunk(
            text=doc.page_content.strip(),
            source=source,
            section=doc.metadata.get("section", ""),
            chunk_id=f"{source}#{i}",
        )
        for i, doc in enumerate(by_size)
        if doc.page_content.strip()
    ]


def load_policy_chunks(directory: Path) -> list[Chunk]:
    """Split every markdown file in `directory`, sorted so chunk ids stay stable."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(split_document(path.read_text(encoding="utf-8"), path.name))
    return chunks
