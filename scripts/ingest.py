"""Build the Chroma index from data/policies/.

Run:  uv run python scripts/ingest.py

Costs one embedding API call per batch of chunks. Re-run after editing any policy
document, or after changing EMBEDDING_MODEL / EMBEDDING_DIM — a dimension change
silently invalidates every stored vector.
"""

import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402
from src.rag.chunking import load_policy_chunks  # noqa: E402
from src.rag.store import build_index  # noqa: E402

POLICIES = Path("data/policies")


def main() -> None:
    chunks = load_policy_chunks(POLICIES)
    if not chunks:
        sys.exit(f"❌ No markdown found in {POLICIES}/")

    per_doc = Counter(c.source for c in chunks)
    lengths = [len(c.text) for c in chunks]
    print(f"Split {len(per_doc)} documents into {len(chunks)} chunks")
    for source, n in sorted(per_doc.items()):
        print(f"   {source:16} {n:>3} chunks")
    print(f"   length: min={min(lengths)} avg={sum(lengths) // len(lengths)} max={max(lengths)}")

    print(f"\nEmbedding with {settings.embedding_model} @ {settings.embedding_dim} dims...")
    build_index(chunks)
    print(f"✅ Index written to {settings.chroma_dir}")


if __name__ == "__main__":
    main()
