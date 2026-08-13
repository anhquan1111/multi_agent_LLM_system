"""Chunking is pure text handling — no LLM, no network, no mocks needed."""

from pathlib import Path

from src.rag.chunking import CHUNK_SIZE, load_policy_chunks, split_document

POLICIES = Path("data/policies")

DOC = """# Returns

## Standard return window

You may return most items within 30 days.

## Damaged merchandise

Damaged items may be returned within 45 days.
"""


def test_splits_on_headings_not_mid_section():
    chunks = split_document(DOC, "returns.md")

    sections = [c.section for c in chunks]
    assert "Standard return window" in sections
    assert "Damaged merchandise" in sections


def test_a_chunk_never_merges_two_sections():
    """The whole point of stage-1 splitting: near-duplicate sections stay separate."""
    for chunk in split_document(DOC, "returns.md"):
        assert not ("30 days" in chunk.text and "45 days" in chunk.text)


def test_chunk_ids_are_unique_and_prefixed_by_source():
    chunks = split_document(DOC, "returns.md")

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("returns.md#") for i in ids)


def test_real_corpus_respects_the_size_cap():
    chunks = load_policy_chunks(POLICIES)

    assert len(chunks) > 20
    assert all(len(c.text) <= CHUNK_SIZE for c in chunks)
    assert {c.source for c in chunks} == {p.name for p in POLICIES.glob("*.md")}


def test_citation_shape_matches_the_api_contract():
    chunk = split_document(DOC, "returns.md")[0]

    assert set(chunk.as_citation()) == {"source", "chunk_id"}
