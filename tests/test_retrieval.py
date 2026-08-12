"""Retrieval: index building, vector/keyword/hybrid search, and filters.

Runs entirely on the stub embedder, so it tests the retrieval *machinery* rather
than embedding quality. Embedding quality is measured in the eval suite, where it
belongs — a unit test asserting cosine rankings against a real model would be a
flaky benchmark wearing a test's clothes.
"""

from __future__ import annotations

import pytest

from signalforge.db import connect, upsert
from signalforge.retrieval.index import build_index, index_stats, search

DOCS = {
    "risk_factors": [
        "We identified a material weakness in our internal control over financial reporting.",
        "Our supply chain depends on a limited number of contract manufacturers in Asia.",
        "Pending litigation with a competitor could result in substantial damages.",
    ],
    "mdna": [
        "Revenue increased 14% driven by strong demand for subscription services.",
        "Gross margin improved 210 basis points due to a favorable product mix.",
        "We raised our full-year revenue guidance for fiscal 2026.",
    ],
}


@pytest.fixture
def corpus(warehouse):
    """A tiny two-section corpus for one company."""
    with connect() as con:
        upsert(
            con, "companies", [{"cik": "0000000001", "name": "Testco", "ticker": "TST"}], key="cik"
        )
        upsert(
            con,
            "filings",
            [
                {
                    "accession": "acc-1",
                    "cik": "0000000001",
                    "form": "10-K",
                    "filing_date": "2026-01-15",
                    "primary_doc": "d.htm",
                }
            ],
            key="accession",
        )
        sections, chunks = [], []
        for slug, texts in DOCS.items():
            section_id = f"acc-1:{slug}"
            sections.append(
                {
                    "section_id": section_id,
                    "accession": "acc-1",
                    "slug": slug,
                    "heading": slug,
                    "ordinal": 0,
                    "char_len": sum(map(len, texts)),
                    "text": "\n".join(texts),
                }
            )
            for i, text in enumerate(texts):
                chunks.append(
                    {
                        "chunk_id": f"{section_id}:{i}",
                        "section_id": section_id,
                        "accession": "acc-1",
                        "cik": "0000000001",
                        "ordinal": i,
                        "token_estimate": 20,
                        "text": text,
                    }
                )
        upsert(con, "sections", sections, key="section_id")
        upsert(con, "chunks", chunks, key="chunk_id")
    return 6


def test_build_index_embeds_all_chunks(corpus, router):
    n = build_index(router=router, model="stub-embed")
    assert n == corpus
    stats = index_stats()
    assert stats["chunks"] == corpus
    assert stats["embeddings"]["stub-embed"]["count"] == corpus


def test_index_stats_filters_by_model(corpus, router):
    """Regression: `model` was accepted and then ignored, so asking about one
    model's coverage silently reported every model's."""
    build_index(router=router, model="stub-embed")

    assert index_stats("stub-embed")["embeddings"]["stub-embed"]["count"] == corpus
    assert index_stats("not-an-embedding-model")["embeddings"] == {}
    # The chunk count is not model-scoped and stays reported either way.
    assert index_stats("not-an-embedding-model")["chunks"] == corpus


def test_build_index_is_incremental(corpus, router):
    build_index(router=router, model="stub-embed")
    assert build_index(router=router, model="stub-embed") == 0, "should re-embed nothing"


def test_build_index_scoped_to_accessions(corpus, router):
    assert build_index(router=router, model="stub-embed", accessions=["nope"]) == 0


def test_keyword_search_finds_exact_terms(corpus, router):
    hits = search(
        "material weakness internal control", router=router, model="stub-embed", mode="keyword", k=3
    )
    assert hits
    assert "material weakness" in hits[0].text.lower()


def test_keyword_search_ignores_stopwords_only_query(corpus, router):
    assert search("the and of", router=router, model="stub-embed", mode="keyword") == []


def test_vector_search_returns_scored_hits(corpus, router):
    build_index(router=router, model="stub-embed")
    hits = search("guidance", router=router, model="stub-embed", mode="vector", k=3)
    assert len(hits) == 3
    assert all(h.sources == ("vector",) for h in hits)
    # Descending by score.
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_hybrid_search_merges_both_retrievers(corpus, router):
    build_index(router=router, model="stub-embed")
    hits = search("material weakness", router=router, model="stub-embed", mode="hybrid", k=6)
    assert hits
    assert any("keyword" in h.sources for h in hits)
    assert any("vector" in h.sources for h in hits)


def test_hybrid_fusion_needs_no_score_normalisation(corpus, router):
    """RRF scores are positional, so an unbounded keyword score cannot dominate."""
    build_index(router=router, model="stub-embed")
    hits = search("litigation competitor damages", router=router, model="stub-embed", k=4)
    assert all(0 < h.score < 1 for h in hits)


def test_section_filter_narrows_results(corpus, router):
    build_index(router=router, model="stub-embed")
    hits = search("revenue", router=router, model="stub-embed", slug="mdna", k=5)
    assert hits
    assert all(h.slug == "mdna" for h in hits)


def test_cik_filter_excludes_other_companies(corpus, router):
    build_index(router=router, model="stub-embed")
    assert search("revenue", router=router, model="stub-embed", cik="0000000009") == []


def test_search_on_empty_index_is_not_an_error(warehouse, router):
    assert search("anything", router=router, model="stub-embed") == []


def test_k_is_respected(corpus, router):
    build_index(router=router, model="stub-embed")
    assert len(search("revenue growth guidance", router=router, model="stub-embed", k=2)) == 2


def test_embedding_resolution_respects_the_env_override(registry, monkeypatch):
    """One variable must be able to make the whole system hermetic.

    Regression test for a real CI failure: the /search endpoint embeds without
    naming a model, so it resolved the registry default (real Ollama) even with
    the completion provider stubbed. It passed locally because Ollama was
    running, and failed on a runner where it was not.
    """
    from signalforge.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SF_EMBED_MODEL", "stub-embed")
    assert registry.embedding().provider == "stub"

    get_settings.cache_clear()
    monkeypatch.delenv("SF_EMBED_MODEL", raising=False)
    # With no override, configs/models.yaml remains the source of truth.
    assert registry.embedding().name == registry.default_embedding
    get_settings.cache_clear()
