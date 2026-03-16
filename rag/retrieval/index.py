"""
RAG Retrieval Pipeline — main orchestrator.

Retrieves the most relevant schema-documentation chunks for a user's
natural-language question so the SQL agent can generate accurate queries.

Strategies (auto-selected from ``project_settings``):
  - **vector**      — embedding similarity only
  - **hybrid**      — vector + keyword (FTS) fused with RRF
  - **multi-query** — LLM generates query variations, each is searched,
                       results are fused with RRF
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rag.retrieval.utils import (
    build_schema_context,
    embed_query,
    fetch_project_settings,
    generate_query_variations,
    get_pg_connection,
    keyword_search,
    rrf_rank_and_fuse,
    vector_search,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Everything the agent needs from a single retrieval call."""
    context: str
    chunk_ids: list[str] = field(default_factory=list)
    total_chunks_searched: int = 0


# ───────────────────────────────────────────────────────────────────────────
# Internal search wrappers
# ───────────────────────────────────────────────────────────────────────────

def _run_vector_search(
    pg_conn,
    query: str,
    project_id: str,
    settings: dict,
    filter_doc_type: str | None = None,
    filter_table_names: list[str] | None = None,
) -> list[dict]:
    embedding = embed_query(query, model=settings.get("embedding_model", "text-embedding-3-small"))
    return vector_search(
        pg_conn,
        embedding,
        project_id,
        match_threshold=float(settings.get("similarity_threshold", 0.3)),
        chunks_per_search=int(settings.get("chunks_per_search", 20)),
        filter_doc_type=filter_doc_type,
        filter_table_names=filter_table_names,
    )


def _run_keyword_search(
    pg_conn,
    query: str,
    project_id: str,
    settings: dict,
    filter_doc_type: str | None = None,
    filter_table_names: list[str] | None = None,
) -> list[dict]:
    return keyword_search(
        pg_conn,
        query,
        project_id,
        chunks_per_search=int(settings.get("chunks_per_search", 20)),
        filter_doc_type=filter_doc_type,
        filter_table_names=filter_table_names,
    )


def _run_hybrid_search(
    pg_conn,
    query: str,
    project_id: str,
    settings: dict,
    filter_doc_type: str | None = None,
    filter_table_names: list[str] | None = None,
) -> list[dict]:
    vec = _run_vector_search(pg_conn, query, project_id, settings, filter_doc_type, filter_table_names)
    kw = _run_keyword_search(pg_conn, query, project_id, settings, filter_doc_type, filter_table_names)
    v_weight = float(settings.get("vector_weight", 0.7))
    k_weight = float(settings.get("keyword_weight", 0.3))
    return rrf_rank_and_fuse([vec, kw], weights=[v_weight, k_weight])


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────

def retrieve_context(
    project_id: str,
    user_query: str,
    pg_conn=None,
    filter_doc_type: str | None = None,
    filter_table_names: list[str] | None = None,
) -> RetrievalResult:
    """
    Run the full retrieval pipeline for *user_query* within *project_id*.

    Strategy is determined automatically from ``project_settings``:
      - ``number_of_queries > 1`` ⇒ multi-query expansion
      - ``keyword_weight > 0``    ⇒ hybrid (vector + keyword)
      - otherwise                 ⇒ pure vector search

    Returns a :class:`RetrievalResult` with the formatted context string
    and provenance chunk IDs.
    """
    own_conn = pg_conn is None
    if own_conn:
        pg_conn = get_pg_connection()

    try:
        settings = fetch_project_settings(pg_conn, project_id)
        num_queries = int(settings.get("number_of_queries", 1))
        k_weight = float(settings.get("keyword_weight", 0.0))
        final_k = int(settings.get("final_context_size", 10))
        use_hybrid = k_weight > 0
        use_multi_query = num_queries > 1

        search_fn = _run_hybrid_search if use_hybrid else _run_vector_search
        logger.info(
            "Retrieval config: hybrid=%s, multi_query=%s (n=%d), final_k=%d",
            use_hybrid, use_multi_query, num_queries, final_k,
        )

        if use_multi_query:
            queries = generate_query_variations(
                user_query, n=num_queries, model=settings.get("llm_model", "gpt-4o"),
            )
            all_results: list[list[dict]] = []
            for q in queries:
                all_results.append(
                    search_fn(pg_conn, q, project_id, settings, filter_doc_type, filter_table_names)
                )
            total_searched = sum(len(r) for r in all_results)
            chunks = rrf_rank_and_fuse(all_results)
        else:
            chunks = search_fn(pg_conn, user_query, project_id, settings, filter_doc_type, filter_table_names)
            total_searched = len(chunks)

        chunks = chunks[:final_k]
        context, chunk_ids = build_schema_context(chunks)

        logger.info(
            "Retrieved %d chunks (searched %d) for project %s",
            len(chunks), total_searched, project_id,
        )
        return RetrievalResult(
            context=context,
            chunk_ids=chunk_ids,
            total_chunks_searched=total_searched,
        )
    finally:
        if own_conn:
            pg_conn.close()
