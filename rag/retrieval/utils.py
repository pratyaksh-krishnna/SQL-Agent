"""
Retrieval utilities for the SQL-agent RAG pipeline.

Talks directly to the PostgreSQL functions defined in database.txt
(vector_search_document_chunks, keyword_search_document_chunks) and
provides multi-query expansion, RRF fusion, and context formatting.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PostgreSQL connection (shared with ingestion pipeline)
# ---------------------------------------------------------------------------

def get_pg_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def fetch_project_settings(pg_conn, project_id: str) -> dict:
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM project_settings WHERE project_id = %s", (project_id,)
        )
        if cur.description is None:
            return {}
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        return dict(zip(cols, row)) if row else {}


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_query(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Embed a single query string via OpenAI."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# Search — vector & keyword (call the PostgreSQL functions)
# ---------------------------------------------------------------------------

def vector_search(
    pg_conn,
    query_embedding: list[float],
    project_id: str,
    match_threshold: float = 0.3,
    chunks_per_search: int = 20,
    filter_doc_type: str | None = None,
    filter_table_names: list[str] | None = None,
) -> list[dict]:
    """Call ``vector_search_document_chunks`` and return rows as dicts."""
    embedding_str = str(query_embedding)
    with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT * FROM vector_search_document_chunks(
                %s::vector, %s::uuid, %s, %s, %s, %s
            )
            """,
            (
                embedding_str,
                project_id,
                match_threshold,
                chunks_per_search,
                filter_doc_type,
                filter_table_names,
            ),
        )
        return [dict(row) for row in cur.fetchall()]


def keyword_search(
    pg_conn,
    query_text: str,
    project_id: str,
    chunks_per_search: int = 20,
    filter_doc_type: str | None = None,
    filter_table_names: list[str] | None = None,
) -> list[dict]:
    """Call ``keyword_search_document_chunks`` and return rows as dicts."""
    with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT * FROM keyword_search_document_chunks(
                %s, %s::uuid, %s, %s, %s
            )
            """,
            (
                query_text,
                project_id,
                chunks_per_search,
                filter_doc_type,
                filter_table_names,
            ),
        )
        return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def rrf_rank_and_fuse(
    search_results_list: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked result lists using weighted RRF.

    Each chunk's fused score is::

        score = Σ  weight_i / (k + rank_i + 1)

    Chunks are returned in descending fused-score order.
    """
    if not search_results_list or not any(search_results_list):
        return []

    if weights is None:
        weights = [1.0 / len(search_results_list)] * len(search_results_list)

    chunk_scores: dict[str, float] = {}
    all_chunks: dict[str, dict] = {}

    for search_idx, results in enumerate(search_results_list):
        w = weights[search_idx]
        for rank, chunk in enumerate(results):
            chunk_id = str(chunk.get("id", ""))
            if not chunk_id:
                continue
            rrf_score = w / (k + rank + 1)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + rrf_score
            all_chunks.setdefault(chunk_id, chunk)

    sorted_ids = sorted(chunk_scores, key=lambda cid: chunk_scores[cid], reverse=True)
    return [all_chunks[cid] for cid in sorted_ids]


# ---------------------------------------------------------------------------
# Multi-query expansion
# ---------------------------------------------------------------------------

def generate_query_variations(
    user_query: str,
    n: int = 3,
    model: str = "gpt-4o",
) -> list[str]:
    """
    Use an LLM to rephrase *user_query* into *n* alternative queries that
    target different angles of the same information need.

    The original query is always included as the first element.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0.7)
    prompt = (
        f"You are a helpful assistant that generates search queries for a "
        f"database-schema documentation search engine.\n\n"
        f"Original question: {user_query}\n\n"
        f"Generate {n} alternative phrasings of this question that might "
        f"retrieve different relevant schema documentation. Focus on:\n"
        f"- Different table/column names the user might be referring to\n"
        f"- Different ways to describe the same data relationship\n"
        f"- Technical SQL terminology vs business terminology\n\n"
        f"Return ONLY the queries, one per line. No numbering, no bullets."
    )
    response = llm.invoke(prompt)
    variations = [
        line.strip()
        for line in response.content.strip().splitlines()
        if line.strip()
    ][:n]

    return [user_query] + variations


# ---------------------------------------------------------------------------
# Context builder — format retrieved chunks for the agent / LLM
# ---------------------------------------------------------------------------

def build_schema_context(chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Convert retrieved ``document_chunks`` rows into a structured context
    string and a list of chunk IDs (for provenance tracking).

    Returns
    -------
    (context_text, chunk_ids)
    """
    if not chunks:
        return "", []

    chunk_ids: list[str] = []
    sections: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        cid = str(chunk.get("id", ""))
        chunk_ids.append(cid)

        doc_type = chunk.get("doc_type", "unknown")
        tables = chunk.get("table_names") or []
        table_label = ", ".join(tables) if tables else "cross-table"
        similarity = chunk.get("similarity") or chunk.get("keyword_rank") or ""
        score_str = f"  (score: {similarity:.3f})" if isinstance(similarity, float) else ""

        sections.append(
            f"--- Schema Context [{i}] | type: {doc_type} | "
            f"tables: {table_label}{score_str} ---\n"
            f"{chunk.get('content', '')}"
        )

    context = "\n\n".join(sections)
    return context, chunk_ids
