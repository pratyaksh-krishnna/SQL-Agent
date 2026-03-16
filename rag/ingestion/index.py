"""
RAG Ingestion Pipeline — main orchestrator.

Workflow
--------
1. Download files from S3 → load into a temporary DuckDB instance.
2. Extract column-level schema (DuckDB introspection + LangChain DDL).
3. Detect inter-table relationships (three pluggable strategies).
4. Generate rich natural-language documents from schema + relationships.
5. Chunk the documents and embed them (OpenAI).
6. Persist everything to PostgreSQL so the retrieval layer can use it.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.ingestion.utils import (
    ColumnSchema,
    DetectedRelationship,
    DocumentChunk,
    SchemaDocument,
    chunk_text,
    cleanup_existing_documents,
    cleanup_existing_relationships,
    combine_detected_relationships,
    create_duckdb_connection,
    detect_relationships_llm,
    detect_relationships_name_match,
    detect_relationships_value_overlap,
    download_from_s3,
    embed_texts,
    extract_column_schemas,
    fetch_project_settings,
    fetch_uploaded_tables,
    fetch_user_relationships,
    get_langchain_table_info,
    get_pg_connection,
    load_file_to_duckdb,
    save_document_chunks,
    save_relationships,
    save_schema_document,
    save_table_schemas,
    update_uploaded_table_status,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Prompt templates for document generation
# ═══════════════════════════════════════════════════════════════════════════

TABLE_OVERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior data engineer writing internal documentation for a "
        "SQL database.  Your documentation will be embedded and used by a "
        "text-to-SQL agent, so be precise about column names, data types, and "
        "what each column represents.  Write in Markdown.",
    ),
    (
        "human",
        "Write a comprehensive overview document for the table **{table_name}**.\n\n"
        "### DDL / Schema\n"
        "{ddl_info}\n\n"
        "### Column details\n"
        "{column_details}\n\n"
        "Cover:\n"
        "1. What this table likely represents (its business purpose).\n"
        "2. A short description of every column — what it stores, its type, "
        "whether it can be NULL, and a few example values.\n"
        "3. Any primary-key or unique constraints.\n"
        "4. Suggested query patterns someone might use on this table.",
    ),
])

COLUMN_DETAIL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a data documentation specialist.  For each column in the "
        "table below, write a detailed one-paragraph description that a "
        "text-to-SQL agent can use to choose the correct column in a query.",
    ),
    (
        "human",
        "Table: **{table_name}**\n\n"
        "{column_details}\n\n"
        "For each column write:\n"
        "- A plain-English description of what the column stores.\n"
        "- The SQL data type and nullability.\n"
        "- Representative sample values.\n"
        "- Any naming conventions (e.g. _id suffix → foreign key).",
    ),
])

RELATIONSHIP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are documenting foreign-key relationships between database "
        "tables.  The documentation will feed a RAG pipeline for a "
        "text-to-SQL agent.",
    ),
    (
        "human",
        "Document the following relationship:\n\n"
        "**{from_table}.{from_column}** → **{to_table}.{to_column}**\n"
        "Type: {rel_type}  |  Confidence: {confidence}  |  Detection: {method}\n\n"
        "From-table columns:\n{from_columns}\n\n"
        "To-table columns:\n{to_columns}\n\n"
        "Explain:\n"
        "1. What this relationship means in business terms.\n"
        "2. The exact JOIN clause to use.\n"
        "3. Common query patterns involving both tables.",
    ),
])

JOIN_GUIDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are writing a comprehensive join-guide for a multi-table SQL "
        "database.  This guide will be the primary reference for a "
        "text-to-SQL agent when it needs to combine data from several tables.",
    ),
    (
        "human",
        "### Tables in this project\n{table_summaries}\n\n"
        "### Detected relationships\n{relationship_summaries}\n\n"
        "Write a Markdown guide that covers:\n"
        "1. A summary of every table and its role.\n"
        "2. Every detected relationship with the exact JOIN syntax.\n"
        "3. Recommended multi-table query patterns.\n"
        "4. Pitfalls (e.g. NULLable FKs, many-to-many resolution).",
    ),
])


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — formatting data for prompts
# ═══════════════════════════════════════════════════════════════════════════

def _format_column_details(columns: list[ColumnSchema]) -> str:
    lines: list[str] = []
    for c in columns:
        pk = " [PK]" if c.is_primary_key else ""
        fk = " [FK]" if c.is_foreign_key else ""
        nullable = "NULL" if c.is_nullable else "NOT NULL"
        samples = ", ".join(c.sample_values[:5]) if c.sample_values else "—"
        lines.append(
            f"- **{c.column_name}** ({c.data_type}, {nullable}{pk}{fk})  "
            f"samples: {samples}"
        )
    return "\n".join(lines)


def _format_table_summary(table_name: str, columns: list[ColumnSchema]) -> str:
    col_names = ", ".join(c.column_name for c in columns)
    return f"- **{table_name}** — columns: {col_names}"


def _format_relationship_summary(rel: DetectedRelationship) -> str:
    return (
        f"- {rel.from_table_name}.{rel.from_column} → "
        f"{rel.to_table_name}.{rel.to_column} "
        f"({rel.relationship_type}, confidence={rel.confidence:.2f})"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Document generation
# ═══════════════════════════════════════════════════════════════════════════

def _get_llm(model_name: str = "gpt-4o", temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=temperature)


def generate_table_overview(
    llm: ChatOpenAI,
    table_name: str,
    columns: list[ColumnSchema],
    ddl_info: str,
) -> SchemaDocument:
    """Produce a *table_overview* document for one table."""
    chain = TABLE_OVERVIEW_PROMPT | llm
    result = chain.invoke({
        "table_name": table_name,
        "ddl_info": ddl_info,
        "column_details": _format_column_details(columns),
    })
    return SchemaDocument(
        doc_type="table_overview",
        title=f"Table Overview — {table_name}",
        content=result.content,
        table_name=table_name,
    )


def generate_column_details(
    llm: ChatOpenAI,
    table_name: str,
    columns: list[ColumnSchema],
) -> SchemaDocument:
    """Produce a *column_detail* document for one table."""
    chain = COLUMN_DETAIL_PROMPT | llm
    result = chain.invoke({
        "table_name": table_name,
        "column_details": _format_column_details(columns),
    })
    return SchemaDocument(
        doc_type="column_detail",
        title=f"Column Details — {table_name}",
        content=result.content,
        table_name=table_name,
    )


def generate_relationship_doc(
    llm: ChatOpenAI,
    rel: DetectedRelationship,
    table_schemas: dict[str, list[ColumnSchema]],
) -> SchemaDocument:
    """Produce a *relationship* document for a single detected FK."""
    chain = RELATIONSHIP_PROMPT | llm
    result = chain.invoke({
        "from_table": rel.from_table_name,
        "from_column": rel.from_column,
        "to_table": rel.to_table_name,
        "to_column": rel.to_column,
        "rel_type": rel.relationship_type,
        "confidence": rel.confidence,
        "method": rel.detection_method,
        "from_columns": _format_column_details(table_schemas.get(rel.from_table_name, [])),
        "to_columns": _format_column_details(table_schemas.get(rel.to_table_name, [])),
    })
    return SchemaDocument(
        doc_type="relationship",
        title=f"Relationship — {rel.from_table_name}.{rel.from_column} → {rel.to_table_name}.{rel.to_column}",
        content=result.content,
        table_name=None,
    )


def generate_join_guide(
    llm: ChatOpenAI,
    table_schemas: dict[str, list[ColumnSchema]],
    relationships: list[DetectedRelationship],
) -> SchemaDocument:
    """Produce a single *join_guide* document for the whole project."""
    table_summaries = "\n".join(
        _format_table_summary(t, cols) for t, cols in table_schemas.items()
    )
    rel_summaries = "\n".join(_format_relationship_summary(r) for r in relationships)
    if not rel_summaries:
        rel_summaries = "(no relationships detected)"

    chain = JOIN_GUIDE_PROMPT | llm
    result = chain.invoke({
        "table_summaries": table_summaries,
        "relationship_summaries": rel_summaries,
    })
    return SchemaDocument(
        doc_type="join_guide",
        title="Join Guide",
        content=result.content,
        table_name=None,
    )


def generate_all_documents(
    llm: ChatOpenAI,
    table_schemas: dict[str, list[ColumnSchema]],
    relationships: list[DetectedRelationship],
    langchain_ddl: str,
) -> list[SchemaDocument]:
    """
    Generate the full set of RAG documents for a project:
      - one table_overview per table
      - one column_detail per table
      - one relationship doc per detected FK
      - one global join_guide
    """
    docs: list[SchemaDocument] = []

    for table_name, columns in table_schemas.items():
        docs.append(generate_table_overview(llm, table_name, columns, langchain_ddl))
        docs.append(generate_column_details(llm, table_name, columns))

    for rel in relationships:
        docs.append(generate_relationship_doc(llm, rel, table_schemas))

    docs.append(generate_join_guide(llm, table_schemas, relationships))

    logger.info("Generated %d schema documents", len(docs))
    return docs


# ═══════════════════════════════════════════════════════════════════════════
# Chunking + embedding wrapper
# ═══════════════════════════════════════════════════════════════════════════

def chunk_and_embed_document(
    doc: SchemaDocument,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """Split a SchemaDocument into chunks and embed each chunk."""
    raw_chunks = chunk_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not raw_chunks:
        return []

    embeddings = embed_texts(raw_chunks, model=embedding_model)

    table_names = [doc.table_name] if doc.table_name else []
    return [
        DocumentChunk(
            content=text,
            chunk_index=idx,
            doc_type=doc.doc_type,
            table_names=table_names,
            embedding=emb,
        )
        for idx, (text, emb) in enumerate(zip(raw_chunks, embeddings))
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_ingestion_pipeline(
    project_id: str,
    uploaded_table_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    End-to-end ingestion for one project.

    Parameters
    ----------
    project_id         : UUID of the project.
    uploaded_table_ids : specific tables to (re-)ingest.  When *None*, all
                         tables in the project are processed.

    Returns
    -------
    dict with summary stats (tables loaded, schemas saved, docs generated, etc.).
    """
    pg_conn = get_pg_connection()
    settings = fetch_project_settings(pg_conn, project_id)
    embedding_model = settings.get("embedding_model", "text-embedding-3-small")
    llm_model = settings.get("llm_model", "gpt-4o")

    all_tables = fetch_uploaded_tables(pg_conn, project_id)
    if uploaded_table_ids:
        all_tables = [t for t in all_tables if str(t["id"]) in uploaded_table_ids]

    if not all_tables:
        logger.warning("No uploaded tables found for project %s", project_id)
        return {"status": "no_tables"}

    # ------------------------------------------------------------------
    # 1. Download from S3 → load into DuckDB
    # ------------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix=f"sql_agent_{project_id}_")
    duckdb_path = os.path.join(tmp_dir, "project.duckdb")
    duck_con = create_duckdb_connection(duckdb_path)

    table_id_map: dict[str, str] = {}        # table_name → uploaded_table UUID
    all_schemas: dict[str, list[ColumnSchema]] = {}

    for tbl in all_tables:
        tid = str(tbl["id"])
        tname = tbl["table_name"]
        try:
            update_uploaded_table_status(pg_conn, tid, "processing")

            local_path = download_from_s3(tbl["s3_key"])
            row_count = load_file_to_duckdb(
                duck_con, local_path, tbl["file_type"], tname,
            )
            table_id_map[tname] = tid

            # ----------------------------------------------------------
            # 2. Extract schema → push to table_schemas
            # ----------------------------------------------------------
            columns = extract_column_schemas(duck_con, tname)
            save_table_schemas(pg_conn, columns, tid, project_id)
            all_schemas[tname] = columns

            update_uploaded_table_status(pg_conn, tid, "ready", row_count=row_count)
            logger.info("Table %s ingested (%d rows, %d columns)", tname, row_count, len(columns))

        except Exception:
            logger.exception("Failed to ingest table %s", tname)
            update_uploaded_table_status(
                pg_conn, tid, "failed", error_details=str(Exception)
            )

    # LangChain DDL for prompts and LLM strategy
    langchain_ddl = get_langchain_table_info(duckdb_path)

    # ------------------------------------------------------------------
    # 3. Relationship detection (three strategies → combine → save)
    #    User-defined relationships are NEVER deleted or overridden.
    # ------------------------------------------------------------------
    cleanup_existing_relationships(pg_conn, project_id)
    user_rels = fetch_user_relationships(pg_conn, project_id, table_id_map)

    strategy_results: list[list[DetectedRelationship]] = []
    for strategy_fn, extra_args in [
        (detect_relationships_name_match, (all_schemas, table_id_map)),
        (detect_relationships_value_overlap, (duck_con, all_schemas, table_id_map)),
        (detect_relationships_llm, (langchain_ddl, all_schemas, table_id_map)),
    ]:
        try:
            strategy_results.append(strategy_fn(*extra_args))
        except Exception:
            logger.exception("Relationship strategy %s failed", strategy_fn.__name__)

    relationships = combine_detected_relationships(
        *strategy_results,
        user_relationships=user_rels,
    )
    if relationships:
        save_relationships(pg_conn, relationships, project_id, table_id_map)

    # ------------------------------------------------------------------
    # 4. Document generation via LLM
    # ------------------------------------------------------------------
    cleanup_existing_documents(pg_conn, project_id)
    llm = _get_llm(model_name=llm_model)

    documents = generate_all_documents(llm, all_schemas, relationships, langchain_ddl)

    # ------------------------------------------------------------------
    # 5. Chunking + embedding → save to document_chunks
    # ------------------------------------------------------------------
    total_chunks = 0
    for doc in documents:
        uploaded_table_id = table_id_map.get(doc.table_name) if doc.table_name else None
        doc_id = save_schema_document(
            pg_conn, doc, project_id, uploaded_table_id, llm_model,
        )
        chunks = chunk_and_embed_document(doc, embedding_model=embedding_model)
        if chunks:
            save_document_chunks(pg_conn, chunks, doc_id, project_id)
            total_chunks += len(chunks)

    duck_con.close()
    pg_conn.close()

    summary = {
        "status": "success",
        "tables_loaded": len(table_id_map),
        "schemas_saved": sum(len(v) for v in all_schemas.values()),
        "relationships_detected": len(relationships),
        "documents_generated": len(documents),
        "chunks_embedded": total_chunks,
    }
    logger.info("Ingestion complete for project %s: %s", project_id, summary)
    return summary
