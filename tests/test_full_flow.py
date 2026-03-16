"""
Full end-to-end test: ingestion → retrieval → agent.

Every step uses real pipeline functions. The only manual inputs allowed are:
  - The sample table data (CUSTOMERS_CSV, ORDERS_CSV, PRODUCTS_CSV) at the top.
  - Test harness: one project + uploaded_tables rows (no "create project" or
    "register table" API in this repo; pipeline expects these to exist).

Functions exercised: get_pg_connection, create_duckdb_connection, load_file_to_duckdb,
extract_column_schemas, save_table_schemas, update_uploaded_table_status,
cleanup_existing_documents, cleanup_existing_relationships, _format_column_details,
fetch_user_relationships, detect_relationships_name_match, detect_relationships_value_overlap,
detect_relationships_llm, combine_detected_relationships, save_relationships,
fetch_project_settings, _get_llm, generate_all_documents, chunk_and_embed_document,
save_schema_document, save_document_chunks, retrieve_context, run_agent.
(DDL is built from schemas to avoid a second DuckDB connection to the same file.)

Requires:
  - PostgreSQL with pgvector running (see DATABASE_URL in .env)
  - OPENAI_API_KEY in .env
  - database.txt schema already applied to Postgres

Run:
    PYTHONPATH=. poetry run python tests/test_full_flow.py
"""

import os
import tempfile
import textwrap

from dotenv import load_dotenv

load_dotenv()

# ── Sample data (created inline — no external files needed) ──

CUSTOMERS_CSV = textwrap.dedent("""\
    id,name,email,city
    1,Alice,alice@example.com,New York
    2,Bob,bob@example.com,San Francisco
    3,Charlie,charlie@example.com,Chicago
    4,Diana,diana@example.com,Boston
    5,Eve,eve@example.com,Seattle
""")

ORDERS_CSV = textwrap.dedent("""\
    id,customer_id,product,amount,order_date
    1,1,Widget A,29.99,2024-01-15
    2,1,Widget B,49.99,2024-02-01
    3,2,Widget A,29.99,2024-01-20
    4,3,Widget C,19.99,2024-02-10
    5,4,Widget B,49.99,2024-02-15
    6,5,Widget A,29.99,2024-03-01
    7,2,Widget C,19.99,2024-03-05
    8,1,Widget A,29.99,2024-03-10
""")

PRODUCTS_CSV = textwrap.dedent("""\
    id,name,category,price
    1,Widget A,Gadgets,29.99
    2,Widget B,Gadgets,49.99
    3,Widget C,Tools,19.99
""")


def write_sample_csvs(tmp_dir: str) -> dict[str, str]:
    paths = {}
    for name, content in [("customers", CUSTOMERS_CSV), ("orders", ORDERS_CSV), ("products", PRODUCTS_CSV)]:
        path = os.path.join(tmp_dir, f"{name}.csv")
        with open(path, "w") as f:
            f.write(content)
        paths[name] = path
    return paths


def main():
    from rag.ingestion.utils import (
        create_duckdb_connection,
        load_file_to_duckdb,
        extract_column_schemas,
        detect_relationships_name_match,
        detect_relationships_value_overlap,
        detect_relationships_llm,
        combine_detected_relationships,
        save_table_schemas,
        save_relationships,
        save_schema_document,
        save_document_chunks,
        get_pg_connection,
        cleanup_existing_documents,
        cleanup_existing_relationships,
        fetch_project_settings,
        fetch_user_relationships,
        update_uploaded_table_status,
    )
    from rag.ingestion.index import (
        generate_all_documents,
        chunk_and_embed_document,
        _get_llm,
        _format_column_details,
    )
    from rag.retrieval.index import retrieve_context
    from agent.simple_agent import run_agent

    PROJECT_ID = "11111111-1111-1111-1111-111111111111"

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 1: Seed Postgres with test project")
    print("=" * 60)
    pg = get_pg_connection()
    cur = pg.cursor()
    cur.execute("INSERT INTO users (clerk_id) VALUES ('test_user') ON CONFLICT DO NOTHING")
    cur.execute(f"""
        INSERT INTO projects (id, name, clerk_id)
        VALUES ('{PROJECT_ID}', 'Test Project', 'test_user')
        ON CONFLICT (id) DO NOTHING
    """)
    cur.execute(f"""
        INSERT INTO project_settings (project_id, similarity_threshold)
        VALUES ('{PROJECT_ID}', 0.3)
        ON CONFLICT (project_id) DO UPDATE SET similarity_threshold = 0.3
    """)
    pg.commit()
    print("  Project seeded.\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 2: Create sample CSVs and load into DuckDB")
    print("=" * 60)
    tmp_dir = tempfile.mkdtemp(prefix="sql_agent_test_")
    csv_paths = write_sample_csvs(tmp_dir)

    db_path = os.path.join(tmp_dir, "test.duckdb")
    con = create_duckdb_connection(db_path)

    table_names = list(csv_paths.keys())
    table_row_counts: dict[str, int] = {}
    for tname in table_names:
        rows = load_file_to_duckdb(con, csv_paths[tname], "csv", tname)
        table_row_counts[tname] = rows
        print(f"  Loaded {tname}: {rows} rows")
    print()

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 3: Extract schemas and save to Postgres")
    print("=" * 60)

    cleanup_existing_documents(pg, PROJECT_ID)
    cleanup_existing_relationships(pg, PROJECT_ID)
    cur.execute("DELETE FROM table_schemas WHERE project_id = %s", (PROJECT_ID,))
    cur.execute("DELETE FROM uploaded_tables WHERE project_id = %s", (PROJECT_ID,))
    pg.commit()

    schemas: dict = {}
    table_id_map: dict[str, str] = {}

    for tname in table_names:
        cur.execute(
            """
            INSERT INTO uploaded_tables
                (project_id, clerk_id, original_filename, s3_key,
                 file_size, file_type, table_name, processing_status)
            VALUES (%s, 'test_user', %s, 'local', 0, 'csv', %s, 'processing')
            RETURNING id
            """,
            (PROJECT_ID, f"{tname}.csv", tname),
        )
        tid = str(cur.fetchone()[0])
        pg.commit()
        table_id_map[tname] = tid

        cols = extract_column_schemas(con, tname)
        schemas[tname] = cols
        save_table_schemas(pg, cols, tid, PROJECT_ID)
        update_uploaded_table_status(
            pg, tid, "ready", row_count=table_row_counts[tname]
        )
        print(f"  {tname}: {len(cols)} columns saved")

    print()

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 4: Detect relationships")
    print("=" * 60)
    # Build DDL from existing schemas to avoid opening a second DuckDB connection
    # (DuckDB forbids same file open with different connection configs).
    ddl_parts = [
        f"## Table: {tname}\n{_format_column_details(schemas[tname])}"
        for tname in table_names
    ]
    ddl = "\n\n".join(ddl_parts)
    user_rels = fetch_user_relationships(pg, PROJECT_ID, table_id_map)

    name_rels = detect_relationships_name_match(schemas, table_id_map)
    print(f"  Name-match strategy: {len(name_rels)} relationships")

    value_rels = detect_relationships_value_overlap(con, schemas, table_id_map, min_distinct=2)
    print(f"  Value-overlap strategy: {len(value_rels)} relationships")

    llm_rels = detect_relationships_llm(ddl, schemas, table_id_map)
    print(f"  LLM strategy: {len(llm_rels)} relationships")

    combined = combine_detected_relationships(
        name_rels, value_rels, llm_rels, user_relationships=user_rels
    )
    print(f"  Combined (deduped): {len(combined)} relationships")

    if combined:
        save_relationships(pg, combined, PROJECT_ID, table_id_map)
        for rel in combined:
            print(
                f"    {rel.from_table_name}.{rel.from_column} -> "
                f"{rel.to_table_name}.{rel.to_column}  "
                f"(conf={rel.confidence:.2f}, method={rel.detection_method})"
            )
    print()

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 5: Generate documents via LLM (calling OpenAI...)")
    print("=" * 60)

    settings = fetch_project_settings(pg, PROJECT_ID)
    llm_model = settings.get("llm_model", "gpt-4o")
    llm = _get_llm(model_name=llm_model)
    docs = generate_all_documents(llm, schemas, combined, ddl)
    print(f"  Generated {len(docs)} documents:")
    for doc in docs:
        print(f"    - [{doc.doc_type}] {doc.title}  ({len(doc.content)} chars)")

    docs_file = os.path.join(tmp_dir, "generated_documents.txt")
    with open(docs_file, "w") as f:
        for i, doc in enumerate(docs, 1):
            f.write(f"{'=' * 70}\n")
            f.write(f"Document {i}: [{doc.doc_type}] {doc.title}\n")
            f.write(f"Table: {doc.table_name or '(global)'}\n")
            f.write(f"{'=' * 70}\n")
            f.write(doc.content)
            f.write("\n\n")
    print(f"  Documents written to: {docs_file}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 6: Chunk + embed + save (calling OpenAI embeddings...)")
    print("=" * 60)
    embedding_model = settings.get("embedding_model", "text-embedding-3-small")
    total_chunks = 0
    all_chunks: list = []
    for doc in docs:
        uploaded_table_id = table_id_map.get(doc.table_name)
        doc_id = save_schema_document(
            pg, doc, PROJECT_ID, uploaded_table_id, llm_model
        )
        chunks = chunk_and_embed_document(doc, embedding_model=embedding_model)
        if chunks:
            save_document_chunks(pg, chunks, doc_id, PROJECT_ID)
            all_chunks.extend((doc, ch) for ch in chunks)
            total_chunks += len(chunks)

    chunks_file = os.path.join(tmp_dir, "generated_chunks.txt")
    with open(chunks_file, "w") as f:
        for i, (doc, ch) in enumerate(all_chunks, 1):
            f.write(f"{'=' * 70}\n")
            f.write(f"Chunk {i} (index={ch.chunk_index}, doc_type={ch.doc_type})\n")
            f.write(f"Source: [{doc.doc_type}] {doc.title}\n")
            f.write(f"Tables: {ch.table_names}\n")
            f.write(f"Embedding dims: {len(ch.embedding)}\n")
            f.write(f"{'=' * 70}\n")
            f.write(ch.content)
            f.write("\n\n")
    print(f"  Saved {total_chunks} chunks with embeddings.")
    print(f"  Chunks written to: {chunks_file}")
    print("  Ingestion complete!\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 7: Test RAG retrieval")
    print("=" * 60)
    test_queries = [
        "How many orders per customer?",
        "What are the product categories?",
        "Show customer cities",
    ]
    for q in test_queries:
        result = retrieve_context(PROJECT_ID, q)
        print(f"  Query: '{q}'")
        print(f"    Retrieved {len(result.chunk_ids)} chunks ({result.total_chunks_searched} searched)")
        if result.context:
            preview = result.context[:200].replace("\n", " ")
            print(f"    Preview: {preview}...")
        print()

    # Close DuckDB so the agent can open its own read-only connection
    con.close()

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 8: Run the full LangGraph agent!")
    print("=" * 60)
    user_questions = [
        "Show me the top 3 customers by total order amount",
        "How many orders were placed each month?",
    ]

    for question in user_questions:
        print(f"\n  Question: {question}")
        print("  " + "-" * 50)
        answer = run_agent(
            project_id=PROJECT_ID,
            duckdb_path=db_path,
            user_query=question,
        )
        print(f"  SQL: {answer['sql']}")
        print(f"  Retries: {answer['retry_count']}")
        print(f"  Answer:\n{answer['answer']}")
        print()

    # ── Cleanup ──
    pg.close()
    print("=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

