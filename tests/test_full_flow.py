"""
Full end-to-end test: ingestion → retrieval → agent.

Requires:
  - PostgreSQL with pgvector running (see DATABASE_URL in .env)
  - OPENAI_API_KEY in .env
  - database.txt schema already applied to Postgres

Run:
    poetry run python tests/test_full_flow.py
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
        combine_detected_relationships,
        save_table_schemas,
        save_relationships,
        get_pg_connection,
        save_schema_document,
        save_document_chunks,
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
        INSERT INTO project_settings (project_id)
        VALUES ('{PROJECT_ID}')
        ON CONFLICT (project_id) DO NOTHING
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
    for tname in table_names:
        rows = load_file_to_duckdb(con, csv_paths[tname], "csv", tname)
        print(f"  Loaded {tname}: {rows} rows")
    print()

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 3: Extract schemas and save to Postgres")
    print("=" * 60)

    # Clean up previous test data for this project
    cur.execute(f"DELETE FROM schema_documents WHERE project_id = '{PROJECT_ID}'")
    cur.execute(f"DELETE FROM table_relationships WHERE project_id = '{PROJECT_ID}'")
    cur.execute(f"DELETE FROM table_schemas WHERE project_id = '{PROJECT_ID}'")
    cur.execute(f"DELETE FROM uploaded_tables WHERE project_id = '{PROJECT_ID}'")
    pg.commit()

    schemas: dict = {}
    table_id_map: dict[str, str] = {}

    for tname in table_names:
        cols = extract_column_schemas(con, tname)
        schemas[tname] = cols

        cur.execute(
            """
            INSERT INTO uploaded_tables
                (project_id, clerk_id, original_filename, s3_key,
                 file_size, file_type, table_name, processing_status)
            VALUES (%s, 'test_user', %s, 'local', 0, 'csv', %s, 'ready')
            RETURNING id
            """,
            (PROJECT_ID, f"{tname}.csv", tname),
        )
        tid = str(cur.fetchone()[0])
        pg.commit()
        table_id_map[tname] = tid
        save_table_schemas(pg, cols, tid, PROJECT_ID)
        print(f"  {tname}: {len(cols)} columns saved")

    print()

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 4: Detect relationships")
    print("=" * 60)
    name_rels = detect_relationships_name_match(schemas, table_id_map)
    print(f"  Name-match strategy: {len(name_rels)} relationships")

    value_rels = detect_relationships_value_overlap(con, schemas, table_id_map, min_distinct=2)
    print(f"  Value-overlap strategy: {len(value_rels)} relationships")

    combined = combine_detected_relationships(name_rels, value_rels)
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

    # Build a simple DDL string from DuckDB directly (avoids LangChain SQLAlchemy compat issue)
    ddl_parts = []
    for tname in table_names:
        col_info = _format_column_details(schemas[tname])
        ddl_parts.append(f"## Table: {tname}\n{col_info}")
    ddl = "\n\n".join(ddl_parts)

    llm = _get_llm()
    docs = generate_all_documents(llm, schemas, combined, ddl)
    print(f"  Generated {len(docs)} documents:")
    for doc in docs:
        print(f"    - [{doc.doc_type}] {doc.title}  ({len(doc.content)} chars)")
    print()

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 6: Chunk + embed + save (calling OpenAI embeddings...)")
    print("=" * 60)
    total_chunks = 0
    for doc in docs:
        uploaded_table_id = table_id_map.get(doc.table_name)
        doc_id = save_schema_document(pg, doc, PROJECT_ID, uploaded_table_id, "gpt-4o")
        chunks = chunk_and_embed_document(doc)
        if chunks:
            save_document_chunks(pg, chunks, doc_id, PROJECT_ID)
            total_chunks += len(chunks)
    print(f"  Saved {total_chunks} chunks with embeddings.")
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

