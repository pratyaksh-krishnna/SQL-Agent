# SQL Agent

A **natural-language-to-SQL** agent that uses RAG (retrieval-augmented generation) and a LangGraph pipeline to answer questions over your data. Upload tables (CSV, etc.), get schema-aware documentation and embeddings, then ask questions in plain English and get SQL plus answers.

## What it does

- **Ingestion**: Loads your data (e.g. CSV) into DuckDB, extracts column schemas, detects relationships between tables, and generates natural-language documentation. Chunks are embedded (OpenAI) and stored in PostgreSQL with pgvector.
- **Retrieval**: For each user question, the system retrieves the most relevant schema chunks (vector + keyword search) to build context for the LLM.
- **Agent**: A LangGraph-based agent with guardrails that (1) retrieves schema context, (2) generates SQL, (3) validates and optionally retries on failure, (4) runs the query against DuckDB, and (5) returns a natural-language answer.

So: **upload data → automatic schema + docs + embeddings → ask questions in English → get SQL and answers.**

## Prerequisites

- **Python 3.14** (see `pyproject.toml`)
- **PostgreSQL** with the **pgvector** extension (for schemas, chunks, and embeddings)
- **OpenAI API key** (for embeddings and the LLM)
- **AWS credentials** (optional; only needed if you use S3 for file uploads)

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/SQL-AGENT-MAIN-v1.git
cd SQL-AGENT-MAIN-v1
poetry install
```

### 2. Environment variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (e.g. `postgresql://user:password@localhost:5432/sql_agent`) |
| `OPENAI_API_KEY` | Your OpenAI API key (used for embeddings and the SQL/answer LLM) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `S3_BUCKET_NAME` | Only required if you use S3 for uploaded files |

### 3. Database schema

Create your PostgreSQL database and enable pgvector, then run the migration:

```bash
psql -d your_database -f database.txt
```

This creates tables for users, projects, uploaded tables, table schemas, relationships, schema documents, and document chunks (with vector embeddings).

## Running

### Full flow test (ingestion → retrieval → agent)

Runs a full pipeline: seed a test project, load sample CSVs into DuckDB, extract schemas, detect relationships, generate and embed documents, run RAG retrieval, and call the LangGraph agent.

```bash
PYTHONPATH=. poetry run python tests/test_full_flow.py
```

Requires:

- PostgreSQL with the schema from `database.txt` applied
- `OPENAI_API_KEY` and `DATABASE_URL` set in `.env`

### Using the agent in code

```python
from agent.simple_agent import run_agent

result = run_agent(
    project_id="<your-project-uuid>",
    duckdb_path="/path/to/project.duckdb",
    user_query="Show me the top 3 customers by total order amount",
)

print(result["sql"])    # Generated SQL
print(result["answer"]) # Natural-language answer
print(result["retry_count"])
```

The agent uses RAG to pull in relevant schema context for the project, then generates and executes SQL against the DuckDB file (with guardrails and retries).

### Ingestion pipeline

The ingestion pipeline (see `rag/ingestion/index.py`) is intended to be run when new files are uploaded: it loads files (from S3 or local), extracts schemas, detects relationships, generates documents, chunks and embeds them, and saves everything to PostgreSQL. The test flow in `tests/test_full_flow.py` demonstrates a minimal version of this.

## Project structure

```
.
├── agent/
│   └── simple_agent.py    # LangGraph SQL agent (retrieve → generate_sql → execute → summarize)
├── rag/
│   ├── ingestion/        # Load data, extract schemas, detect relationships, generate & embed docs
│   │   ├── index.py      # Main pipeline entry (run_ingestion_pipeline)
│   │   └── utils.py      # DuckDB, Postgres, S3, embedding helpers
│   └── retrieval/        # RAG retrieval for schema context
│       ├── index.py      # retrieve_context(project_id, query)
│       └── utils.py      # Vector + keyword search, reranking
├── database.txt          # PostgreSQL schema (pgvector, tables for projects, schemas, chunks)
├── tests/
│   └── test_full_flow.py # End-to-end: ingestion → retrieval → agent
├── .env.example          # Template for DATABASE_URL, OPENAI_API_KEY, AWS (optional)
└── pyproject.toml        # Python 3.14, Poetry, dependencies
```

## Tech stack

- **LangGraph** – agent graph (retrieve → generate SQL → validate → execute → summarize)
- **LangChain / LangChain OpenAI** – LLM and embeddings
- **DuckDB** – in-process SQL over CSV (and other) data
- **PostgreSQL + pgvector** – storage for schemas, relationships, document chunks, and vector search
- **FastAPI** – listed as a dependency for API use (see `rag/retreival` for HTTP-oriented code)


