"""
Lower-level utilities for the RAG ingestion pipeline.

Responsibilities:
  - Data transfer objects (dataclasses)
  - S3 download
  - DuckDB file loading + schema introspection
  - PostgreSQL CRUD helpers
  - Relationship detection strategy stubs (3 strategies)
  - Relationship deduplication / merge
  - Text chunking
  - Embedding via OpenAI
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

import boto3
import duckdb
import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColumnSchema:
    column_name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    column_index: int
    sample_values: list[str]
    description: Optional[str] = None
    is_foreign_key: bool = False


@dataclass
class DetectedRelationship:
    from_table_name: str
    from_column: str
    to_table_name: str
    to_column: str
    relationship_type: str      # many_to_one | one_to_one | many_to_many
    confidence: float           # 0.0 – 1.0
    detection_method: str       # auto_name_match | auto_value_overlap | auto_llm


@dataclass
class SchemaDocument:
    doc_type: str               # table_overview | column_detail | relationship | join_guide
    title: str
    content: str
    table_name: Optional[str] = None


@dataclass
class DocumentChunk:
    content: str
    chunk_index: int
    doc_type: str
    table_names: list[str]
    embedding: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def download_from_s3(
    s3_key: str,
    bucket_name: str | None = None,
    local_dir: str | None = None,
) -> str:
    """Download *s3_key* from *bucket_name* and return the local file path."""
    bucket_name = bucket_name or os.environ["S3_BUCKET_NAME"]
    s3 = boto3.client("s3")
    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix="sql_agent_")
    local_path = os.path.join(local_dir, os.path.basename(s3_key))
    s3.download_file(bucket_name, s3_key, local_path)
    logger.info("Downloaded s3://%s/%s → %s", bucket_name, s3_key, local_path)
    return local_path


# ---------------------------------------------------------------------------
# DuckDB — loading files
# ---------------------------------------------------------------------------

def create_duckdb_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection (file-backed or in-memory)."""
    if db_path is None:
        fd, db_path = tempfile.mkstemp(suffix=".duckdb", prefix="sql_agent_")
        os.close(fd)
    return duckdb.connect(db_path)


def load_file_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    file_path: str,
    file_type: str,
    table_name: str,
) -> int:
    """
    Load a local file into a DuckDB table.

    Supported *file_type* values: csv, json, xlsx, sql.
    Returns the row count of the newly created table.
    """
    ft = file_type.lower()
    safe_name = f'"{table_name}"'

    if ft == "csv":
        con.execute(
            f"CREATE TABLE {safe_name} AS SELECT * FROM read_csv_auto(?)",
            [file_path],
        )
    elif ft == "json":
        con.execute(
            f"CREATE TABLE {safe_name} AS SELECT * FROM read_json_auto(?)",
            [file_path],
        )
    elif ft == "xlsx":
        con.install_extension("spatial")
        con.load_extension("spatial")
        con.execute(
            f"CREATE TABLE {safe_name} AS SELECT * FROM st_read(?)",
            [file_path],
        )
    elif ft == "sql":
        with open(file_path, "r") as fh:
            con.execute(fh.read())
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    row_count: int = con.execute(f"SELECT COUNT(*) FROM {safe_name}").fetchone()[0]
    logger.info("Loaded %s into DuckDB table %s (%d rows)", file_path, table_name, row_count)
    return row_count


# ---------------------------------------------------------------------------
# Schema extraction (DuckDB + LangChain)
# ---------------------------------------------------------------------------

def extract_column_schemas(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    sample_size: int = 5,
) -> list[ColumnSchema]:
    """Pull structured column metadata from a DuckDB table."""
    cols = con.execute(
        """
        SELECT column_name, data_type, is_nullable, ordinal_position
        FROM   information_schema.columns
        WHERE  table_name = ?
        ORDER  BY ordinal_position
        """,
        [table_name],
    ).fetchall()

    pk_columns: set[str] = set()
    try:
        pks = con.execute(
            """
            SELECT column_name
            FROM   information_schema.table_constraints tc
            JOIN   information_schema.key_column_usage kcu
                   ON tc.constraint_name = kcu.constraint_name
            WHERE  tc.table_name = ? AND tc.constraint_type = 'PRIMARY KEY'
            """,
            [table_name],
        ).fetchall()
        pk_columns = {r[0] for r in pks}
    except Exception:
        pass

    schemas: list[ColumnSchema] = []
    safe_name = f'"{table_name}"'
    for col_name, dtype, nullable, ordinal in cols:
        try:
            samples = con.execute(
                f'SELECT DISTINCT "{col_name}" FROM {safe_name} '
                f'WHERE "{col_name}" IS NOT NULL LIMIT ?',
                [sample_size],
            ).fetchall()
            sample_values = [str(s[0]) for s in samples]
        except Exception:
            sample_values = []

        schemas.append(
            ColumnSchema(
                column_name=col_name,
                data_type=dtype,
                is_nullable=(nullable == "YES"),
                is_primary_key=(col_name in pk_columns),
                column_index=ordinal - 1,
                sample_values=sample_values,
            )
        )
    return schemas


def get_langchain_table_info(duckdb_path: str) -> str:
    """Use LangChain SQLDatabase.get_table_info() for a human-readable DDL dump."""
    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_uri(f"duckdb:///{duckdb_path}")
    return db.get_table_info()


# ---------------------------------------------------------------------------
# PostgreSQL helpers
# ---------------------------------------------------------------------------

def get_pg_connection():
    """Return a psycopg2 connection using DATABASE_URL."""
    return psycopg2.connect(os.environ["DATABASE_URL"])


def fetch_uploaded_tables(pg_conn, project_id: str) -> list[dict]:
    """Fetch all uploaded_tables for a project."""
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT id, original_filename, s3_key, file_type, table_name "
            "FROM uploaded_tables WHERE project_id = %s ORDER BY created_at",
            (project_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_project_settings(pg_conn, project_id: str) -> dict:
    """Return project_settings row as a dict (or sensible defaults)."""
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM project_settings WHERE project_id = %s", (project_id,)
        )
        if cur.description is None:
            return {}
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        return dict(zip(cols, row)) if row else {}


def update_uploaded_table_status(
    pg_conn,
    uploaded_table_id: str,
    status: str,
    row_count: int | None = None,
    error_details: str | None = None,
):
    """Set processing_status (and optionally row_count / error) on an uploaded table."""
    details = json.dumps({"error": error_details}) if error_details else "{}"
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            UPDATE uploaded_tables
            SET    processing_status  = %s,
                   row_count          = COALESCE(%s, row_count),
                   processing_details = %s::json
            WHERE  id = %s
            """,
            (status, row_count, details, uploaded_table_id),
        )
    pg_conn.commit()


def save_table_schemas(
    pg_conn,
    schemas: list[ColumnSchema],
    uploaded_table_id: str,
    project_id: str,
) -> list[str]:
    """Insert rows into table_schemas. Returns list of new UUIDs."""
    ids: list[str] = []
    with pg_conn.cursor() as cur:
        for s in schemas:
            cur.execute(
                """
                INSERT INTO table_schemas
                    (uploaded_table_id, project_id, column_name, data_type,
                     is_nullable, is_primary_key, is_foreign_key,
                     sample_values, column_index, description)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (
                    uploaded_table_id, project_id,
                    s.column_name, s.data_type,
                    s.is_nullable, s.is_primary_key, s.is_foreign_key,
                    Json(s.sample_values), s.column_index, s.description,
                ),
            )
            ids.append(str(cur.fetchone()[0]))
    pg_conn.commit()
    logger.info("Saved %d column schemas for table %s", len(ids), uploaded_table_id)
    return ids


def save_relationships(
    pg_conn,
    relationships: list[DetectedRelationship],
    project_id: str,
    table_id_map: dict[str, str],
) -> list[str]:
    """Insert rows into table_relationships. Returns new UUIDs."""
    ids: list[str] = []
    with pg_conn.cursor() as cur:
        for rel in relationships:
            cur.execute(
                """
                INSERT INTO table_relationships
                    (project_id, from_table_id, from_column,
                     to_table_id, to_column, relationship_type,
                     confidence, detection_method)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (
                    project_id,
                    table_id_map[rel.from_table_name], rel.from_column,
                    table_id_map[rel.to_table_name], rel.to_column,
                    rel.relationship_type, rel.confidence, rel.detection_method,
                ),
            )
            ids.append(str(cur.fetchone()[0]))
    pg_conn.commit()
    logger.info("Saved %d relationships for project %s", len(ids), project_id)
    return ids


def save_schema_document(
    pg_conn,
    doc: SchemaDocument,
    project_id: str,
    uploaded_table_id: str | None,
    generation_model: str,
) -> str:
    """Insert a single schema_documents row. Returns the new UUID."""
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO schema_documents
                (project_id, uploaded_table_id, doc_type, title, content, generation_model)
            VALUES (%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (project_id, uploaded_table_id, doc.doc_type, doc.title, doc.content, generation_model),
        )
        doc_id = str(cur.fetchone()[0])
    pg_conn.commit()
    return doc_id


def save_document_chunks(
    pg_conn,
    chunks: list[DocumentChunk],
    document_id: str,
    project_id: str,
):
    """Batch-insert chunks (with embeddings) into document_chunks."""
    with pg_conn.cursor() as cur:
        for ch in chunks:
            cur.execute(
                """
                INSERT INTO document_chunks
                    (document_id, project_id, content, chunk_index,
                     char_count, doc_type, table_names, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)
                """,
                (
                    document_id, project_id, ch.content, ch.chunk_index,
                    len(ch.content), ch.doc_type, ch.table_names, str(ch.embedding),
                ),
            )
    pg_conn.commit()
    logger.info("Saved %d chunks for document %s", len(chunks), document_id)


def cleanup_existing_documents(pg_conn, project_id: str):
    """Remove old schema_documents (cascades to document_chunks) before regeneration."""
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM schema_documents WHERE project_id = %s", (project_id,))
    pg_conn.commit()


def cleanup_existing_relationships(pg_conn, project_id: str):
    """Remove old auto-detected relationships before re-detection."""
    with pg_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM table_relationships WHERE project_id = %s AND detection_method != 'user'",
            (project_id,),
        )
    pg_conn.commit()


def fetch_user_relationships(
    pg_conn,
    project_id: str,
    table_id_map: dict[str, str],
) -> list[DetectedRelationship]:
    """
    Load user-confirmed relationships from the DB.

    These are the ground-truth — auto-detected relationships must never
    override them.
    """
    id_to_name = {v: k for k, v in table_id_map.items()}
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT from_table_id, from_column, to_table_id, to_column,
                   relationship_type, confidence
            FROM   table_relationships
            WHERE  project_id = %s AND detection_method = 'user'
            """,
            (project_id,),
        )
        rows = cur.fetchall()

    results: list[DetectedRelationship] = []
    for from_tid, from_col, to_tid, to_col, rel_type, conf in rows:
        from_name = id_to_name.get(str(from_tid))
        to_name = id_to_name.get(str(to_tid))
        if from_name and to_name:
            results.append(DetectedRelationship(
                from_table_name=from_name,
                from_column=from_col,
                to_table_name=to_name,
                to_column=to_col,
                relationship_type=rel_type,
                confidence=float(conf),
                detection_method="user",
            ))
    return results


# ---------------------------------------------------------------------------
# Relationship detection — three strategies
# ---------------------------------------------------------------------------

# -- Type-compatibility helpers for strategy 2 --

_INTEGER_TYPES = frozenset({
    "INTEGER", "BIGINT", "SMALLINT", "INT", "INT4", "INT8", "INT2",
    "TINYINT", "HUGEINT", "UBIGINT", "UINTEGER", "USMALLINT", "UTINYINT",
})
_STRING_TYPES = frozenset({
    "VARCHAR", "TEXT", "CHAR", "STRING", "NVARCHAR", "BPCHAR",
})
_SKIP_TYPES = frozenset({
    "BOOLEAN", "BOOL", "BLOB", "BYTEA", "DATE", "TIME",
    "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE",
    "INTERVAL",
})


def _base_type(dtype: str) -> str:
    """Strip length / precision qualifiers: VARCHAR(255) → VARCHAR."""
    return dtype.upper().split("(")[0].strip()


def _types_compatible(t1: str, t2: str) -> bool:
    a, b = _base_type(t1), _base_type(t2)
    if a == b:
        return True
    if a in _INTEGER_TYPES and b in _INTEGER_TYPES:
        return True
    if a in _STRING_TYPES and b in _STRING_TYPES:
        return True
    return False


# -- Plural / singular helpers for strategy 1 --

def _singular_forms(name: str) -> list[str]:
    """Return plausible singular forms of *name* (lowercase)."""
    low = name.lower()
    forms = [low]
    if low.endswith("ies"):
        forms.append(low[:-3] + "y")
    elif low.endswith("ses") or low.endswith("xes") or low.endswith("zes"):
        forms.append(low[:-2])
    elif low.endswith("s") and not low.endswith("ss"):
        forms.append(low[:-1])
    return forms


def _build_table_name_lookup(
    table_names: set[str],
) -> dict[str, str]:
    """Map every plausible singular/plural/lower form → actual table name."""
    lookup: dict[str, str] = {}
    for t in table_names:
        lookup[t.lower()] = t
        for form in _singular_forms(t):
            lookup.setdefault(form, t)
    return lookup


def _camel_to_parts(name: str) -> list[str]:
    """Split a camelCase or PascalCase identifier into lowercase parts.

    >>> _camel_to_parts("userId")
    ['user', 'id']
    >>> _camel_to_parts("orderItemId")
    ['order', 'item', 'id']
    """
    return [m.lower() for m in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)", name)]


# ---- Strategy 1: name matching ----

def detect_relationships_name_match(
    table_schemas: dict[str, list[ColumnSchema]],
    table_id_map: dict[str, str],
) -> list[DetectedRelationship]:
    """
    **Strategy 1 — Column-name matching.**

    Patterns detected (highest → lowest confidence):
    1. ``<table>_id``  →  ``<table>.id``           (0.90)
    2. camelCase ``<Table>Id`` → ``<table>.id``     (0.85)
    3. Exact column name in table A matches a PK
       column name in table B                       (0.70)
    """
    table_names = set(table_schemas.keys())
    lookup = _build_table_name_lookup(table_names)
    relationships: list[DetectedRelationship] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    def _add(from_t: str, from_c: str, to_t: str, to_c: str, conf: float):
        key = (from_t, from_c, to_t, to_c)
        rev = (to_t, to_c, from_t, from_c)
        if key in seen_pairs or rev in seen_pairs:
            return
        seen_pairs.add(key)
        relationships.append(DetectedRelationship(
            from_table_name=from_t,
            from_column=from_c,
            to_table_name=to_t,
            to_column=to_c,
            relationship_type="many_to_one",
            confidence=conf,
            detection_method="auto_name_match",
        ))

    def _target_has_column(target_table: str, col_name: str) -> str | None:
        """Return the actual column name if *target_table* has *col_name* (case-insensitive)."""
        for c in table_schemas.get(target_table, []):
            if c.column_name.lower() == col_name.lower():
                return c.column_name
        return None

    for from_table, columns in table_schemas.items():
        for col in columns:
            cname = col.column_name
            clow = cname.lower()

            # Pattern 1: snake_case  <ref>_id
            if clow.endswith("_id") and len(clow) > 3:
                ref_part = clow[:-3]                       # e.g. "customer"
                target = lookup.get(ref_part)
                if target and target != from_table:
                    actual_col = _target_has_column(target, "id")
                    if actual_col:
                        _add(from_table, cname, target, actual_col, 0.90)
                        continue

            # Pattern 2: camelCase  <Ref>Id
            parts = _camel_to_parts(cname)
            if len(parts) >= 2 and parts[-1] == "id":
                ref_part = "_".join(parts[:-1])            # e.g. "order_item"
                target = lookup.get(ref_part)
                if target and target != from_table:
                    actual_col = _target_has_column(target, "id")
                    if actual_col:
                        _add(from_table, cname, target, actual_col, 0.85)
                        continue

            # Pattern 3: exact column name matches a PK in another table
            if not col.is_primary_key:
                for other_table, other_cols in table_schemas.items():
                    if other_table == from_table:
                        continue
                    for oc in other_cols:
                        if oc.is_primary_key and oc.column_name.lower() == clow:
                            _add(from_table, cname, other_table, oc.column_name, 0.70)

    logger.info("Name-match strategy found %d relationships", len(relationships))
    return relationships


# ---- Strategy 2: value overlap ----

def detect_relationships_value_overlap(
    con: duckdb.DuckDBPyConnection,
    table_schemas: dict[str, list[ColumnSchema]],
    table_id_map: dict[str, str],
    sample_limit: int = 10_000,
    containment_threshold: float = 0.70,
    min_distinct: int = 5,
) -> list[DetectedRelationship]:
    """
    **Strategy 2 — Statistical value overlap.**

    For every cross-table column pair with compatible types, sample up to
    *sample_limit* distinct values and compute the **containment ratio**::

        containment(A → B) = |A ∩ B| / |A|

    If containment(A → B) ≥ *containment_threshold*, column A likely
    references column B (i.e. A is the FK side).
    """
    ColKey = tuple[str, str]  # (table_name, column_name)

    col_values: dict[ColKey, set[str]] = {}
    col_distinct_count: dict[ColKey, int] = {}
    col_dtype: dict[ColKey, str] = {}
    col_is_pk: dict[ColKey, bool] = {}

    for tname, columns in table_schemas.items():
        safe_t = f'"{tname}"'
        for col in columns:
            bt = _base_type(col.data_type)
            if bt in _SKIP_TYPES:
                continue

            key: ColKey = (tname, col.column_name)
            safe_c = f'"{col.column_name}"'
            try:
                dc = con.execute(
                    f"SELECT COUNT(DISTINCT {safe_c}) FROM {safe_t}"
                ).fetchone()[0]
                if dc < min_distinct:
                    continue
                rows = con.execute(
                    f"SELECT DISTINCT CAST({safe_c} AS VARCHAR) FROM {safe_t} "
                    f"WHERE {safe_c} IS NOT NULL LIMIT {sample_limit}"
                ).fetchall()
                col_values[key] = {str(r[0]) for r in rows}
                col_distinct_count[key] = dc
                col_dtype[key] = col.data_type
                col_is_pk[key] = col.is_primary_key
            except Exception:
                continue

    relationships: list[DetectedRelationship] = []
    checked: set[tuple[ColKey, ColKey]] = set()
    keys = list(col_values.keys())

    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            t1, c1 = k1
            t2, c2 = k2
            if t1 == t2:
                continue
            pair = (k1, k2) if k1 < k2 else (k2, k1)
            if pair in checked:
                continue
            checked.add(pair)

            if not _types_compatible(col_dtype[k1], col_dtype[k2]):
                continue

            vals1, vals2 = col_values[k1], col_values[k2]
            intersection = vals1 & vals2
            if not intersection:
                continue

            # containment(A→B) = how much of A is found in B
            cont_1_in_2 = len(intersection) / len(vals1) if vals1 else 0.0
            cont_2_in_1 = len(intersection) / len(vals2) if vals2 else 0.0

            # The side whose values are almost entirely contained in the
            # other side is the FK (many) side.
            if cont_1_in_2 >= cont_2_in_1 and cont_1_in_2 >= containment_threshold:
                from_t, from_c = t1, c1   # FK side
                to_t, to_c = t2, c2       # PK side
                confidence = round(cont_1_in_2 * 0.80, 3)
            elif cont_2_in_1 >= containment_threshold:
                from_t, from_c = t2, c2
                to_t, to_c = t1, c1
                confidence = round(cont_2_in_1 * 0.80, 3)
            else:
                continue

            relationships.append(DetectedRelationship(
                from_table_name=from_t,
                from_column=from_c,
                to_table_name=to_t,
                to_column=to_c,
                relationship_type="many_to_one",
                confidence=min(confidence, 0.80),
                detection_method="auto_value_overlap",
            ))

    logger.info("Value-overlap strategy found %d relationships", len(relationships))
    return relationships


# ---- Strategy 3: LLM inference ----

_LLM_RELATIONSHIP_PROMPT = """\
You are an expert database architect.  Analyze the schema below and identify
all foreign-key / join relationships between tables.

### DDL
{ddl}

### Sample values per column
{samples}

Return **only** a JSON array.  Each element must have exactly these keys:
  "from_table"          – table containing the foreign key
  "from_column"         – the FK column name
  "to_table"            – the referenced (primary-key) table
  "to_column"           – the referenced column
  "relationship_type"   – one of "many_to_one", "one_to_one", "many_to_many"

If you find no relationships, return an empty array: []
Do NOT wrap the JSON in markdown fences or add commentary.
"""


def _format_samples_for_llm(
    table_schemas: dict[str, list[ColumnSchema]],
) -> str:
    lines: list[str] = []
    for tname, cols in table_schemas.items():
        lines.append(f"## {tname}")
        for c in cols:
            samples = ", ".join(c.sample_values[:5]) if c.sample_values else "—"
            lines.append(f"  {c.column_name} ({c.data_type}): {samples}")
        lines.append("")
    return "\n".join(lines)


def detect_relationships_llm(
    langchain_table_info: str,
    table_schemas: dict[str, list[ColumnSchema]],
    table_id_map: dict[str, str],
) -> list[DetectedRelationship]:
    """
    **Strategy 3 — LLM-inferred relationships.**

    Sends the DDL + sample values to an LLM and parses the structured
    JSON response into ``DetectedRelationship`` objects.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = _LLM_RELATIONSHIP_PROMPT.format(
        ddl=langchain_table_info,
        samples=_format_samples_for_llm(table_schemas),
    )
    response = llm.invoke(prompt)
    text: str = response.content.strip()

    # Strip accidental markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    valid_tables = set(table_schemas.keys())
    try:
        raw: list[dict] = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM strategy returned unparseable JSON: %s", text[:200])
        return []

    relationships: list[DetectedRelationship] = []
    for item in raw:
        ft = item.get("from_table", "")
        fc = item.get("from_column", "")
        tt = item.get("to_table", "")
        tc = item.get("to_column", "")
        rt = item.get("relationship_type", "many_to_one")
        if ft not in valid_tables or tt not in valid_tables:
            continue
        if rt not in ("many_to_one", "one_to_one", "many_to_many"):
            rt = "many_to_one"
        relationships.append(DetectedRelationship(
            from_table_name=ft,
            from_column=fc,
            to_table_name=tt,
            to_column=tc,
            relationship_type=rt,
            confidence=0.75,
            detection_method="auto_llm",
        ))

    logger.info("LLM strategy found %d relationships", len(relationships))
    return relationships


# ---- Combine all strategies (user corrections always win) ----

def combine_detected_relationships(
    *strategy_results: list[DetectedRelationship],
    user_relationships: list[DetectedRelationship] | None = None,
) -> list[DetectedRelationship]:
    """
    Merge results from all three auto-detection strategies while
    **always preserving user-defined corrections**.

    Priority order:
    1. User-confirmed relationships (``detection_method='user'``) — untouchable.
    2. Among auto-detected, the entry with the highest confidence wins
       for each unique ``(from_table, from_col, to_table, to_col)`` pair.
    3. Reversed pairs (A→B vs B→A) are treated as the same relationship.
    """
    user_relationships = user_relationships or []

    # Build a set of column-pair keys already covered by user corrections.
    # Both directions are added so auto-detected matches are suppressed.
    user_pair_keys: set[tuple[str, str, str, str]] = set()
    for rel in user_relationships:
        fwd = (rel.from_table_name, rel.from_column, rel.to_table_name, rel.to_column)
        rev = (rel.to_table_name, rel.to_column, rel.from_table_name, rel.from_column)
        user_pair_keys.add(fwd)
        user_pair_keys.add(rev)

    # Merge auto-detected results, skipping anything that conflicts with a
    # user-defined relationship.
    best: dict[tuple, DetectedRelationship] = {}
    for results in strategy_results:
        for rel in results:
            key = (rel.from_table_name, rel.from_column, rel.to_table_name, rel.to_column)
            rev = (rel.to_table_name, rel.to_column, rel.from_table_name, rel.from_column)

            if key in user_pair_keys or rev in user_pair_keys:
                continue  # user correction takes precedence

            existing = best.get(key) or best.get(rev)
            if existing is None or rel.confidence > existing.confidence:
                best[key] = rel

    # Return auto-detected only — user relationships are already persisted
    # in the DB and are never deleted by cleanup_existing_relationships.
    return list(best.values())


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split *text* into overlapping chunks using LangChain splitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> list[list[float]]:
    """
    Generate embeddings via OpenAI.

    Automatically batches long lists to stay within API limits.
    """
    from openai import OpenAI

    client = OpenAI()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in resp.data])
    return all_embeddings
