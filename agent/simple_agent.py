"""
LangGraph SQL Agent with guardrails.

Graph topology::

    START
      │
      ▼
    input_guardrail ──[rejected]──► handle_blocked ──► END
      │ [passed]
      ▼
    retrieve_schema
      │
      ▼
    generate_sql ◄─────────────────────────────┐
      │                                         │
      ▼                                         │
    sql_guardrail ──[rejected + retries left]───┘
      │ [passed]    [rejected + no retries]──► handle_blocked ──► END
      ▼
    execute_sql ────[error + retries left]──► generate_sql
      │ [success]   [error + no retries]───► handle_blocked ──► END
      ▼
    summarize
      │
      ▼
    output_guardrail
      │
      ▼
    END

Usage::

    from agent.simple_agent import run_agent

    result = run_agent(
        project_id="<uuid>",
        duckdb_path="/tmp/project.duckdb",
        user_query="How many orders per month?",
    )
    print(result["answer"])
    print(result["sql"])
"""

from __future__ import annotations

import json
import logging
import operator
import os
import re
from typing import Annotated, Any, TypedDict

import duckdb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag.retrieval.index import retrieve_context

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. State
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    # --- Inputs (set at invocation) ---
    user_query: str
    chat_history: list[dict]
    project_id: str
    duckdb_path: str

    # --- Settings ---
    llm_model: str
    allow_write_queries: bool
    max_rows_returned: int
    max_retries: int

    # --- Input guardrail ---
    input_safe: bool
    rejection_reason: str

    # --- Retrieval ---
    schema_context: str
    retrieved_chunk_ids: list[str]

    # --- SQL generation ---
    generated_sql: str
    sql_explanation: str

    # --- SQL guardrail ---
    sql_safe: bool
    sql_issues: list[str]

    # --- Execution ---
    execution_result: str
    execution_error: str
    execution_columns: list[str]
    execution_row_count: int

    # --- Final output ---
    final_answer: str

    # --- Control flow ---
    retry_count: int
    error_history: Annotated[list[str], operator.add]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Structured-output models
# ═══════════════════════════════════════════════════════════════════════════

class SQLGeneration(BaseModel):
    sql: str = Field(description="A single DuckDB-dialect SELECT statement")
    explanation: str = Field(description="One-sentence explanation of the query logic")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Guardrail checks (pure functions — no LLM, no I/O)
# ═══════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?(your\s+)?previous",
    r"you\s+are\s+now\s+a",
    r"<\s*/?system\s*>",
    r"pretend\s+(you\s+are|to\s+be)",
    r"\bsystem\s*:\s*",
    r"disregard\s+(all\s+)?(prior|above)",
    r"override\s+(your\s+)?instructions",
]

_WRITE_KW = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"COPY|EXECUTE|VACUUM|REINDEX|ATTACH|DETACH)\b",
    re.IGNORECASE,
)
_MULTI_STMT = re.compile(r";\s*\S")
_SYS_CATALOG = re.compile(
    r"\b(pg_catalog|pg_stat|pg_settings|pg_shadow|pg_authid)\b",
    re.IGNORECASE,
)
_COMMENT_INJ = re.compile(r"(/\*|\*/|--\s*\S)")


def check_input(query: str) -> tuple[bool, str]:
    """Rule-based input guardrail. Returns *(is_safe, reason)*."""
    if not query or len(query.strip()) < 3:
        return False, "Query is too short or empty."
    if len(query) > 10_000:
        return False, "Query exceeds the maximum allowed length."
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            return False, "Potential prompt-injection attempt detected."
    return True, ""


def check_sql(sql: str, allow_writes: bool = False) -> tuple[bool, list[str]]:
    """Rule-based SQL guardrail. Returns *(is_safe, issues)*."""
    issues: list[str] = []
    stripped = sql.strip() if sql else ""
    if not stripped:
        issues.append("Empty SQL query.")
        return False, issues
    if not allow_writes and _WRITE_KW.search(stripped):
        found = _WRITE_KW.findall(stripped)
        issues.append(f"Write / DDL operations are disabled (found: {found}).")
    if _MULTI_STMT.search(stripped):
        issues.append("Multiple statements detected — only a single query is allowed.")
    if _SYS_CATALOG.search(stripped):
        issues.append("Direct access to system catalogs is not permitted.")
    if _COMMENT_INJ.search(stripped):
        issues.append("SQL comments are not allowed (possible injection vector).")
    return len(issues) == 0, issues


_PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN-like pattern"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit-card-like pattern"),
]


def check_output(answer: str) -> str | None:
    """Return a warning prefix if PII-like patterns appear, else *None*."""
    warnings = [
        label for pat, label in _PII_PATTERNS if re.search(pat, answer)
    ]
    if warnings:
        return (
            f"**Warning:** The response may contain sensitive data "
            f"({', '.join(warnings)}). Handle with care."
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 4. LLM helper
# ═══════════════════════════════════════════════════════════════════════════

def _llm(model: str = "gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Graph nodes
# ═══════════════════════════════════════════════════════════════════════════

# ---------- guardrail nodes ----------

def input_guardrail_node(state: AgentState) -> dict:
    ok, reason = check_input(state.get("user_query", ""))
    return {"input_safe": ok, "rejection_reason": reason}


def sql_guardrail_node(state: AgentState) -> dict:
    ok, issues = check_sql(
        state.get("generated_sql", ""),
        allow_writes=state.get("allow_write_queries", False),
    )
    return {"sql_safe": ok, "sql_issues": issues}


def output_guardrail_node(state: AgentState) -> dict:
    answer = state.get("final_answer", "")
    warning = check_output(answer)
    if warning:
        return {"final_answer": f"{warning}\n\n{answer}"}
    return {}


# ---------- core pipeline nodes ----------

def retrieve_schema_node(state: AgentState) -> dict:
    result = retrieve_context(state["project_id"], state["user_query"])
    return {
        "schema_context": result.context,
        "retrieved_chunk_ids": result.chunk_ids,
    }


def generate_sql_node(state: AgentState) -> dict:
    model = state.get("llm_model", "gpt-4o")
    llm = _llm(model).with_structured_output(SQLGeneration)

    # Detect retry and build error feedback
    prev_issues = state.get("sql_issues") or []
    prev_error = state.get("execution_error") or ""
    is_retry = bool(prev_issues) or bool(prev_error)
    retry_count = state.get("retry_count", 0) + (1 if is_retry else 0)

    feedback_parts: list[str] = []
    if prev_issues:
        feedback_parts.append(
            "**Guardrail rejection** — fix these issues:\n"
            + "\n".join(f"- {i}" for i in prev_issues)
            + f"\nPrevious SQL:\n```\n{state.get('generated_sql', '')}\n```"
        )
    if prev_error:
        feedback_parts.append(
            f"**Execution error:**\n{prev_error}\n"
            f"Previous SQL:\n```\n{state.get('generated_sql', '')}\n```"
        )

    max_rows = state.get("max_rows_returned", 500)
    messages = [
        SystemMessage(content=(
            "You are an expert SQL analyst.  Write a **single DuckDB SELECT** "
            "statement to answer the user's question using ONLY the schema "
            "context provided.\n\n"
            "Rules:\n"
            "- Use ONLY table / column names from the schema context.\n"
            "- Use explicit JOIN … ON syntax.\n"
            f"- Add LIMIT {max_rows} unless the user asks otherwise.\n"
            "- Do NOT include SQL comments (-- or /* */).\n"
            "- Do NOT include multiple statements.\n"
        )),
        HumanMessage(content=(
            f"## Schema Context\n{state.get('schema_context', '(none)')}\n\n"
            f"## User Question\n{state['user_query']}\n\n"
            + (
                "## Error Feedback (fix before retrying)\n"
                + "\n\n".join(feedback_parts)
                if feedback_parts else ""
            )
        )),
    ]

    result = llm.invoke(messages)
    return {
        "generated_sql": result.sql,
        "sql_explanation": result.explanation,
        "retry_count": retry_count,
        "sql_issues": [],
        "execution_error": "",
    }


def execute_sql_node(state: AgentState) -> dict:
    sql = state["generated_sql"]
    max_rows = state.get("max_rows_returned", 500)

    con = duckdb.connect(state["duckdb_path"], read_only=True)
    try:
        df = con.execute(sql).fetchdf()
        total = len(df)
        if total > max_rows:
            df = df.head(max_rows)
        columns = list(df.columns)
        rows = df.to_dict(orient="records")

        preview = json.dumps(rows[:50], default=str, indent=2)
        if len(rows) > 50:
            preview += f"\n... ({len(rows) - 50} more rows, {total} total)"

        return {
            "execution_result": preview,
            "execution_error": "",
            "execution_columns": columns,
            "execution_row_count": total,
        }
    except Exception as exc:
        return {
            "execution_result": "",
            "execution_error": str(exc),
            "execution_columns": [],
            "execution_row_count": 0,
            "error_history": [f"SQL error: {exc}"],
        }
    finally:
        con.close()


def summarize_node(state: AgentState) -> dict:
    model = state.get("llm_model", "gpt-4o")
    llm = _llm(model)
    messages = [
        SystemMessage(content=(
            "You are a data analyst presenting query results.  Give a clear, "
            "concise answer to the user's question based on the SQL results.  "
            "Include key numbers.  If the data is tabular, format it as a "
            "markdown table."
        )),
        HumanMessage(content=(
            f"**User question:** {state['user_query']}\n\n"
            f"**SQL executed:**\n```sql\n{state.get('generated_sql', '')}\n```\n\n"
            f"**Explanation:** {state.get('sql_explanation', '')}\n\n"
            f"**Results** ({state.get('execution_row_count', 0)} rows, "
            f"columns: {state.get('execution_columns', [])}):\n"
            f"```json\n{state.get('execution_result', '(empty)')}\n```"
        )),
    ]
    response = llm.invoke(messages)
    return {"final_answer": response.content}


# ---------- terminal / error node ----------

def handle_blocked_node(state: AgentState) -> dict:
    reason = state.get("rejection_reason", "")
    if reason:
        return {"final_answer": f"I can't process this request. {reason}"}

    issues = state.get("sql_issues") or []
    err = state.get("execution_error") or ""
    parts = ["I wasn't able to generate a valid SQL query after multiple attempts."]
    if issues:
        parts.append(f"Last guardrail issues: {', '.join(issues)}")
    if err:
        parts.append(f"Last execution error: {err}")
    parts.append("Please try rephrasing your question.")
    return {"final_answer": " ".join(parts)}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Routing (conditional edges)
# ═══════════════════════════════════════════════════════════════════════════

def _after_input_guard(state: AgentState) -> str:
    return "retrieve_schema" if state.get("input_safe") else "handle_blocked"


def _after_sql_guard(state: AgentState) -> str:
    if state.get("sql_safe"):
        return "execute_sql"
    if state.get("retry_count", 0) < state.get("max_retries", 3):
        return "generate_sql"
    return "handle_blocked"


def _after_execution(state: AgentState) -> str:
    if not state.get("execution_error"):
        return "summarize"
    if state.get("retry_count", 0) < state.get("max_retries", 3):
        return "generate_sql"
    return "handle_blocked"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Graph construction
# ═══════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("retrieve_schema", retrieve_schema_node)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("sql_guardrail", sql_guardrail_node)
    g.add_node("execute_sql", execute_sql_node)
    g.add_node("summarize", summarize_node)
    g.add_node("output_guardrail", output_guardrail_node)
    g.add_node("handle_blocked", handle_blocked_node)

    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", _after_input_guard,
                            ["retrieve_schema", "handle_blocked"])
    g.add_edge("retrieve_schema", "generate_sql")
    g.add_edge("generate_sql", "sql_guardrail")
    g.add_conditional_edges("sql_guardrail", _after_sql_guard,
                            ["execute_sql", "generate_sql", "handle_blocked"])
    g.add_conditional_edges("execute_sql", _after_execution,
                            ["summarize", "generate_sql", "handle_blocked"])
    g.add_edge("summarize", "output_guardrail")
    g.add_edge("output_guardrail", END)
    g.add_edge("handle_blocked", END)

    return g


# ═══════════════════════════════════════════════════════════════════════════
# 8. Public API
# ═══════════════════════════════════════════════════════════════════════════

def create_sql_agent():
    """Compile and return the LangGraph SQL agent (reusable across calls)."""
    return build_graph().compile()


def run_agent(
    project_id: str,
    duckdb_path: str,
    user_query: str,
    chat_history: list | None = None,
    model: str = "gpt-4o",
    allow_write_queries: bool = False,
    max_rows_returned: int = 500,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Run the full SQL-agent graph for a single user query.

    Returns
    -------
    dict with keys:
      - ``answer``              – natural-language response
      - ``sql``                 – the SQL that was executed (or None)
      - ``sql_explanation``     – one-line explanation
      - ``row_count``           – number of result rows
      - ``columns``             – list of column names
      - ``retrieved_chunk_ids`` – RAG provenance
      - ``retry_count``         – how many SQL-generation retries occurred
      - ``error_history``       – accumulated error messages (if any)
    """
    agent = create_sql_agent()
    final = agent.invoke({
        "user_query": user_query,
        "chat_history": chat_history or [],
        "project_id": project_id,
        "duckdb_path": duckdb_path,
        "llm_model": model,
        "allow_write_queries": allow_write_queries,
        "max_rows_returned": max_rows_returned,
        "max_retries": max_retries,
        "retry_count": 0,
        "error_history": [],
    })
    return {
        "answer": final.get("final_answer", ""),
        "sql": final.get("generated_sql"),
        "sql_explanation": final.get("sql_explanation"),
        "row_count": final.get("execution_row_count"),
        "columns": final.get("execution_columns"),
        "retrieved_chunk_ids": final.get("retrieved_chunk_ids"),
        "retry_count": final.get("retry_count", 0),
        "error_history": final.get("error_history", []),
    }
