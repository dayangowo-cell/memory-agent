from pathlib import Path

from memory_agent.rag.ingestion import (
    ingest_file,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

LANGGRAPH_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "langgraph"
)


md_files = sorted(
    LANGGRAPH_DIR.glob("*.md")
)

if not md_files:
    raise FileNotFoundError(
        "没有找到 Markdown 文件。"
    )


result = ingest_file(
    md_files[0],
    category="langgraph",
    project_root=PROJECT_ROOT,
    batch_size=16,
)


print()
print("=" * 80)
print("Markdown Ingestion Result")
print("=" * 80)

print(
    "status:",
    result.status,
)

print(
    "document_id:",
    result.document_id,
)

print(
    "documents:",
    result.document_count,
)

print(
    "chunks:",
    result.chunk_count,
)

print(
    "embeddings:",
    result.embedding_count,
)

print(
    "message:",
    result.message,
)