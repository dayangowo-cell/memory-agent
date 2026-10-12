from pathlib import Path

from memory_agent.rag.ingestion import (
    ingest_file,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

pdf_files = sorted(
    (
        PROJECT_ROOT
        / "knowledge_base"
        / "langchain"
    ).glob("*.pdf")
)

if not pdf_files:
    raise FileNotFoundError(
        "没有找到测试 PDF。"
    )


result = ingest_file(
    pdf_files[0],
    category="langchain",
    project_root=PROJECT_ROOT,
    batch_size=16,
    force=True,
)


print("\n")
print("=" * 80)
print("Ingestion Result")
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