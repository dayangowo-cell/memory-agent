from pathlib import Path

from memory_agent.rag.loaders import (
    load_document,
)

from memory_agent.rag.splitter import (
    split_documents,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

KNOWLEDGE_BASE = (
    PROJECT_ROOT
    / "knowledge_base"
)


# =========================================================
# PDF
# =========================================================

pdf_files = sorted(
    (
        KNOWLEDGE_BASE
        / "langchain"
    ).glob("*.pdf")
)

pdf_path = pdf_files[0]

pdf_documents = load_document(
    pdf_path,
    category="langchain",
)

pdf_chunks = split_documents(
    pdf_documents
)


print("=" * 80)
print("PDF Splitter 测试")
print("=" * 80)

print(
    "原始 Document 数量：",
    len(pdf_documents),
)

print(
    "切分后 Chunk 数量：",
    len(pdf_chunks),
)

print()


for chunk in pdf_chunks[:3]:

    print(
        "metadata:",
        chunk.metadata,
    )

    print(
        "字符数:",
        len(chunk.page_content),
    )

    print(
        chunk.page_content[:300]
    )

    print("-" * 80)


# =========================================================
# Markdown
# =========================================================

md_files = sorted(
    (
        KNOWLEDGE_BASE
        / "langgraph"
    ).glob("*.md")
)

md_path = md_files[0]

md_documents = load_document(
    md_path,
    category="langgraph",
)

md_chunks = split_documents(
    md_documents
)


print("\n")
print("=" * 80)
print("Markdown Splitter 测试")
print("=" * 80)

print(
    "原始 Document 数量：",
    len(md_documents),
)

print(
    "切分后 Chunk 数量：",
    len(md_chunks),
)

print()


for chunk in md_chunks[:5]:

    print(
        "chunk_index:",
        chunk.metadata.get(
            "chunk_index"
        ),
    )

    print(
        "heading_path:",
        chunk.metadata.get(
            "heading_path"
        ),
    )

    print(
        "字符数:",
        len(chunk.page_content),
    )

    print(
        chunk.page_content[:400]
    )

    print("-" * 80)