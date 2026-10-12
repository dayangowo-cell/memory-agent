from pathlib import Path

from memory_agent.rag.loaders import load_document


# =========================================================
# 项目根目录
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_BASE = (
    PROJECT_ROOT
    / "knowledge_base"
)

LANGCHAIN_DIR = (
    KNOWLEDGE_BASE
    / "langchain"
)

LANGGRAPH_DIR = (
    KNOWLEDGE_BASE
    / "langgraph"
)


# =========================================================
# 自动寻找测试文件
# =========================================================

pdf_files = sorted(
    LANGCHAIN_DIR.glob("*.pdf")
)

md_files = sorted(
    list(
        LANGGRAPH_DIR.glob("*.md")
    )
    + list(
        LANGGRAPH_DIR.glob("*.markdown")
    )
)


if not pdf_files:
    raise FileNotFoundError(
        f"没有在以下目录找到 PDF："
        f"{LANGCHAIN_DIR}"
    )

if not md_files:
    raise FileNotFoundError(
        f"没有在以下目录找到 Markdown："
        f"{LANGGRAPH_DIR}"
    )


pdf_path = pdf_files[0]
md_path = md_files[0]


# =========================================================
# PDF Loader 测试
# =========================================================

print("=" * 80)
print("PDF Loader 测试")
print("=" * 80)

print(
    f"测试文件：{pdf_path.name}"
)

pdf_documents = load_document(
    pdf_path,
    category="langchain",
)

print(
    f"Document 数量："
    f"{len(pdf_documents)}"
)

if pdf_documents:

    first_pdf_doc = (
        pdf_documents[0]
    )

    print("\n第一页 metadata：")

    for key, value in (
        first_pdf_doc.metadata.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print(
        "\n第一页文本前 500 个字符："
    )

    print("-" * 80)

    print(
        first_pdf_doc.page_content[
            :500
        ]
    )

    print("-" * 80)


# =========================================================
# Markdown Loader 测试
# =========================================================

print("\n")
print("=" * 80)
print("Markdown Loader 测试")
print("=" * 80)

print(
    f"测试文件：{md_path.name}"
)

md_documents = load_document(
    md_path,
    category="langgraph",
)

print(
    f"Document 数量："
    f"{len(md_documents)}"
)

if md_documents:

    first_md_doc = (
        md_documents[0]
    )

    print("\nmetadata：")

    for key, value in (
        first_md_doc.metadata.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print(
        "\nMarkdown 前 800 个字符："
    )

    print("-" * 80)

    print(
        first_md_doc.page_content[
            :800
        ]
    )

    print("-" * 80)