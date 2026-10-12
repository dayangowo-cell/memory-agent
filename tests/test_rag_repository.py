from hashlib import sha256
from pathlib import Path

from memory_agent.embeddings import (
    embed_rag_texts,
)
from memory_agent.rag.loaders import (
    load_document,
)
from memory_agent.rag.repository import (
    get_connection,
    upsert_chunk,
    upsert_document,
)
from memory_agent.rag.splitter import (
    split_documents,
)


# =========================================================
# 路径
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

LANGCHAIN_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "langchain"
)


# =========================================================
# 1. 找第一份 PDF
# =========================================================

pdf_files = sorted(
    LANGCHAIN_DIR.glob("*.pdf")
)

if not pdf_files:
    raise FileNotFoundError(
        f"没有找到 PDF："
        f"{LANGCHAIN_DIR}"
    )

pdf_path = pdf_files[0]

print("=" * 80)
print("RAG Repository 端到端测试")
print("=" * 80)

print(
    "测试文件：",
    pdf_path.name,
)


# =========================================================
# 2. 文件 Hash
# =========================================================

file_hash = sha256(
    pdf_path.read_bytes()
).hexdigest()

print(
    "file_hash：",
    file_hash,
)


# =========================================================
# 3. Loader
# =========================================================

documents = load_document(
    pdf_path,
    category="langchain",
)

print(
    "Document 数量：",
    len(documents),
)


# =========================================================
# 4. Splitter
# =========================================================

chunks = split_documents(
    documents
)

print(
    "Chunk 数量：",
    len(chunks),
)


# =========================================================
# 5. 只取第一个 Chunk
#
# 当前只做最小闭环，
# 不要一次灌入全部 110 个。
# =========================================================

chunk = chunks[0]

print(
    "测试 Chunk 字符数：",
    len(chunk.page_content),
)

print(
    "page_number：",
    chunk.metadata.get(
        "page_number"
    ),
)


# =========================================================
# 6. Embedding
# =========================================================

embedding = embed_rag_texts(
    [
        chunk.page_content
    ]
)[0]

print(
    "Embedding 维度：",
    len(embedding),
)


# =========================================================
# 7. Hash Chunk
# =========================================================

content_hash = sha256(
    chunk.page_content.encode(
        "utf-8"
    )
).hexdigest()


# =========================================================
# 8. 使用相对路径
#
# 不把：
# C:\\pycharm2\\memory-agent\\...
#
# 写死进数据库。
#
# 保存为：
# knowledge_base/langchain/xxx.pdf
# =========================================================

relative_source_path = (
    pdf_path
    .relative_to(PROJECT_ROOT)
    .as_posix()
)


# =========================================================
# 9. PostgreSQL
# =========================================================

with get_connection() as conn:

    # -----------------------------------------------------
    # rag_documents
    # -----------------------------------------------------

    document_id = upsert_document(
        conn,
        title=pdf_path.stem,
        source_type="pdf",
        category="langchain",
        source_path=relative_source_path,
        file_hash=file_hash,
        metadata={
            "file_name": pdf_path.name,
        },
    )

    print(
        "document_id：",
        document_id,
    )

    # -----------------------------------------------------
    # 调整 Chunk metadata
    #
    # 测试阶段顺手把 source_path
    # 转成项目相对路径。
    # -----------------------------------------------------

    chunk_metadata = (
        chunk.metadata.copy()
    )

    chunk_metadata[
        "source_path"
    ] = relative_source_path

    chunk_metadata[
        "source"
    ] = relative_source_path

    # -----------------------------------------------------
    # rag_chunks
    # -----------------------------------------------------

    chunk_id = upsert_chunk(
        conn,
        document_id=document_id,
        chunk_index=chunk.metadata[
            "chunk_index"
        ],
        content=chunk.page_content,
        page_number=chunk.metadata.get(
            "page_number"
        ),
        heading_path=chunk.metadata.get(
            "heading_path"
        ),
        token_count=None,
        content_hash=content_hash,
        metadata=chunk_metadata,
        embedding=embedding,
    )

    print(
        "chunk_id：",
        chunk_id,
    )

    # -----------------------------------------------------
    # 提交事务
    # -----------------------------------------------------

    conn.commit()


print("=" * 80)
print("PASS")
print("=" * 80)