"""RAG PostgreSQL 数据访问层。

职责：
- 连接 PostgreSQL
- 写入 rag_documents
- 写入 rag_chunks

这里不负责：
- 文件加载
- Chunk 切分
- Embedding

这些职责分别属于其他模块。
"""

import os
from typing import Any
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


# =========================================================
# 环境变量
# =========================================================

load_dotenv(override=True)

POSTGRES_URI = os.getenv(
    "POSTGRES_URI"
)

EMBEDDING_DIM = 1024


if not POSTGRES_URI:
    raise RuntimeError(
        "没有读取到 POSTGRES_URI，"
        "请检查项目根目录 .env 文件。"
    )


# =========================================================
# PostgreSQL Connection
# =========================================================

def get_connection() -> Connection:
    """
    创建 PostgreSQL 连接。

    使用项目现有的 POSTGRES_URI，
    不单独创建第二套数据库配置。
    """

    return psycopg.connect(
        POSTGRES_URI,
        row_factory=dict_row,
    )


# =========================================================
# Vector 转换
# =========================================================

def _to_vector_literal(
    vector: list[float],
) -> str:
    """
    将 Python list[float] 转换成 pgvector 接受的格式：

        [0.1,0.2,0.3,...]

    PostgreSQL 中最终通过：

        %s::vector

    转换为真正的 vector 类型。
    """

    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding 维度错误："
            f"{len(vector)}，"
            f"期望 {EMBEDDING_DIM}"
        )

    return (
        "["
        + ",".join(
            str(float(value))
            for value in vector
        )
        + "]"
    )


# =========================================================
# Document
# =========================================================

def get_document_by_hash(
    conn: Connection,
    *,
    file_hash: str,
) -> dict[str, Any] | None:
    """
    根据文件 SHA256 查询文档。

    用途：
        判断完全相同的文件是否已经导入过。
    """

    row = conn.execute(
        """
        SELECT
            document_id,
            title,
            source_type,
            category,
            source_path,
            file_hash,
            metadata,
            created_at,
            updated_at
        FROM rag_documents
        WHERE file_hash = %s
        LIMIT 1;
        """,
        (file_hash,),
    ).fetchone()

    return row


def delete_documents_by_source_path(
    conn: Connection,
    *,
    source_path: str,
) -> int:
    """
    删除相同 source_path 的旧文档。

    rag_chunks.document_id 设置了：

        ON DELETE CASCADE

    因此删除 rag_documents 后，
    对应旧 Chunk 会自动删除。

    Returns:
        删除的 document 数量。
    """

    cursor = conn.execute(
        """
        DELETE FROM rag_documents
        WHERE source_path = %s;
        """,
        (source_path,),
    )

    return cursor.rowcount



def upsert_document(
    conn: Connection,
    *,
    title: str,
    source_type: str,
    category: str,
    source_path: str,
    file_hash: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    写入一条 RAG 原始文档记录。

    file_hash 已有：
        更新已有文档基本信息。

    file_hash 不存在：
        新增 Document。

    Returns:
        document_id
    """

    document_id = uuid4()

    row = conn.execute(
        """
        INSERT INTO rag_documents (
            document_id,
            title,
            source_type,
            category,
            source_path,
            file_hash,
            metadata
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )

        ON CONFLICT (file_hash)
        DO UPDATE SET
            title = EXCLUDED.title,
            source_type = EXCLUDED.source_type,
            category = EXCLUDED.category,
            source_path = EXCLUDED.source_path,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()

        RETURNING document_id;
        """,
        (
            document_id,
            title,
            source_type,
            category,
            source_path,
            file_hash,
            Jsonb(metadata or {}),
        ),
    ).fetchone()

    if row is None:
        raise RuntimeError(
            "写入 rag_documents 失败。"
        )

    return str(
        row["document_id"]
    )


# =========================================================
# Chunk
# =========================================================

def upsert_chunk(
    conn: Connection,
    *,
    document_id: str,
    chunk_index: int,
    content: str,
    page_number: int | None,
    heading_path: str | None,
    token_count: int | None,
    content_hash: str,
    metadata: dict[str, Any] | None,
    embedding: list[float],
) -> str:
    """
    写入一条 RAG Chunk。

    当前利用：
        UNIQUE(document_id, content_hash)

    避免同一个文档中的相同 Chunk 被重复写入。

    Returns:
        chunk_id
    """

    chunk_id = uuid4()

    vector_literal = (
        _to_vector_literal(
            embedding
        )
    )

    row = conn.execute(
        """
        INSERT INTO rag_chunks (
            chunk_id,
            document_id,
            chunk_index,
            content,
            page_number,
            heading_path,
            token_count,
            content_hash,
            metadata,
            embedding
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s::vector
        )

        ON CONFLICT (
            document_id,
            content_hash
        )
        DO UPDATE SET
            content = EXCLUDED.content,
            page_number = EXCLUDED.page_number,
            heading_path = EXCLUDED.heading_path,
            token_count = EXCLUDED.token_count,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding

        RETURNING chunk_id;
        """,
        (
            chunk_id,
            document_id,
            chunk_index,
            content,
            page_number,
            heading_path,
            token_count,
            content_hash,
            Jsonb(metadata or {}),
            vector_literal,
        ),
    ).fetchone()

    if row is None:
        raise RuntimeError(
            "写入 rag_chunks 失败。"
        )

    return str(
        row["chunk_id"]
    )


__all__ = [
    "get_connection",
    "get_document_by_hash",
    "delete_documents_by_source_path",
    "upsert_document",
    "upsert_chunk",
]