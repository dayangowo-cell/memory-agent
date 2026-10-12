"""RAG 向量检索模块。

流程：

用户 Query
    ↓
BGE-M3 Embedding
    ↓
PostgreSQL pgvector
    ↓
Cosine Similarity
    ↓
Top-K Chunks
"""

from dataclasses import dataclass
from typing import Any

from memory_agent.embeddings import (
    embed_rag_texts,
)
from memory_agent.rag.repository import (
    get_connection,
)


# =========================================================
# Retrieval Result
# =========================================================

@dataclass
class RetrievalResult:
    """
    单条 RAG 检索结果。
    """

    chunk_id: str
    document_id: str

    content: str

    score: float

    chunk_index: int

    page_number: int | None

    heading_path: str | None

    title: str

    category: str

    source_type: str

    source_path: str

    metadata: dict[str, Any]


# =========================================================
# Vector Literal
# =========================================================

def _to_vector_literal(
    vector: list[float],
) -> str:
    """
    Python list[float]
        ↓
    pgvector literal

    例如：

        [0.1,0.2,0.3]
    """

    return (
        "["
        + ",".join(
            str(float(value))
            for value in vector
        )
        + "]"
    )


# =========================================================
# Retriever
# =========================================================

def retrieve_chunks(
    query: str,
    *,
    top_k: int = 5,
    category: str | None = None,
) -> list[RetrievalResult]:
    """
    根据用户问题检索最相关的 Chunk。

    Args:
        query:
            用户查询。

        top_k:
            返回最相关的 K 个 Chunk。

        category:
            可选知识库分类，例如：
                langchain
                langgraph

    Returns:
        list[RetrievalResult]
    """

    # -----------------------------------------------------
    # 1. 参数检查
    # -----------------------------------------------------

    if not query.strip():
        raise ValueError(
            "query 不能为空。"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0。"
        )

    # -----------------------------------------------------
    # 2. Query Embedding
    #
    # 必须和文档 Chunk 使用相同的：
    #
    # BAAI/bge-m3
    #
    # 不能拿 Memory 的 Qwen3 Embedding 来搜。
    # -----------------------------------------------------

    query_vector = embed_rag_texts(
        [query]
    )[0]

    vector_literal = (
        _to_vector_literal(
            query_vector
        )
    )

    # -----------------------------------------------------
    # 3. pgvector 检索
    #
    # <=>:
    # cosine distance
    #
    # 距离越小：
    #     越相似
    #
    # 因此：
    #
    # similarity = 1 - cosine_distance
    # -----------------------------------------------------

    with get_connection() as conn:

        if category is None:

            rows = conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_index,
                    c.page_number,
                    c.heading_path,
                    c.metadata,

                    d.title,
                    d.category,
                    d.source_type,
                    d.source_path,

                    1 - (
                        c.embedding
                        <=> %s::vector
                    ) AS score

                FROM rag_chunks c

                JOIN rag_documents d
                    ON c.document_id =
                       d.document_id

                ORDER BY
                    c.embedding
                    <=> %s::vector

                LIMIT %s;
                """,
                (
                    vector_literal,
                    vector_literal,
                    top_k,
                ),
            ).fetchall()

        else:

            rows = conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_index,
                    c.page_number,
                    c.heading_path,
                    c.metadata,

                    d.title,
                    d.category,
                    d.source_type,
                    d.source_path,

                    1 - (
                        c.embedding
                        <=> %s::vector
                    ) AS score

                FROM rag_chunks c

                JOIN rag_documents d
                    ON c.document_id =
                       d.document_id

                WHERE d.category = %s

                ORDER BY
                    c.embedding
                    <=> %s::vector

                LIMIT %s;
                """,
                (
                    vector_literal,
                    category,
                    vector_literal,
                    top_k,
                ),
            ).fetchall()

    # -----------------------------------------------------
    # 4. 转成结构化结果
    # -----------------------------------------------------

    results: list[
        RetrievalResult
    ] = []

    for row in rows:

        results.append(
            RetrievalResult(
                chunk_id=str(
                    row["chunk_id"]
                ),
                document_id=str(
                    row["document_id"]
                ),
                content=row["content"],
                score=float(
                    row["score"]
                ),
                chunk_index=(
                    row["chunk_index"]
                ),
                page_number=(
                    row["page_number"]
                ),
                heading_path=(
                    row["heading_path"]
                ),
                title=row["title"],
                category=row["category"],
                source_type=(
                    row["source_type"]
                ),
                source_path=(
                    row["source_path"]
                ),
                metadata=(
                    row["metadata"]
                ),
            )
        )

    return results


__all__ = [
    "RetrievalResult",
    "retrieve_chunks",
]