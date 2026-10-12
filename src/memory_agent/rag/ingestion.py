"""RAG 知识库导入模块。

完整流程：

原始文件
    ↓
文件 Hash
    ↓
检查是否已经导入
    ↓
Loader
    ↓
Splitter
    ↓
Batch Embedding
    ↓
PostgreSQL
        ├── rag_documents
        └── rag_chunks
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from memory_agent.embeddings import (
    embed_rag_texts,
)
from memory_agent.rag.loaders import (
    load_document,
)
from memory_agent.rag.repository import (
    delete_documents_by_source_path,
    get_connection,
    get_document_by_hash,
    upsert_chunk,
    upsert_document,
)
from memory_agent.rag.splitter import (
    split_documents,
)


# =========================================================
# 配置
# =========================================================

DEFAULT_BATCH_SIZE = 16


# =========================================================
# 导入结果
# =========================================================

@dataclass
class IngestionResult:
    """
    一次文档导入的结果。
    """

    source_path: str

    status: str

    document_id: str | None

    document_count: int

    chunk_count: int

    embedding_count: int

    message: str


# =========================================================
# Hash
# =========================================================

def calculate_file_hash(
    file_path: Path,
) -> str:
    """
    计算整个文件的 SHA256。

    文件内容完全相同：
        Hash 相同

    文件内容发生变化：
        Hash 改变
    """

    return sha256(
        file_path.read_bytes()
    ).hexdigest()


# =========================================================
# Relative Path
# =========================================================

def get_relative_source_path(
    file_path: Path,
    project_root: Path,
) -> str:
    """
    将绝对路径转换为项目相对路径。

    例如：

    C:\\pycharm2\\memory-agent\\knowledge_base\\xxx.pdf

    转成：

    knowledge_base/langchain/xxx.pdf
    """

    return (
        file_path.resolve()
        .relative_to(
            project_root.resolve()
        )
        .as_posix()
    )


# =========================================================
# Batch Embedding
# =========================================================

def embed_chunks(
    chunks,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """
    批量生成 Chunk Embedding。

    不采用：

        110 chunks
        → 110 次 HTTP 请求

    而采用：

        110 chunks
        → 每批 16 个
        → 约 7 次 HTTP 请求

    可以明显降低 API 请求次数。
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size 必须大于 0。"
        )

    all_vectors: list[
        list[float]
    ] = []

    total = len(chunks)

    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )

        batch = chunks[
            start:end
        ]

        texts = [
            chunk.page_content
            for chunk in batch
        ]

        print(
            f"Embedding: "
            f"{start + 1}-{end}"
            f" / {total}"
        )

        vectors = embed_rag_texts(
            texts
        )

        if len(vectors) != len(batch):
            raise RuntimeError(
                "Embedding 数量与 "
                "Chunk 数量不一致。"
            )

        all_vectors.extend(
            vectors
        )

    return all_vectors


# =========================================================
# 单文件 Ingestion
# =========================================================

def ingest_file(
    file_path: str | Path,
    *,
    category: str,
    project_root: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> IngestionResult:
    """
    将一份 PDF / Markdown 完整导入 RAG。

    处理逻辑：

    1. 计算文件 Hash
    2. 检查是否已经导入
    3. Loader
    4. Splitter
    5. Embedding
    6. 删除同路径旧版本
    7. 写入 Document
    8. 写入全部 Chunk

    Args:
        file_path:
            原始知识库文件。

        category:
            langchain / langgraph 等分类。

        project_root:
            项目根目录。

        batch_size:
            Embedding 每批数量。

        force:
            即使 Hash 相同也重新导入。
    """

    path = Path(
        file_path
    ).resolve()

    root = Path(
        project_root
    ).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"文件不存在: {path}"
        )

    # -----------------------------------------------------
    # 1. Hash + 相对路径
    # -----------------------------------------------------

    file_hash = (
        calculate_file_hash(
            path
        )
    )

    relative_source_path = (
        get_relative_source_path(
            path,
            root,
        )
    )

    print("=" * 80)
    print(
        "开始导入：",
        relative_source_path,
    )
    print(
        "SHA256：",
        file_hash,
    )

    # -----------------------------------------------------
    # 2. 先检查完全相同文件是否已经存在
    #
    # 这里一定放在 Embedding 之前。
    #
    # 否则明明文件没变，
    # 还是会白白调用一次 Embedding API。
    # -----------------------------------------------------

    with get_connection() as conn:

        existing = (
            get_document_by_hash(
                conn,
                file_hash=file_hash,
            )
        )

    if (
        existing is not None
        and not force
    ):

        document_id = str(
            existing[
                "document_id"
            ]
        )

        print(
            "文件已经导入，跳过。"
        )

        return IngestionResult(
            source_path=relative_source_path,
            status="skipped",
            document_id=document_id,
            document_count=0,
            chunk_count=0,
            embedding_count=0,
            message="same_file_hash",
        )

    # -----------------------------------------------------
    # 3. Loader
    # -----------------------------------------------------

    documents = load_document(
        path,
        category=category,
    )

    print(
        "Document 数量：",
        len(documents),
    )

    # -----------------------------------------------------
    # 4. Splitter
    # -----------------------------------------------------

    chunks = split_documents(
        documents
    )

    if not chunks:
        raise RuntimeError(
            f"文档没有生成任何 Chunk: "
            f"{path}"
        )

    print(
        "Chunk 数量：",
        len(chunks),
    )

    # -----------------------------------------------------
    # 5. 修正 metadata 中路径
    #
    # Loader 产生的是本机绝对路径。
    # 数据库里统一保存项目相对路径。
    # -----------------------------------------------------

    for chunk in chunks:

        chunk.metadata[
            "source_path"
        ] = relative_source_path

        chunk.metadata[
            "source"
        ] = relative_source_path

    # -----------------------------------------------------
    # 6. Embedding
    #
    # 先完成所有 Embedding，
    # 再修改数据库。
    #
    # 如果 API 在这里失败，
    # 原来的数据库版本不会受影响。
    # -----------------------------------------------------

    vectors = embed_chunks(
        chunks,
        batch_size=batch_size,
    )

    print(
        "Embedding 数量：",
        len(vectors),
    )

    # -----------------------------------------------------
    # 7. PostgreSQL Transaction
    # -----------------------------------------------------

    with get_connection() as conn:

        # ---------------------------------------------
        # 相同路径存在旧版本：
        # 删除旧 Document
        #
        # rag_chunks 会 ON DELETE CASCADE
        # 自动清理。
        # ---------------------------------------------

        deleted = (
            delete_documents_by_source_path(
                conn,
                source_path=relative_source_path,
            )
        )

        if deleted:
            print(
                f"删除旧版本 Document："
                f"{deleted}"
            )

        # ---------------------------------------------
        # 写 document
        # ---------------------------------------------

        document_id = (
            upsert_document(
                conn,
                title=path.stem,
                source_type=(
                    "pdf"
                    if path.suffix.lower()
                    == ".pdf"
                    else "markdown"
                ),
                category=category,
                source_path=relative_source_path,
                file_hash=file_hash,
                metadata={
                    "file_name": (
                        path.name
                    ),
                },
            )
        )

        # ---------------------------------------------
        # 写所有 chunks
        # ---------------------------------------------

        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        ):

            content_hash = sha256(
                chunk.page_content.encode(
                    "utf-8"
                )
            ).hexdigest()

            upsert_chunk(
                conn,
                document_id=document_id,
                chunk_index=(
                    chunk.metadata[
                        "chunk_index"
                    ]
                ),
                content=(
                    chunk.page_content
                ),
                page_number=(
                    chunk.metadata.get(
                        "page_number"
                    )
                ),
                heading_path=(
                    chunk.metadata.get(
                        "heading_path"
                    )
                ),

                # 暂时不伪造 token 数量。
                # 后续需要时使用对应模型
                # tokenizer 精确统计。
                token_count=None,

                content_hash=(
                    content_hash
                ),
                metadata=(
                    chunk.metadata
                ),
                embedding=vector,
            )

        # ---------------------------------------------
        # 所有 Chunk 都成功以后才提交
        # ---------------------------------------------

        conn.commit()

    print(
        "导入完成：",
        relative_source_path,
    )

    return IngestionResult(
        source_path=relative_source_path,
        status="ingested",
        document_id=document_id,
        document_count=len(
            documents
        ),
        chunk_count=len(
            chunks
        ),
        embedding_count=len(
            vectors
        ),
        message="success",
    )


__all__ = [
    "IngestionResult",
    "calculate_file_hash",
    "embed_chunks",
    "ingest_file",
]