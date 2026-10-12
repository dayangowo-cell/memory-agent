"""统一 Embedding 模块。

当前项目使用两套 Embedding：

1. Long-term Memory
   - Qwen/Qwen3-Embedding-8B
   - 1024 维
   - 保持与现有 Memory 向量空间一致

2. RAG
   - BAAI/bge-m3
   - 1024 维
   - 用于知识库文档和查询向量

重要：
不同 Embedding 模型生成的向量不能混合检索，
即使它们都是 1024 维。
"""

import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 基础配置
# =========================================================


load_dotenv(override=True)


EMBEDDING_DIM = 1024


# ---------------------------------------------------------
# Memory 使用的 Embedding
# ---------------------------------------------------------

MEMORY_EMBEDDING_MODEL = os.getenv(
    "MEMORY_EMBEDDING_MODEL",
    "Qwen/Qwen3-Embedding-8B",
)


# ---------------------------------------------------------
# RAG 使用的 Embedding
# ---------------------------------------------------------

RAG_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "BAAI/bge-m3",
)


# =========================================================
# SiliconFlow Client
# =========================================================

SILICONFLOW_API_KEY = os.getenv(
    "SILICONFLOW_API_KEY"
)

SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1",
)


if not SILICONFLOW_API_KEY:
    raise RuntimeError(
        "没有读取到 SILICONFLOW_API_KEY。"
        "请检查项目根目录 .env 文件。"
    )


client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
)


# =========================================================
# 底层通用 Embedding 函数
# =========================================================

def _embed_texts(
    texts: list[str],
    *,
    model: str,
    dimensions: int | None = None,
) -> list[list[float]]:
    """
    调用 SiliconFlow Embedding API。

    Args:
        texts:
            待向量化文本列表。

        model:
            Embedding 模型名称。

        dimensions:
            可选的输出维度。

            Qwen3-Embedding 支持指定 dimensions。

            BGE-M3 本身输出 1024 维，
            因此调用 BGE-M3 时不传该参数。

    Returns:
        list[list[float]]
    """

    # -----------------------------------------------------
    # 1. 空输入直接返回
    # -----------------------------------------------------

    if not texts:
        return []

    # -----------------------------------------------------
    # 2. 基础数据校验
    # -----------------------------------------------------

    for index, text in enumerate(texts):

        if not isinstance(text, str):
            raise TypeError(
                f"texts[{index}] 不是字符串: "
                f"{type(text)}"
            )

        if not text.strip():
            raise ValueError(
                f"texts[{index}] 不能为空字符串"
            )

    # -----------------------------------------------------
    # 3. 构造 API 参数
    # -----------------------------------------------------

    request_kwargs: dict[str, Any] = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }

    # 只有支持 dimensions 的模型才传入
    if dimensions is not None:
        request_kwargs[
            "dimensions"
        ] = dimensions

    # -----------------------------------------------------
    # 4. 请求 Embedding API
    # -----------------------------------------------------

    response = (
        client.embeddings.create(
            **request_kwargs
        )
    )

    vectors = [
        item.embedding
        for item in response.data
    ]

    # -----------------------------------------------------
    # 5. 数量校验
    # -----------------------------------------------------

    if len(vectors) != len(texts):
        raise RuntimeError(
            "Embedding 返回数量异常："
            f"输入 {len(texts)} 条，"
            f"返回 {len(vectors)} 条。"
        )

    # -----------------------------------------------------
    # 6. 向量维度校验
    #
    # PostgreSQL 当前使用 VECTOR(1024)，
    # 所以必须确保输出维度始终正确。
    # -----------------------------------------------------

    for index, vector in enumerate(
        vectors
    ):

        if len(vector) != EMBEDDING_DIM:

            raise ValueError(
                f"Embedding 维度异常："
                f"model={model}, "
                f"vector[{index}]="
                f"{len(vector)}, "
                f"expected="
                f"{EMBEDDING_DIM}"
            )

    return vectors


# =========================================================
# Memory Embedding
# =========================================================

def embed_memory_texts(
    texts: list[str],
) -> list[list[float]]:
    """
    Long-term Memory 专用 Embedding。

    使用：
        Qwen/Qwen3-Embedding-8B

    注意：
        不要随意更换该模型。

        PostgreSQL 中已经存在的长期记忆向量
        是由该模型生成的。

        如果更换模型，
        旧 Memory 必须全部重新 Embedding。
    """

    return _embed_texts(
        texts,
        model=MEMORY_EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )


# =========================================================
# RAG Embedding
# =========================================================

def embed_rag_texts(
    texts: list[str],
) -> list[list[float]]:
    """
    RAG 知识库专用 Embedding。

    使用：
        BAAI/bge-m3

    BGE-M3 默认输出 1024 维，
    因此这里不传 dimensions 参数。
    """

    return _embed_texts(
        texts,
        model=RAG_EMBEDDING_MODEL,
        dimensions=None,
    )


# =========================================================
# Backward Compatibility
# =========================================================

def embed_texts(
    texts: list[str],
) -> list[list[float]]:
    """
    向后兼容旧代码。

    原 Memory-Agent 已经可能存在：

        embed_texts(...)

    为了避免重构后破坏原有 Memory Store，
    这里继续让 embed_texts 默认走 Memory 模型。

    新代码应明确使用：

        embed_memory_texts()

    或：

        embed_rag_texts()
    """

    return embed_memory_texts(
        texts
    )


__all__ = [
    "EMBEDDING_DIM",
    "MEMORY_EMBEDDING_MODEL",
    "RAG_EMBEDDING_MODEL",
    "embed_texts",
    "embed_memory_texts",
    "embed_rag_texts",
]