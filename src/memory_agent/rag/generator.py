"""RAG Generation 模块。

职责：

用户问题
    ↓
Retriever
    ↓
格式化知识库 Context
    ↓
LLM
    ↓
带来源引用的答案

注意：
本模块不负责文档导入和向量存储。
"""

from dataclasses import dataclass

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from memory_agent.rag.retriever import (
    RetrievalResult,
    retrieve_chunks,
)


# =========================================================
# Prompt
# =========================================================

RAG_SYSTEM_PROMPT = """
你是一个基于外部知识库回答问题的技术助手。

请严格遵守以下规则：

1. 优先依据提供的知识库内容回答。
2. 不要编造知识库中不存在的事实。
3. 如果知识库无法支持问题的答案，请明确说明：
   “当前知识库中没有足够信息回答该问题。”
4. 回答应准确、清晰，并保持知识库原有技术术语。
5. 当回答使用某段知识时，在对应内容后标注来源编号，
   例如：
   [S1]
   [S2]
6. 不要伪造来源编号。
7. 来源编号只能使用给定 Context 中已经出现的编号。
""".strip()


# =========================================================
# Result
# =========================================================

@dataclass
class RAGSource:
    """
    最终提供给上层应用的来源信息。
    """

    source_id: str

    title: str

    source_type: str

    source_path: str

    page_number: int | None

    heading_path: str | None

    score: float


@dataclass
class RAGAnswer:
    """
    一次完整 RAG 问答结果。
    """

    question: str

    answer: str

    sources: list[RAGSource]

    retrieved_chunks: list[
        RetrievalResult
    ]


# =========================================================
# Context Formatting
# =========================================================

def format_context(
    results: list[RetrievalResult],
) -> str:
    """
    将 Retriever 返回的 Chunk
    格式化成适合 LLM 阅读的 Context。

    每一条 Chunk 都分配唯一编号：

        [S1]
        [S2]
        ...
    """

    blocks: list[str] = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        source_id = f"S{index}"

        metadata_parts = [
            f"title={result.title}",
            f"type={result.source_type}",
        ]

        if result.page_number is not None:
            metadata_parts.append(
                f"page={result.page_number}"
            )

        if result.heading_path:
            metadata_parts.append(
                f"heading={result.heading_path}"
            )

        metadata_parts.append(
            f"score={result.score:.4f}"
        )

        metadata_text = ", ".join(
            metadata_parts
        )

        block = f"""
[{source_id}]
Metadata: {metadata_text}

Content:
{result.content}
""".strip()

        blocks.append(block)

    return "\n\n".join(blocks)


# =========================================================
# Source Conversion
# =========================================================

def build_sources(
    results: list[RetrievalResult],
) -> list[RAGSource]:
    """
    将 Retriever 结果转换成
    面向最终应用的来源对象。
    """

    sources: list[RAGSource] = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        sources.append(
            RAGSource(
                source_id=f"S{index}",
                title=result.title,
                source_type=result.source_type,
                source_path=result.source_path,
                page_number=(
                    result.page_number
                ),
                heading_path=(
                    result.heading_path
                ),
                score=result.score,
            )
        )

    return sources


# =========================================================
# Main RAG
# =========================================================

def answer_with_rag(
    question: str,
    *,
    model: BaseChatModel,
    top_k: int = 5,
    category: str | None = None,
) -> RAGAnswer:
    """
    完整 RAG 问答流程。

    Args:
        question:
            用户问题。

        model:
            外部传入的 LangChain ChatModel。

            generator.py 不自行创建模型，
            避免和 Agent 主流程产生重复配置。

        top_k:
            Retriever 返回的 Chunk 数量。

        category:
            可选知识库分类，例如：
                langchain
                langgraph

    Returns:
        RAGAnswer
    """

    if not question.strip():
        raise ValueError(
            "question 不能为空。"
        )

    # -----------------------------------------------------
    # 1. Retrieval
    # -----------------------------------------------------

    results = retrieve_chunks(
        question,
        top_k=top_k,
        category=category,
    )

    # -----------------------------------------------------
    # 2. 没有检索结果
    # -----------------------------------------------------

    if not results:

        return RAGAnswer(
            question=question,
            answer=(
                "当前知识库中没有足够信息"
                "回答该问题。"
            ),
            sources=[],
            retrieved_chunks=[],
        )

    # -----------------------------------------------------
    # 3. Context
    # -----------------------------------------------------

    context = format_context(
        results
    )

    # -----------------------------------------------------
    # 4. User Prompt
    # -----------------------------------------------------

    user_prompt = f"""
请根据下面的知识库内容回答用户问题。

<knowledge_base>
{context}
</knowledge_base>

用户问题：
{question}

回答要求：

- 先直接回答问题。
- 必要时进行解释。
- 使用知识库内容时标注对应来源，例如 [S1]。
- 如果知识库不足以支撑答案，不要自行补充不存在的信息。
""".strip()

    # -----------------------------------------------------
    # 5. Generation
    # -----------------------------------------------------

    response = model.invoke(
        [
            SystemMessage(
                content=RAG_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]
    )

    answer = (
        response.content
        if isinstance(
            response.content,
            str,
        )
        else str(
            response.content
        )
    )

    # -----------------------------------------------------
    # 6. Return
    # -----------------------------------------------------

    return RAGAnswer(
        question=question,
        answer=answer,
        sources=build_sources(
            results
        ),
        retrieved_chunks=results,
    )


__all__ = [
    "RAGAnswer",
    "RAGSource",
    "answer_with_rag",
    "format_context",
]