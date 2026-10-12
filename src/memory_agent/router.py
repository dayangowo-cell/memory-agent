"""Agent Query Router。

负责判断当前用户问题应该使用：

- 普通对话
- Long-term Memory
- RAG
- Memory + RAG

Router 只负责决策，不直接执行 Memory / RAG 检索。
"""

from typing import Literal

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field


# =========================================================
# Route Schema
# =========================================================

RouteMode = Literal[
    "chat",
    "memory",
    "rag",
    "memory_rag",
]


class RouteDecision(BaseModel):
    """
    用户问题的路由决策。
    """

    mode: RouteMode = Field(
        description=(
            "当前问题需要使用的上下文来源。"
        )
    )

    reason: str = Field(
        description=(
            "选择该路由的简短原因。"
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "对路由判断的置信度。"
        ),
    )


# =========================================================
# Router Prompt
# =========================================================

ROUTER_PROMPT = """
你是一个 Agent 查询路由器。

你的任务是判断用户当前的问题应该使用哪种信息来源。

可选 mode 只有四种：

1. chat

不需要用户长期记忆，也不需要外部知识库。

例如：
- 你好
- 帮我润色这句话
- 写一个 Python for 循环示例

2. memory

问题的回答依赖用户过去的信息、偏好、经历、
学习状态、项目状态或以前保存的长期记忆。

例如：
- 我之前说我的本科学校是什么？
- 我最喜欢哪个数据库？
- 我现在主要在做什么项目？
- 我之前在哪个知识点上遇到过困难？

3. rag

问题主要是在询问外部知识库中的技术知识，
不需要用户个人历史信息。

例如：
- LangGraph 的 Reducer 是什么？
- RAG 有哪些缺点？
- StateGraph 如何添加节点？
- Checkpointer 和 Store 有什么区别？

4. memory_rag

回答既依赖用户自己的长期记忆，
又需要外部知识库中的知识。

例如：
- 根据我目前的学习进度，结合 LangGraph 资料，
  告诉我接下来应该学什么。
- 我之前对 Reducer 哪里没理解？
  结合我的 LangGraph 资料重新解释。
- 根据我的项目情况，结合知识库给我一个实现方案。

判断原则：

- 不要因为句子出现“我”就自动选择 memory。
- 如果不读取用户过去的信息仍然可以完整回答，
  就不要选择 memory。
- 单纯技术知识问题优先选择 rag。
- 只有真正依赖历史用户信息时才使用 memory。
- 如果用户历史和知识库缺一不可，则使用 memory_rag。
- chat 只用于不需要 Memory 和 RAG 的普通任务。

请严格输出 JSON，格式如下：

{{
    "mode": "chat | memory | rag | memory_rag",
    "reason": "简短说明原因",
    "confidence": 0.0
}}

不要输出 JSON 以外的其他内容。
""".strip()


# =========================================================
# Router
# =========================================================

def route_query(
    query: str,
    *,
    model: BaseChatModel,
) -> RouteDecision:
    """
    根据用户 Query 判断应该走哪条处理路径。

    Router 本身只做语义分类，
    不执行 Memory / RAG 检索。

    Args:
        query:
            当前用户问题。

        model:
            LangChain ChatModel。
            由外部传入，避免 Router 自己重复初始化模型。

    Returns:
        RouteDecision
    """

    # -----------------------------------------------------
    # 1. 参数检查
    # -----------------------------------------------------

    if not query.strip():
        raise ValueError(
            "query 不能为空。"
        )

    # -----------------------------------------------------
    # 2. Structured Output
    #
    # DeepSeek 默认 structured output 可能使用
    # tool calling。
    #
    # 我们之前已经遇到：
    #
    # Thinking mode does not support tool_choice
    #
    # 所以这里明确使用 json_mode。
    # -----------------------------------------------------

    structured_model = (
        model.with_structured_output(
            RouteDecision,
            method="json_mode",
        )
    )

    # -----------------------------------------------------
    # 3. 调用 Router Model
    # -----------------------------------------------------

    result = structured_model.invoke(
        [
            SystemMessage(
                content=ROUTER_PROMPT
            ),
            HumanMessage(
                content=(
                    "请判断下面用户问题应该使用"
                    "哪一种处理模式：\n\n"
                    f"{query}"
                )
            ),
        ]
    )

    # -----------------------------------------------------
    # 4. 防御性检查
    # -----------------------------------------------------

    if not isinstance(
        result,
        RouteDecision,
    ):
        raise RuntimeError(
            "Router 没有返回 RouteDecision。"
        )

    return result

async def aroute_query(
    query: str,
    *,
    model: BaseChatModel,
) -> RouteDecision:
    """
    route_query() 的异步版本。
    专门给 LangGraph 主图里的 async node 使用。
    """

    if not query.strip():
        raise ValueError(
            "query 不能为空。"
        )

    structured_model = (
        model.with_structured_output(
            RouteDecision,
            method="json_mode",
        )
    )

    result = await structured_model.ainvoke(
        [
            SystemMessage(
                content=ROUTER_PROMPT
            ),
            HumanMessage(
                content=(
                    "请判断下面用户问题应该使用"
                    "哪一种处理模式：\n\n"
                    f"{query}"
                )
            ),
        ]
    )

    if not isinstance(
        result,
        RouteDecision,
    ):
        raise RuntimeError(
            "Router 没有返回 RouteDecision。"
        )

    return result


__all__ = [
    "RouteMode",
    "RouteDecision",
    "ROUTER_PROMPT",
    "route_query",
    "aroute_query",
]