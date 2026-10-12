"""Define the shared graph state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass(kw_only=True)
class State:
    """Memory + RAG Agent shared graph state."""

    # =====================================================
    # Conversation
    # =====================================================

    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    # =====================================================
    # Memory Write Pipeline
    # =====================================================

    memory_candidates: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    memory_decisions: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    memory_actions: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    # Memory 写入阶段的最终事实。
    #
    # status:
    #   stored    -> ADD 已成功写入
    #   updated   -> UPDATE 已成功写入
    #   unchanged -> NOOP，数据库已有相同事实
    #   ignored   -> Memory Gate 拒绝写入
    #   none      -> 本轮没有产生可写入的 Memory
    memory_write_results: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    # =====================================================
    # Query Router
    # =====================================================

    route_mode: str = "chat"

    route_reason: str = ""

    route_confidence: float = 0.0

    # =====================================================
    # Response Context
    #
    # 这里是回答阶段读取出来的上下文，
    # 与上面的 Memory Write Pipeline 是两套不同机制。
    # =====================================================

    memory_context: str = ""

    rag_context: str = ""

    rag_sources: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )


__all__ = [
    "State",
]