from typing import Literal

from pydantic import BaseModel, Field


class MemoryCandidate(BaseModel):
    """从当前用户消息中抽取出的候选长期记忆。"""

    subject: str = Field(
        description=(
            "事实主体。用户本人填写 user；"
            "其他人物填写 mentor、sister 等稳定名称。"
        )
    )

    relation: str = Field(
        description=(
            "事实关系，例如 education.school、"
            "education.degree、job.role、preference.database。"
        )
    )

    value: str = Field(
        description="事实的核心值。"
    )

    memory_type: Literal[
        "profile",
        "preference",
        "relationship",
        "goal",
        "routine",
        "experience",
        "temporary",
        "other",
    ] = Field(
        description="候选记忆类型。"
    )

    importance: Literal[
        "high",
        "medium",
        "low",
    ] = Field(
        description="长期记忆价值。"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="模型对事实抽取正确性的置信度。",
    )

    context: str = Field(
        default="",
        description="帮助理解该事实的简短上下文。",
    )


class MemoryExtraction(BaseModel):
    """一次用户输入可以抽取 0~N 条候选记忆。"""

    memories: list[MemoryCandidate] = Field(
        default_factory=list
    )