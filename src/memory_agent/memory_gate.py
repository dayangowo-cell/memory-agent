from dataclasses import dataclass
from typing import Literal

from memory_agent.memory_schema import MemoryCandidate


MemoryDecision = Literal["store", "ignore"]


@dataclass
class GateResult:
    """
    Memory Gate 对一个候选记忆作出的最终判断。
    """

    decision: MemoryDecision
    reason: str


def evaluate_memory(
    candidate: MemoryCandidate,
) -> GateResult:
    """
    判断候选事实是否值得进入长期记忆。

    V1 第一版采用明确、可解释的规则，
    后续再根据评测结果逐渐扩展。
    """

    # -----------------------------------------------------
    # 1. 置信度太低，不保存
    # -----------------------------------------------------

    if candidate.confidence < 0.70:
        return GateResult(
            decision="ignore",
            reason="confidence_too_low",
        )

    # -----------------------------------------------------
    # 2. 明确属于临时信息，不进入长期记忆
    # -----------------------------------------------------

    if candidate.memory_type == "temporary":
        return GateResult(
            decision="ignore",
            reason="temporary_information",
        )

    # -----------------------------------------------------
    # 3. 高重要度事实直接保存
    # -----------------------------------------------------

    if candidate.importance == "high":
        return GateResult(
            decision="store",
            reason="high_importance",
        )

    # -----------------------------------------------------
    # 4. 稳定用户画像
    # -----------------------------------------------------

    if candidate.memory_type == "profile":
        return GateResult(
            decision="store",
            reason="stable_profile",
        )

    # -----------------------------------------------------
    # 5. 长期偏好
    # -----------------------------------------------------

    if candidate.memory_type == "preference":
        return GateResult(
            decision="store",
            reason="long_term_preference",
        )

    # -----------------------------------------------------
    # 6. 关系信息
    # -----------------------------------------------------

    if candidate.memory_type == "relationship":
        return GateResult(
            decision="store",
            reason="relationship_information",
        )

    # -----------------------------------------------------
    # 7. 明确目标
    # -----------------------------------------------------

    if (
        candidate.memory_type == "goal"
        and candidate.importance == "medium"
    ):
        return GateResult(
            decision="store",
            reason="meaningful_goal",
        )

    # -----------------------------------------------------
    # 8. 其他低价值信息暂不保存
    # -----------------------------------------------------

    return GateResult(
        decision="ignore",
        reason="insufficient_long_term_value",
    )