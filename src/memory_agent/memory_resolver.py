"""决定候选记忆应该 ADD、UPDATE 还是 NOOP。"""

from dataclasses import dataclass
from typing import Any, Literal


MemoryAction = Literal[
    "add",
    "update",
    "noop",
]


@dataclass
class ResolveResult:
    action: MemoryAction
    memory_id: str | None
    reason: str


# =========================================================
# 单值槽位注册表
#
# 对同一个 user + subject + relation，
# 这些 relation 同一时间通常只保留一个当前值。
#
# 注意：
# 不要写 education.school 这种太粗的 relation，
# 否则本科和硕士学校可能互相覆盖。
# =========================================================

SINGLE_VALUE_RELATIONS = {
    "education.bachelor.school",
    "education.bachelor.degree",
    "education.master.school",
    "education.master.degree",
    "education.doctor.school",
    "education.doctor.degree",

    "job.current.company",
    "job.current.role",

    "preference.database.primary",
    "preference.ide.primary",
}


def normalize_text(value: str) -> str:
    """
    对文本做最基础的归一化，
    用于判断两个 value 是否相同。
    """

    return " ".join(
        str(value).strip().lower().split()
    )


def resolve_memory_action(
    candidate: dict[str, Any],
    existing_memories: list[Any],
) -> ResolveResult:
    """
    根据候选事实和数据库中同槽位的已有事实，
    确定执行动作。

    existing_memories 已经由上游按照：
        subject + relation
    精确过滤。
    """

    candidate_value = candidate["value"]

    # -----------------------------------------------------
    # 1. 相同事实已经存在 → NOOP
    # -----------------------------------------------------

    for memory in existing_memories:

        old_value = memory.value.get("value")

        if old_value is None:
            continue

        if normalize_text(old_value) == normalize_text(
            candidate_value
        ):
            return ResolveResult(
                action="noop",
                memory_id=str(memory.key),
                reason="same_fact_already_exists",
            )

    # -----------------------------------------------------
    # 2. 单值槽位已有旧值 → UPDATE
    # -----------------------------------------------------

    relation = candidate["relation"]

    if (
        relation in SINGLE_VALUE_RELATIONS
        and existing_memories
    ):
        old_memory = existing_memories[0]

        return ResolveResult(
            action="update",
            memory_id=str(old_memory.key),
            reason="single_value_relation_changed",
        )

    # -----------------------------------------------------
    # 3. 其他情况 → ADD
    # -----------------------------------------------------

    return ResolveResult(
        action="add",
        memory_id=None,
        reason="new_fact",
    )