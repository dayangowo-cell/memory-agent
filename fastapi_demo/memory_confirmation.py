
from typing import Any


# =========================================================
# Deterministic Memory Confirmation
# =========================================================

STATUS_MESSAGES = {
    "stored": "已成功新增长期记忆。",
    "updated": "已有长期记忆已成功更新。",
    "unchanged": "该信息已存在于长期记忆中，无需重复保存。",
    "ignored": "该信息未保存到长期记忆。",
}


def build_memory_confirmation(
    memory_write_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    根据实际 Memory 写入结果生成程序化确认信息。

    不调用 LLM。
    不分析 LLM 的自然语言回答。
    不执行数据库写入。

    status="none" 表示本轮没有记忆写入操作，
    因此不生成确认信息。

    支持同一个 Turn 中存在多个 Memory 执行结果。
    """

    confirmations: list[dict[str, str]] = []

    for item in memory_write_results:

        status = item["status"]

        # 本轮没有记忆操作
        if status == "none":
            continue

        # 拒绝未知状态
        if status not in STATUS_MESSAGES:
            raise ValueError(
                f"Unsupported memory status: {status}"
            )

        # 依据程序执行状态生成固定确认信息
        confirmation = {
            "status": status,
            "message": STATUS_MESSAGES[status],
        }

        # 关联的 Memory 属性
        relation = item.get("relation")

        if isinstance(relation, str) and relation:
            confirmation["relation"] = relation

        confirmations.append(confirmation)

    return confirmations
