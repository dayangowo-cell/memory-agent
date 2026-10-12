"""长期记忆持久化操作。"""

import uuid

from langgraph.store.base import BaseStore


async def upsert_memory(
    *,
    subject: str,
    relation: str,
    value: str,
    memory_type: str,
    importance: str,
    confidence: float,
    content: str,
    context: str,
    memory_id: str | None = None,
    user_id: str,
    store: BaseStore,
) -> str:
    """
    新增或更新一条结构化长期记忆。

    memory_id is None:
        ADD，新建 UUID。

    memory_id 不为 None:
        UPDATE，复用已有 key。
    """

    mem_id = (
        memory_id
        if memory_id is not None
        else str(uuid.uuid4())
    )

    await store.aput(
        ("memories", user_id),
        key=mem_id,
        value={
            # -------------------------------
            # 机器可计算字段
            # -------------------------------
            "subject": subject,
            "relation": relation,
            "value": value,
            "memory_type": memory_type,
            "importance": importance,
            "confidence": confidence,

            # -------------------------------
            # 给 Embedding / LLM 使用的文本
            # -------------------------------
            "content": content,
            "context": context,
        },
    )

    return f"Stored memory {mem_id}"