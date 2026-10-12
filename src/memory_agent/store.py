import os
from contextlib import asynccontextmanager

from langgraph.store.postgres import AsyncPostgresStore

from memory_agent.embeddings import embed_texts


@asynccontextmanager
async def generate_store():
    """
    创建 PostgreSQL 长期记忆 Store。

    memory-agent 写入的长期记忆最终会持久化到 memory_agent_db。
    """
    db_uri = os.environ["POSTGRES_URI"]

    async with AsyncPostgresStore.from_conn_string(
        db_uri,
        index={
            "dims": 1024,
            "embed": embed_texts,
            "fields": ["content", "context"],
        },
    ) as store:

        # 第一次使用时创建 LangGraph Store 所需的数据表和索引
        # await store.setup()

        yield store