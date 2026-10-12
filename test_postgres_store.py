import asyncio
import os
import sys

from dotenv import load_dotenv

# ① 必须先加载 .env
load_dotenv()

# ② .env 加载后，再导入会读取环境变量的模块
from langgraph.store.postgres import AsyncPostgresStore
from memory_agent.embeddings import embed_texts
from memory_agent.tools import upsert_memory


# Windows 下 psycopg async 需要 SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


async def main():
    db_uri = os.environ["POSTGRES_URI"]

    async with AsyncPostgresStore.from_conn_string(
        db_uri,
        index={
            "dims": 1024,
            "embed": embed_texts,
            "fields": ["content", "context"],
        },
    ) as store:

        result = await upsert_memory(
            content="用户最喜欢的水果是芒果",
            context="用户明确要求记住这个偏好",
            user_id="debug_user",
            store=store,
        )

        print("工具返回：", result)

        memories = await store.asearch(
            ("memories", "debug_user"),
            query="用户喜欢什么水果？",
            limit=5,
        )

        print("检索结果：")
        for memory in memories:
            print(memory)


asyncio.run(main())