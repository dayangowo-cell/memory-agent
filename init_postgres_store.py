import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langgraph.store.postgres import AsyncPostgresStore
from memory_agent.embeddings import embed_texts


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

        await store.setup()

        print("PostgreSQL Store 初始化完成")


asyncio.run(main())