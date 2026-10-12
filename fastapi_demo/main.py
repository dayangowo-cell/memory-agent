import asyncio
import os
import time

import psycopg
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from fastapi_demo.agent_api import router as agent_router


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="FastAPI Learning Demo"
)

# Register Memory-Agent routes:
# /agent/health
# /agent/chat
app.include_router(agent_router)

@app.middleware("http")
async def add_utf8_charset(
    request: Request,
    call_next,
):
    response = await call_next(request)

    content_type = response.headers.get(
        "content-type",
        "",
    )

    if (
        content_type.startswith("application/json")
        and "charset=" not in content_type.lower()
    ):
        response.headers["content-type"] = (
            "application/json; charset=utf-8"
        )

    return response

# =========================================================
# Pydantic Models
# =========================================================

class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    message: str
    answer: str


class MessageRequest(BaseModel):
    message: str


# =========================================================
# Fake Data
# =========================================================

fake_users = {
    "001": "张三",
    "002": "李四",
}


# =========================================================
# Database Check
# =========================================================

@app.get("/db-check")
async def db_check():
    """
    测试 FastAPI Container 是否能够通过 Docker Network
    连接 PostgreSQL + pgvector Container。

    Docker Compose 中：

        DB_HOST=db
        DB_PORT=5432
        DB_NAME=memory_agent_db

    因此 FastAPI Container 实际访问：

        db:5432/memory_agent_db

    而不是 localhost:5432。
    """

    host = os.getenv(
        "DB_HOST",
        "localhost",
    )

    port = os.getenv(
        "DB_PORT",
        "5432",
    )

    dbname = os.getenv(
        "DB_NAME",
        "memory_agent_db",
    )

    user = os.getenv(
        "DB_USER",
        "postgres",
    )

    password = os.getenv(
        "DB_PASSWORD",
        "",
    )

    conninfo = (
        f"host={host} "
        f"port={port} "
        f"dbname={dbname} "
        f"user={user} "
        f"password={password}"
    )

    try:
        async with await psycopg.AsyncConnection.connect(
            conninfo
        ) as conn:

            async with conn.cursor() as cur:

                # -------------------------------------------------
                # 查询当前数据库和当前数据库用户
                # -------------------------------------------------

                await cur.execute(
                    """
                    SELECT
                        current_database(),
                        current_user
                    """
                )

                row = await cur.fetchone()

                if row is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to query PostgreSQL connection info",
                    )

                database, current_user = row

                # -------------------------------------------------
                # 查询 Memory-Agent 核心数据量
                # -------------------------------------------------

                await cur.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM rag_documents),
                        (SELECT COUNT(*) FROM rag_chunks),
                        (SELECT COUNT(*) FROM store),
                        (SELECT COUNT(*) FROM store_vectors)
                    """
                )

                count_row = await cur.fetchone()

                if count_row is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to query database counts",
                    )

                (
                    rag_documents_count,
                    rag_chunks_count,
                    store_count,
                    store_vectors_count,
                ) = count_row

                # -------------------------------------------------
                # 查询 pgvector 扩展版本
                # -------------------------------------------------

                await cur.execute(
                    """
                    SELECT extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                    """
                )

                vector_row = await cur.fetchone()

                vector_version = (
                    vector_row[0]
                    if vector_row is not None
                    else None
                )

        return {
            "database": database,
            "user": current_user,
            "db_host": host,
            "db_port": port,
            "pgvector_version": vector_version,
            "rag_documents": rag_documents_count,
            "rag_chunks": rag_chunks_count,
            "store": store_count,
            "store_vectors": store_vectors_count,
        }

    except psycopg.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PostgreSQL connection/query failed: {exc}",
        ) from exc


# =========================================================
# Basic GET
# =========================================================

@app.get("/hello")
async def hello():
    return {
        "message": "Hello FastAPI V4 - Memory Agent Database"
    }


# =========================================================
# Basic POST + Request / Response Model
# =========================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
):
    return {
        "user_id": request.user_id,
        "message": request.message,
        "answer": (
            f"收到你的消息：{request.message}"
        ),
    }


# =========================================================
# Path Parameter + HTTPException
# =========================================================

@app.get("/users/{user_id}")
async def get_user(
    user_id: str,
):
    if user_id not in fake_users:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return {
        "user_id": user_id,
        "name": fake_users[user_id],
    }


# =========================================================
# Query Parameters
# =========================================================

@app.get("/messages")
async def get_messages(
    limit: int = 10,
    offset: int = 0,
):
    return {
        "limit": limit,
        "offset": offset,
    }


# =========================================================
# Path + Query + Request Body
# =========================================================

@app.post("/users/{user_id}/chat")
async def user_chat(
    user_id: str,
    request: MessageRequest,
    debug: bool = False,
):
    return {
        "user_id": user_id,
        "message": request.message,
        "debug": debug,
    }


# =========================================================
# Blocking Demo
# =========================================================

@app.get("/blocking")
async def blocking():
    """
    同步阻塞示例。

    time.sleep() 会阻塞 Event Loop。
    在 async endpoint 中一般不应该这样做。
    """

    time.sleep(5)

    return {
        "message": "blocking finished"
    }


# =========================================================
# Non-blocking Demo
# =========================================================

@app.get("/non-blocking")
async def non_blocking():
    """
    异步非阻塞示例。

    await asyncio.sleep() 会主动让出 Event Loop，
    等待期间服务器仍然可以处理其他请求。
    """

    await asyncio.sleep(5)

    return {
        "message": "non-blocking finished"
    }