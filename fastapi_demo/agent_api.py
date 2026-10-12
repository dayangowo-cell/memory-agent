
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from langgraph_sdk import get_client
from pydantic import BaseModel

from .memory_confirmation import build_memory_confirmation


logger = logging.getLogger(__name__)


# =========================================================
# Router
# =========================================================

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


# =========================================================
# LangGraph Client
# =========================================================

LANGGRAPH_URL = os.getenv(
    "LANGGRAPH_URL",
    "http://127.0.0.1:2025",
)

client = get_client(
    url=LANGGRAPH_URL
)


# =========================================================
# Request / Response Models
# =========================================================

class AgentRequest(BaseModel):
    user_id: str
    message: str
    thread_id: str | None = None

    # 是否为专门的长期记忆操作请求
    #
    # False：
    # 普通聊天，保留 LLM 的自然语言回答。
    #
    # True：
    # 最终 answer 由 Memory 实际执行结果决定，
    # 不允许 LLM 自行确认保存是否成功。
    #
    # 默认为 False，保持原有客户端兼容性。
    memory_command: bool = False


class AgentResponse(BaseModel):
    thread_id: str

    # 最终返回给客户端的回答
    answer: str

    # Query Router 选择的路由
    route: str

    # RAG 检索来源
    sources: list[dict[str, Any]]

    # LangGraph Memory Pipeline 的实际执行结果
    #
    # stored    -> 新增成功
    # updated   -> 更新成功
    # unchanged -> 已有相同记忆，无需重复写入
    # ignored   -> Memory Gate 拒绝写入
    # none      -> 本轮没有记忆写入
    memory_write_results: list[dict[str, Any]]

    # FastAPI 根据 Memory 执行状态生成的确认信息
    #
    # 不依赖 LLM 生成的回答。
    memory_confirmation: list[dict[str, str]]


# =========================================================
# Health Check
# =========================================================

@router.get("/health")
async def health():
    return {
        "message": "Memory Agent FastAPI is running"
    }


# =========================================================
# Agent Chat
# =========================================================

@router.post(
    "/chat",
    response_model=AgentResponse,
)
async def agent_chat(
    request: AgentRequest,
):
    try:

        # -------------------------------------------------
        # 1. Thread Management
        #
        # 不传 thread_id：创建新对话
        # 传入 thread_id：继续已有对话
        # -------------------------------------------------

        if request.thread_id is None:

            thread = await client.threads.create()

            thread_id = thread["thread_id"]

        else:

            thread_id = request.thread_id

        # -------------------------------------------------
        # 2. Invoke LangGraph Agent
        #
        # memory_command 只控制最终回答方式。
        # Memory 是否实际写入，仍由 LangGraph 内部
        # 的 Extractor、Gate、Resolver 和 Storage 决定。
        # -------------------------------------------------

        result = await client.runs.wait(
            thread_id,
            "agent",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": request.message,
                    }
                ]
            },
            context={
                "user_id": request.user_id,
            },
        )

        # -------------------------------------------------
        # 3. Extract Final AI Message
        # -------------------------------------------------

        messages = result.get(
            "messages",
            [],
        )

        if not messages:

            raise HTTPException(
                status_code=502,
                detail="Agent returned no messages",
            )

        last_message = messages[-1]

        if isinstance(last_message, dict):

            answer = str(
                last_message.get(
                    "content",
                    "",
                )
            )

        else:

            answer = str(
                getattr(
                    last_message,
                    "content",
                    last_message,
                )
            )

        # -------------------------------------------------
        # 4. Extract Memory Write Results
        #
        # 数据来自 LangGraph 的 Storage 执行结果，
        # 不能根据 LLM 回答推测是否保存成功。
        # -------------------------------------------------

        memory_write_results = result.get(
            "memory_write_results",
            None,
        )

        # -------------------------------------------------
        # 5. Validate Memory Write Results
        #
        # Fail Closed：
        #
        # 当执行状态缺失、为空或包含非法状态时，
        # 不返回正常的成功响应。
        # -------------------------------------------------

        allowed_statuses = {
            "stored",
            "updated",
            "unchanged",
            "ignored",
            "none",
        }

        if (
            not isinstance(memory_write_results, list)
            or len(memory_write_results) == 0
            or any(
                not isinstance(item, dict)
                or item.get("status") not in allowed_statuses
                for item in memory_write_results
            )
        ):

            raise HTTPException(
                status_code=502,
                detail=(
                    "Agent returned missing or invalid "
                    "memory_write_results"
                ),
            )

        # -------------------------------------------------
        # 6. Deterministic Memory Confirmation
        #
        # 根据 Storage 执行结果生成程序化确认。
        #
        # 不调用 LLM。
        # 不重复执行数据库操作。
        # 不分析 answer 的自然语言。
        # -------------------------------------------------

        memory_confirmation = build_memory_confirmation(
            memory_write_results
        )

        # -------------------------------------------------
        # 7. Deterministic Memory Command Response
        #
        # 当 memory_command=True 时，
        # 将最终 answer 改为程序化确认信息。
        #
        # 防止 LLM 的回答与实际 Memory 状态矛盾。
        # -------------------------------------------------

        if request.memory_command:

            if memory_confirmation:

                # 可能同时产生多个 Memory 执行结果。
                # 将各项确认消息组合为最终回答。
                answer = "\n".join(
                    item["message"]
                    for item in memory_confirmation
                )

            else:

                # status="none" 表示没有执行写入。
                # 不允许将其解释成保存成功。
                answer = (
                    "本轮没有执行长期记忆保存或更新，"
                    "因此无法确认信息已保存。"
                )

        # -------------------------------------------------
        # 8. Extract Route And Sources
        # -------------------------------------------------

        route = result.get(
            "route_mode",
            "",
        )

        sources = result.get(
            "rag_sources",
            [],
        )

        # -------------------------------------------------
        # 9. Return Structured Response
        #
        # memory_command=False：
        # answer 保留 LLM 的回答。
        #
        # memory_command=True：
        # answer 由 Memory 状态确定性生成。
        #
        # memory_write_results：
        # LangGraph 实际执行状态。
        #
        # memory_confirmation：
        # 程序化生成的 Memory 确认信息。
        # -------------------------------------------------

        return AgentResponse(
            thread_id=thread_id,
            answer=answer,
            route=route,
            sources=sources,
            memory_write_results=memory_write_results,
            memory_confirmation=memory_confirmation,
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Agent request failed"
        )

        raise HTTPException(
            status_code=502,
            detail="Agent service request failed",
        ) from exc
