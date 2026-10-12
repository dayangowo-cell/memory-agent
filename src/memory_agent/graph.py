"""Memory Agent V1 graph.

V1 memory pipeline:

User message
    ↓
Structured Memory Extraction
    ↓
Memory Gate
    ↓
Controlled ADD / UPDATE / IGNORE
    ↓
Long-term Store
    ↓
Normal response generation
"""
import json
import logging
import uuid
from datetime import datetime
from typing import Any, cast
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent.router import aroute_query
from memory_agent.rag.retriever import retrieve_chunks
from memory_agent.rag.generator import format_context

from memory_agent import prompts, tools, utils
from memory_agent.context import Context
from memory_agent.memory_gate import evaluate_memory
from memory_agent.memory_schema import MemoryExtraction
from memory_agent.state import State
from memory_agent.memory_resolver import resolve_memory_action

logger = logging.getLogger(__name__)


# =========================================================
# Helper: 获取当前最新的一条用户消息
# =========================================================

def get_latest_user_message(state: State) -> str:
    """
    从 State.messages 中倒序寻找最新 HumanMessage。

    不再使用最后 3 条任意消息作为检索 query，
    避免 AIMessage / ToolMessage 污染检索。
    """

    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            return str(message.content)

    return ""


# =========================================================
# Helper: 格式化已有长期记忆
# =========================================================

def format_memories(memories) -> str:
    """
    将已有长期记忆格式化为提供给 Memory Extractor 的文本。

    V1.1 中不再向 Extractor 暴露数据库 memory_id，
    因为 ADD / UPDATE / NOOP 已交给 Resolver 决定。
    """

    if not memories:
        return "(none)"

    return "\n".join(
        (
            memory.value.get("content", str(memory.value))
            if isinstance(memory.value, dict)
            else str(memory.value)
        )
        for memory in memories
    )


# =========================================================
# Node 1: Structured Memory Extraction
# =========================================================

async def extract_memories(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """
    从当前 User Turn 中抽取结构化 MemoryCandidate。

    注意：

    这一节点只负责：
        “用户说了哪些可能值得记忆的事实？”

    它不负责：
        “最终到底要不要存？”

    STORE / IGNORE 交给下一层 Memory Gate。
    """

    user_id = runtime.context.user_id

    latest_user_message = get_latest_user_message(state)

    if not latest_user_message:
        return {
            "memory_candidates": []
        }

    store = cast(
        BaseStore,
        runtime.store,
    )

    # -----------------------------------------------------
    # 使用当前 User Message 检索旧 Memory
    #
    # 主要目的：
    # 1. 判断当前事实是否是已有事实的更新
    # 2. 给 extractor 提供 memory_id
    # -----------------------------------------------------

    memories = await store.asearch(
        ("memories", user_id),
        query=latest_user_message,
        limit=10,
    )

    formatted_memories = format_memories(
        memories
    )

    # -----------------------------------------------------
    # 给 Extractor 一小段最近对话
    #
    # 这里允许上下文帮助理解：
    # “对，就是那个”
    # “现在改成 MongoDB 了”
    #
    # 但长期记忆检索 query 本身仍然只使用
    # 最新 User Message。
    # -----------------------------------------------------

    recent_messages = state.messages[-5:]

    conversation = "\n".join(
        (
            f"{getattr(message, 'type', 'message')}: "
            f"{message.content}"
        )
        for message in recent_messages
    )

    extraction_prompt = (
        prompts.MEMORY_EXTRACTION_PROMPT.format(
            existing_memories=formatted_memories,
            conversation=conversation,
            current_user_message=latest_user_message,
        )
    )

    llm = utils.load_chat_model(
        runtime.context.model
    )

    # -----------------------------------------------------
    # 关键变化：
    #
    # V0:
    # llm.bind_tools([upsert_memory])
    #
    # V1:
    # llm.with_structured_output(MemoryExtraction)
    #
    # LLM 必须输出我们定义好的结构，
    # 不再自由决定是否调用数据库写入工具。
    # -----------------------------------------------------

    structured_llm = llm.with_structured_output(
        MemoryExtraction,
        method="json_mode",
    )

    extraction = await structured_llm.ainvoke(
        extraction_prompt
    )

    candidates = [
        candidate.model_dump()
        for candidate in extraction.memories
    ]

    logger.info(
        "Extracted %s memory candidate(s)",
        len(candidates),
    )

    return {
        "memory_candidates": candidates
    }


# =========================================================
# Node 2: Memory Gate
# =========================================================

async def gate_memories(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """
    对 Extractor 得到的候选 Memory
    进行确定性的程序规则判断。

    输出：
        store / ignore
        +
        reason
    """

    decisions: list[dict[str, Any]] = []

    for candidate_dict in state.memory_candidates:

        # 重新经过 Pydantic 校验
        from memory_agent.memory_schema import MemoryCandidate

        candidate = MemoryCandidate.model_validate(
            candidate_dict
        )

        result = evaluate_memory(
            candidate
        )

        decision = {
            "candidate":
                candidate.model_dump(),

            "decision":
                result.decision,

            "reason":
                result.reason,
        }

        decisions.append(
            decision
        )

        logger.info(
            "Memory Gate: relation=%s, "
            "decision=%s, reason=%s",
            candidate.relation,
            result.decision,
            result.reason,
        )

    return {
        "memory_decisions": decisions
    }

async def resolve_memories(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """
    对已经通过 Memory Gate 的候选记忆，
    使用数据库中的结构化字段决定：

    ADD
    UPDATE
    NOOP

    这一阶段不调用 LLM。
    """

    store = cast(
        BaseStore,
        runtime.store,
    )

    user_id = runtime.context.user_id

    actions: list[dict[str, Any]] = []

    for item in state.memory_decisions:

        # Gate 已经拒绝的 Candidate，
        # 不需要进入 Resolver。
        if item["decision"] != "store":
            continue

        candidate = item["candidate"]

        subject = candidate["subject"]
        relation = candidate["relation"]

        # =================================================
        # 关键变化：
        #
        # 这里不是向量相似度搜索，
        # 而是结构化精确过滤。
        #
        # 找：
        # 当前 user_id 下
        # subject 相同
        # relation 相同
        # 的 Memory。
        # =================================================

        existing_memories = await store.asearch(
            ("memories", user_id),
            filter={
                "subject": subject,
                "relation": relation,
            },
            limit=20,
        )

        result = resolve_memory_action(
            candidate,
            existing_memories,
        )

        actions.append(
            {
                "candidate": candidate,
                "action": result.action,
                "memory_id": result.memory_id,
                "reason": result.reason,
            }
        )

        logger.info(
            "Memory Resolver: "
            "subject=%s relation=%s "
            "action=%s reason=%s",
            subject,
            relation,
            result.action,
            result.reason,
        )

    return {
        "memory_actions": actions
    }
# =========================================================
# Node 3: Controlled Memory Storage
# =========================================================

async def store_memories(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """
    Execute the deterministic Memory Resolver results.

    Possible final statuses:

        stored
        updated
        unchanged
        ignored
        none

    memory_write_results is the authoritative record of what
    actually happened during the current turn.
    """

    store = cast(
        BaseStore,
        runtime.store,
    )

    user_id = runtime.context.user_id

    write_results: list[dict[str, Any]] = []

    # -------------------------------------------------
    # 1. Record candidates rejected by Memory Gate
    # -------------------------------------------------

    for decision in state.memory_decisions:

        if decision["decision"] != "ignore":
            continue

        candidate = decision["candidate"]

        write_results.append(
            {
                "status": "ignored",
                "subject": candidate.get("subject"),
                "relation": candidate.get("relation"),
                "value": candidate.get("value"),
                "reason": decision.get(
                    "reason",
                    "Rejected by Memory Gate.",
                ),
            }
        )

    # -------------------------------------------------
    # 2. Execute Resolver actions
    # -------------------------------------------------

    for item in state.memory_actions:

        action = item["action"]
        candidate = item["candidate"]

        subject = candidate["subject"]
        relation = candidate["relation"]
        value = candidate["value"]

        # -------------------------------------------------
        # NOOP
        #
        # 数据库中已经存在相同事实。
        # 没有发生新的数据库写入。
        # -------------------------------------------------

        if action == "noop":

            logger.info(
                "Skip memory write: NOOP, "
                "relation=%s value=%s",
                relation,
                value,
            )

            write_results.append(
                {
                    "status": "unchanged",
                    "subject": subject,
                    "relation": relation,
                    "value": value,
                    "reason": item.get(
                        "reason",
                        "Equivalent memory already exists.",
                    ),
                }
            )

            continue

        # -------------------------------------------------
        # ADD / UPDATE
        # -------------------------------------------------

        if action not in {
            "add",
            "update",
        }:
            raise ValueError(
                f"Unsupported memory action: {action}"
            )

        if subject == "user":
            content = (
                f"User's {relation}: {value}"
            )
        else:
            content = (
                f"{subject}'s {relation}: {value}"
            )

        await tools.upsert_memory(
            subject=subject,
            relation=relation,
            value=value,
            memory_type=candidate["memory_type"],
            importance=candidate["importance"],
            confidence=candidate["confidence"],
            content=content,
            context=candidate.get(
                "context",
                "",
            ),
            memory_id=item["memory_id"],
            user_id=user_id,
            store=store,
        )

        status = (
            "stored"
            if action == "add"
            else "updated"
        )

        write_results.append(
            {
                "status": status,
                "subject": subject,
                "relation": relation,
                "value": value,
                "memory_id": item["memory_id"],
                "reason": item.get(
                    "reason",
                    "",
                ),
            }
        )

        logger.info(
            "Memory write completed: "
            "action=%s relation=%s value=%s",
            action,
            relation,
            value,
        )

    # -------------------------------------------------
    # 3. No Memory operation occurred this turn
    # -------------------------------------------------

    if not write_results:

        write_results.append(
            {
                "status": "none",
                "reason": (
                    "No memory was written during "
                    "the current turn."
                ),
            }
        )

    return {
        "memory_write_results": write_results
    }


# =========================================================
# Node 4: Query Router
# =========================================================

async def route_query_node(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """
    判断最终回答需要哪种上下文：

    chat
    memory
    rag
    memory_rag

    注意：
    Router 控制的是 Memory Read / RAG Read，
    不控制前面的 Memory Write。
    """

    latest_user_message = (
        get_latest_user_message(
            state
        )
    )

    if not latest_user_message:
        return {
            "route_mode": "chat",
            "route_reason": (
                "没有找到用户消息"
            ),
            "route_confidence": 1.0,
            "memory_context": "",
            "rag_context": "",
            "rag_sources": [],
        }

    llm = utils.load_chat_model(
        runtime.context.model
    )

    decision = await aroute_query(
        latest_user_message,
        model=llm,
    )

    logger.info(
        "Query Router: "
        "mode=%s confidence=%.2f reason=%s",
        decision.mode,
        decision.confidence,
        decision.reason,
    )

    # -----------------------------------------------------
    # 非常重要：
    #
    # 每轮都主动清空上一轮 Response Context。
    #
    # LangGraph Thread 中 State 会持续存在。
    # 如果不清空：
    #
    # 第1轮 rag
    # 第2轮 chat
    #
    # 第2轮可能错误继承第1轮 rag_context。
    # -----------------------------------------------------

    return {
        "route_mode": decision.mode,
        "route_reason": decision.reason,
        "route_confidence": (
            decision.confidence
        ),
        "memory_context": "",
        "rag_context": "",
        "rag_sources": [],
    }

# =========================================================
# Helper: Memory Read
# =========================================================

async def fetch_memory_context(
    state: State,
    runtime: Runtime[Context],
) -> str:
    """
    根据当前问题读取相关长期 Memory。
    """

    query = get_latest_user_message(
        state
    )

    if not query:
        return ""

    store = cast(
        BaseStore,
        runtime.store,
    )

    memories = await store.asearch(
        (
            "memories",
            runtime.context.user_id,
        ),
        query=query,
        limit=10,
    )

    formatted = format_memories(
        memories
    )

    if formatted == "(none)":
        return ""

    return formatted


# =========================================================
# Helper: RAG Read
# =========================================================

async def fetch_rag_context(
    state: State,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    """
    从 pgvector 知识库中读取 Top-K Chunk。

    当前 retrieve_chunks() 是同步实现，
    因此使用 asyncio.to_thread()
    避免阻塞 LangGraph async event loop。
    """

    query = get_latest_user_message(
        state
    )

    if not query:
        return "", []

    results = await asyncio.to_thread(
        retrieve_chunks,
        query,
        top_k=5,
    )

    if not results:
        return "", []

    context = format_context(
        results
    )

    sources = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        sources.append(
            {
                "source_id": (
                    f"S{index}"
                ),
                "title":
                    result.title,
                "source_type":
                    result.source_type,
                "source_path":
                    result.source_path,
                "page_number":
                    result.page_number,
                "heading_path":
                    result.heading_path,
                "score":
                    result.score,
            }
        )

    return context, sources

async def prepare_chat_context(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """
    普通聊天：
    不读取 Memory，
    不读取 RAG。
    """

    return {
        "memory_context": "",
        "rag_context": "",
        "rag_sources": [],
    }

async def prepare_memory_context(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:

    memory_context = (
        await fetch_memory_context(
            state,
            runtime,
        )
    )

    return {
        "memory_context":
            memory_context,
        "rag_context": "",
        "rag_sources": [],
    }

async def prepare_rag_context(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:

    rag_context, sources = (
        await fetch_rag_context(
            state
        )
    )

    return {
        "memory_context": "",
        "rag_context":
            rag_context,
        "rag_sources":
            sources,
    }

async def prepare_memory_rag_context(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """
    Memory 和 RAG 互不依赖，
    因此可以并发读取。
    """

    memory_task = (
        fetch_memory_context(
            state,
            runtime,
        )
    )

    rag_task = (
        fetch_rag_context(
            state
        )
    )

    memory_context, rag_result = (
        await asyncio.gather(
            memory_task,
            rag_task,
        )
    )

    rag_context, sources = (
        rag_result
    )

    return {
        "memory_context":
            memory_context,
        "rag_context":
            rag_context,
        "rag_sources":
            sources,
    }



# =========================================================
# Node 4: Normal Chat Response
# =========================================================

# =========================================================
# Final Node: Response Generation
# =========================================================

async def call_model(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, list[Any]]:
    """
    最终回答节点。

    本节点不再主动检索 Memory / RAG。

    它只消费 Router 已经准备好的：

        memory_context
        rag_context
    """

    # -----------------------------------------------------
    # Memory Context
    # -----------------------------------------------------

    formatted_memory = ""

    if state.memory_context:

        formatted_memory = f"""
<memories>
{state.memory_context}
</memories>
""".strip()

    # -----------------------------------------------------
    # 原系统 Prompt
    # -----------------------------------------------------

    system_prompt = (
        runtime.context.system_prompt
    )

    sys = system_prompt.format(
        user_info=formatted_memory,
        time=datetime.now().isoformat(),
    )

    # -----------------------------------------------------
    # Memory Write Result
    #
    # 这是当前 Turn 中 Memory Write Pipeline 的真实执行结果。
    # 最终回答必须服从这里的状态，不能自行声称已经保存 Memory。
    # -----------------------------------------------------

    memory_write_result = json.dumps(
        state.memory_write_results,
        ensure_ascii=False,
        indent=2,
    )

    sys += f"""

    <memory_write_result>
    {memory_write_result}
    </memory_write_result>

    The memory_write_result above is authoritative.

    Rules:

    1. status="stored":
       The memory was successfully added.
       You may confirm that it was saved.

    2. status="updated":
       An existing memory was successfully updated.
       You may confirm that it was updated.

    3. status="unchanged":
       The information already existed.
       Do not claim that a new memory was written.

    4. status="ignored":
       The information was not stored.
       Never say that you remembered, saved, stored,
       or recorded this information.

    5. status="none":
       No memory was written during this turn.
       Never claim that memory persistence occurred.

    6. If multiple results exist, describe them according
       to their individual statuses.

    Never contradict memory_write_result.
    """

    # -----------------------------------------------------
    # RAG Context
    # -----------------------------------------------------

    if state.rag_context:

        sys += f"""

<knowledge_base>
{state.rag_context}
</knowledge_base>

回答知识库相关内容时：

1. 优先依据 knowledge_base 中提供的信息。
2. 不要编造知识库中不存在的事实。
3. 使用某个知识片段时，
   在相应内容后标注来源编号，
   例如 [S1]、[S2]。
4. 不要使用 knowledge_base 中不存在的来源编号。
5. 如果知识库信息不足，
   明确说明知识库中缺少足够信息。
"""

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    llm = utils.load_chat_model(
        runtime.context.model
    )

    msg = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": sys,
            },
            *state.messages,
        ]
    )

    return {
        "messages": [msg]
    }

def route_response_context(
    state: State,
) -> str:
    """
    根据 Router 的 mode，
    决定读取哪种上下文。
    """

    return state.route_mode



# =========================================================
# Build Graph
# =========================================================

builder = StateGraph(
    State,
    context_schema=Context,
)

builder.add_node(
    "extract_memories",
    extract_memories,
)

builder.add_node(
    "gate_memories",
    gate_memories,
)

builder.add_node(
    "resolve_memories",
    resolve_memories,
)

builder.add_node(
    "store_memories",
    store_memories,
)

builder.add_node(
    "route_query",
    route_query_node,
)

builder.add_node(
    "prepare_chat_context",
    prepare_chat_context,
)

builder.add_node(
    "prepare_memory_context",
    prepare_memory_context,
)

builder.add_node(
    "prepare_rag_context",
    prepare_rag_context,
)

builder.add_node(
    "prepare_memory_rag_context",
    prepare_memory_rag_context,
)

builder.add_node(
    "call_model",
    call_model,
)


builder.add_edge(
    START,
    "extract_memories",
)

builder.add_edge(
    "extract_memories",
    "gate_memories",
)

builder.add_edge(
    "gate_memories",
    "resolve_memories",
)

builder.add_edge(
    "resolve_memories",
    "store_memories",
)

builder.add_edge(
    "store_memories",
    "route_query",
)

builder.add_conditional_edges(
    "route_query",
    route_response_context,
    {
        "chat":
            "prepare_chat_context",

        "memory":
            "prepare_memory_context",

        "rag":
            "prepare_rag_context",

        "memory_rag":
            "prepare_memory_rag_context",
    },
)


builder.add_edge(
    "prepare_chat_context",
    "call_model",
)

builder.add_edge(
    "prepare_memory_context",
    "call_model",
)

builder.add_edge(
    "prepare_rag_context",
    "call_model",
)

builder.add_edge(
    "prepare_memory_rag_context",
    "call_model",
)

builder.add_edge(
    "call_model",
    END,
)

graph = builder.compile()

graph.name = "MemoryAgentV1"


__all__ = [
    "graph"
]