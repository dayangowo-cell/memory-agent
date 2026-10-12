import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

from memory_agent.rag.generator import (
    answer_with_rag,
)


# =========================================================
# 环境变量
# =========================================================

load_dotenv(
    override=True
)


# =========================================================
# Model
# =========================================================

api_key = os.getenv(
    "DEEPSEEK_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "没有读取到 DEEPSEEK_API_KEY"
    )


model = ChatDeepSeek(
    api_key=api_key,
    model_name="deepseek-v4-flash",
)


# =========================================================
# Question
# =========================================================

question = (
    "LangGraph 中 State、Node "
    "和 Edge 分别是什么？"
)


# =========================================================
# RAG
# =========================================================

result = answer_with_rag(
    question,
    model=model,
    top_k=5,
    category="langgraph",
)


# =========================================================
# Output
# =========================================================

print("=" * 80)
print("RAG Generation 测试")
print("=" * 80)

print()

print(
    "Question:"
)

print(
    result.question
)

print()

print(
    "Answer:"
)

print(
    result.answer
)


print()
print("=" * 80)
print("Sources")
print("=" * 80)


for source in result.sources:

    print()

    print(
        f"[{source.source_id}]"
    )

    print(
        "Title:",
        source.title,
    )

    print(
        "Score:",
        f"{source.score:.4f}",
    )

    if (
        source.page_number
        is not None
    ):
        print(
            "Page:",
            source.page_number,
        )

    if source.heading_path:
        print(
            "Heading:",
            source.heading_path,
        )

    print(
        "Source:",
        source.source_path,
    )