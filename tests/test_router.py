import os

from dotenv import load_dotenv
from langchain_deepseek import (
    ChatDeepSeek,
)

from memory_agent.router import (
    route_query,
)


# =========================================================
# 环境变量
# =========================================================

load_dotenv(
    override=True
)

api_key = os.getenv(
    "DEEPSEEK_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "没有读取到 DEEPSEEK_API_KEY"
    )


# =========================================================
# Model
# =========================================================

model = ChatDeepSeek(
    api_key=api_key,
    model_name="deepseek-v4-flash",
)


# =========================================================
# Test Cases
# =========================================================

test_cases = [
    # -----------------------------------------------------
    # chat
    # -----------------------------------------------------

    {
        "query": "你好，今天怎么样？",
        "expected": "chat",
    },

    {
        "query": "帮我把这句话写得简洁一点。",
        "expected": "chat",
    },

    # -----------------------------------------------------
    # memory
    # -----------------------------------------------------

    {
        "query": "我之前说我的本科学校是什么？",
        "expected": "memory",
    },

    {
        "query": "我现在主要在做什么项目？",
        "expected": "memory",
    },

    # -----------------------------------------------------
    # rag
    # -----------------------------------------------------

    {
        "query": (
            "LangGraph 中 State、"
            "Node 和 Edge 分别是什么？"
        ),
        "expected": "rag",
    },

    {
        "query": "LangGraph 的 Reducer 是什么？",
        "expected": "rag",
    },

    # -----------------------------------------------------
    # memory + rag
    # -----------------------------------------------------

    {
        "query": (
            "根据我目前的学习进度，"
            "结合 LangGraph 资料告诉我"
            "下一步应该学什么？"
        ),
        "expected": "memory_rag",
    },

    {
        "query": (
            "我之前对 Reducer 哪里没理解？"
            "结合我的 LangGraph 资料"
            "重新给我讲一下。"
        ),
        "expected": "memory_rag",
    },
]


# =========================================================
# Run
# =========================================================

passed = 0


for index, case in enumerate(
    test_cases,
    start=1,
):

    print()
    print("=" * 80)

    print(
        f"Case {index}"
    )

    print(
        "Query:",
        case["query"],
    )

    result = route_query(
        case["query"],
        model=model,
    )

    actual = result.mode
    expected = case["expected"]

    is_pass = (
        actual == expected
    )

    if is_pass:
        passed += 1

    print(
        "Expected:",
        expected,
    )

    print(
        "Actual:",
        actual,
    )

    print(
        "Confidence:",
        result.confidence,
    )

    print(
        "Reason:",
        result.reason,
    )

    print(
        "Result:",
        "PASS"
        if is_pass
        else "FAIL",
    )


# =========================================================
# Summary
# =========================================================

print()
print("=" * 80)
print("Router Test Summary")
print("=" * 80)

print(
    f"Passed: {passed}"
    f"/{len(test_cases)}"
)