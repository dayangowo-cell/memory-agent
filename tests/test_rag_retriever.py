from memory_agent.rag.retriever import (
    retrieve_chunks,
)


query = "LangGraph 中 State、Node 和 Edge 分别是什么？"

results = retrieve_chunks(
    query,
    top_k=5,
    category="langgraph",
)


print("=" * 80)
print("RAG Retriever 测试")
print("=" * 80)

print(
    "Query:",
    query,
)

print(
    "Result Count:",
    len(results),
)


for rank, result in enumerate(
    results,
    start=1,
):

    print()
    print("=" * 80)

    print(
        f"Rank: {rank}"
    )

    print(
        f"Score: "
        f"{result.score:.4f}"
    )

    print(
        f"Title: "
        f"{result.title}"
    )

    print(
        f"Page: "
        f"{result.page_number}"
    )

    print(
        f"Chunk: "
        f"{result.chunk_index}"
    )

    print(
        f"Source: "
        f"{result.source_path}"
    )

    print("-" * 80)

    print(
        result.content[:800]
    )