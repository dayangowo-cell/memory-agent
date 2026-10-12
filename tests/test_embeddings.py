from memory_agent.embeddings import (
    EMBEDDING_DIM,
    MEMORY_EMBEDDING_MODEL,
    RAG_EMBEDDING_MODEL,
    embed_rag_texts,
)


text = """
LangGraph 是一个用于构建有状态 Agent
和复杂工作流的底层编排框架。
"""


print("=" * 80)
print("RAG Embedding 测试")
print("=" * 80)

print(
    "模型：",
    RAG_EMBEDDING_MODEL,
)

vectors = embed_rag_texts(
    [text]
)

print(
    "向量数量：",
    len(vectors),
)

print(
    "向量维度：",
    len(vectors[0]),
)

print(
    "期望维度：",
    EMBEDDING_DIM,
)

print(
    "前 10 个值："
)

print(
    vectors[0][:10]
)