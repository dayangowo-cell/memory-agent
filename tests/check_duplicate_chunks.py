from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from memory_agent.rag.loaders import load_document
from memory_agent.rag.splitter import split_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]


FILES = [
    (
        PROJECT_ROOT
        / "knowledge_base"
        / "langgraph"
        / "03-LangGraph持久化与记忆管理.md"
    ),
    (
        PROJECT_ROOT
        / "knowledge_base"
        / "langgraph"
        / "05-LangGraph高级特性.md"
    ),
]


for path in FILES:

    print()
    print("=" * 80)
    print(path.name)
    print("=" * 80)

    documents = load_document(
        path,
        category="langgraph",
    )

    chunks = split_documents(
        documents
    )

    hash_to_chunks = defaultdict(list)

    for chunk in chunks:

        content_hash = sha256(
            chunk.page_content.encode(
                "utf-8"
            )
        ).hexdigest()

        hash_to_chunks[
            content_hash
        ].append(chunk)

    duplicate_groups = {
        content_hash: items
        for content_hash, items
        in hash_to_chunks.items()
        if len(items) > 1
    }

    duplicate_count = sum(
        len(items) - 1
        for items
        in duplicate_groups.values()
    )

    print(
        "Generated chunks:",
        len(chunks),
    )

    print(
        "Unique chunks:",
        len(hash_to_chunks),
    )

    print(
        "Duplicate chunks:",
        duplicate_count,
    )

    for content_hash, items in (
        duplicate_groups.items()
    ):

        print()
        print(
            "Hash:",
            content_hash,
        )

        print(
            "Occurrences:",
            len(items),
        )

        print(
            "Chunk indexes:",
            [
                item.metadata.get(
                    "chunk_index"
                )
                for item in items
            ],
        )

        print(
            "Heading:",
            items[0].metadata.get(
                "heading_path"
            ),
        )

        print(
            "Content preview:"
        )

        print(
            items[0].page_content[:300]
        )