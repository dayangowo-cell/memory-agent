"""批量导入项目知识库。

扫描：

knowledge_base/langchain
knowledge_base/langgraph

支持：
- PDF
- Markdown

已经导入且文件内容未变化：
    skipped

新文件：
    ingested

文件发生变化：
    根据 ingestion.py 的逻辑重新处理
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_agent.rag.ingestion import (
    IngestionResult,
    ingest_file,
)


# =========================================================
# Project
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

KNOWLEDGE_BASE = (
    PROJECT_ROOT
    / "knowledge_base"
)


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".md",
    ".markdown",
}


# =========================================================
# Collect Files
# =========================================================

def collect_files(
    directory: Path,
) -> list[Path]:
    """
    递归寻找支持的知识库文件。
    """

    if not directory.exists():
        return []

    files = [
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    ]

    return sorted(files)


# =========================================================
# Import Category
# =========================================================

def ingest_category(
    *,
    category: str,
    directory: Path,
    batch_size: int,
    force: bool,
) -> tuple[
    list[IngestionResult],
    list[tuple[Path, Exception]],
]:
    """
    导入某一个知识库分类。
    """

    files = collect_files(
        directory
    )

    print()
    print("=" * 80)
    print(
        f"Category: {category}"
    )
    print(
        f"Directory: {directory}"
    )
    print(
        f"Files: {len(files)}"
    )
    print("=" * 80)

    results: list[
        IngestionResult
    ] = []

    failures: list[
        tuple[Path, Exception]
    ] = []

    for index, file_path in enumerate(
        files,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(files)}] "
            f"{file_path.name}"
        )

        try:

            result = ingest_file(
                file_path,
                category=category,
                project_root=PROJECT_ROOT,
                batch_size=batch_size,
                force=force,
            )

            results.append(
                result
            )

            print(
                "Result:",
                result.status,
            )

        except Exception as exc:

            failures.append(
                (
                    file_path,
                    exc,
                )
            )

            print(
                "FAILED:",
                repr(exc),
            )

    return results, failures


# =========================================================
# Main
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "批量导入 Memory-Agent RAG 知识库"
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Embedding API 每批处理的 Chunk 数量"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "强制重新导入已经存在的文件"
        ),
    )

    args = parser.parse_args()

    categories = {
        "langchain":
            KNOWLEDGE_BASE
            / "langchain",

        "langgraph":
            KNOWLEDGE_BASE
            / "langgraph",
    }

    all_results = []
    all_failures = []

    for category, directory in (
        categories.items()
    ):

        results, failures = (
            ingest_category(
                category=category,
                directory=directory,
                batch_size=args.batch_size,
                force=args.force,
            )
        )

        all_results.extend(
            results
        )

        all_failures.extend(
            failures
        )

    # =====================================================
    # Summary
    # =====================================================

    ingested = sum(
        result.status == "ingested"
        for result in all_results
    )

    skipped = sum(
        result.status == "skipped"
        for result in all_results
    )

    chunks = sum(
        result.chunk_count
        for result in all_results
    )

    embeddings = sum(
        result.embedding_count
        for result in all_results
    )

    print()
    print("=" * 80)
    print("Knowledge Base Ingestion Summary")
    print("=" * 80)

    print(
        "Ingested:",
        ingested,
    )

    print(
        "Skipped:",
        skipped,
    )

    print(
        "Failed:",
        len(all_failures),
    )

    print(
        "New chunks:",
        chunks,
    )

    print(
        "New embeddings:",
        embeddings,
    )

    if all_failures:

        print()
        print("Failed Files:")

        for file_path, exc in (
            all_failures
        ):

            print(
                "-",
                file_path,
            )

            print(
                " ",
                repr(exc),
            )


if __name__ == "__main__":
    main()