"""RAG 文档加载模块。

负责将不同格式的原始知识库文件统一转换为
LangChain Document。

当前支持：
- PDF
- Markdown
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)


# =========================================================
# PDF Loader
# =========================================================

def load_pdf(
    file_path: str | Path,
    *,
    category: str,
) -> list[Document]:
    """
    加载 PDF 文件。

    PyPDFLoader 默认按页返回 Document，
    因此通常一页对应一个 Document。

    Args:
        file_path:
            PDF 文件路径。

        category:
            知识库分类，例如：
            - langchain
            - langgraph

    Returns:
        list[Document]
    """

    path = Path(file_path)

    # -----------------------------------------------------
    # 1. 检查文件是否存在
    # -----------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"PDF 文件不存在: {path}"
        )

    # -----------------------------------------------------
    # 2. 检查扩展名
    # -----------------------------------------------------

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"不是 PDF 文件: {path}"
        )

    # -----------------------------------------------------
    # 3. 使用 PyPDFLoader 读取 PDF
    #
    # 通常会按页生成多个 Document。
    # -----------------------------------------------------

    loader = PyPDFLoader(
        str(path)
    )

    documents = loader.load()

    # -----------------------------------------------------
    # 4. 给每个 Document 增加统一 metadata
    # -----------------------------------------------------

    for document in documents:

        # PyPDFLoader 中 page 通常从 0 开始
        raw_page = document.metadata.get(
            "page"
        )

        # 转成人类习惯的 1、2、3...
        page_number = (
            raw_page + 1
            if isinstance(raw_page, int)
            else None
        )

        document.metadata.update(
            {
                # 不包含扩展名的文件名
                "title": path.stem,

                # 完整文件名
                "file_name": path.name,

                # 文件路径
                "source_path": str(path),

                # 文件类型
                "source_type": "pdf",

                # langchain / langgraph
                "category": category,

                # PDF 页码
                "page_number": page_number,
            }
        )

    return documents


# =========================================================
# Markdown Loader
# =========================================================

def load_markdown(
    file_path: str | Path,
    *,
    category: str,
) -> list[Document]:
    """
    加载 Markdown 文件。

    当前阶段只负责把完整 Markdown 原文读进来，
    暂时不在 Loader 阶段切分。

    后续会在 splitter.py 中：

    Markdown
        ↓
    按 # / ## / ### 标题切分
        ↓
    再进行更细粒度 Chunking

    Args:
        file_path:
            Markdown 文件路径。

        category:
            知识库分类，例如：
            - langchain
            - langgraph

    Returns:
        list[Document]
    """

    path = Path(file_path)

    # -----------------------------------------------------
    # 1. 检查文件是否存在
    # -----------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"Markdown 文件不存在: {path}"
        )

    # -----------------------------------------------------
    # 2. 检查文件扩展名
    # -----------------------------------------------------

    if path.suffix.lower() not in {
        ".md",
        ".markdown",
    }:
        raise ValueError(
            f"不是 Markdown 文件: {path}"
        )

    # -----------------------------------------------------
    # 3. Markdown 本质上是文本文件
    #
    # 这里暂时使用 TextLoader，
    # 保留原来的：
    #
    # # 标题
    # ## 二级标题
    # ```python
    # 代码块
    # ```
    #
    # 这些结构。
    # -----------------------------------------------------

    loader = TextLoader(
        str(path),
        encoding="utf-8",
        autodetect_encoding=True,
    )

    documents = loader.load()

    # -----------------------------------------------------
    # 4. 增加统一 metadata
    # -----------------------------------------------------

    for document in documents:

        document.metadata.update(
            {
                "title": path.stem,
                "file_name": path.name,
                "source_path": str(path),
                "source_type": "markdown",
                "category": category,

                # Markdown 没有 PDF 页码
                "page_number": None,
            }
        )

    return documents


# =========================================================
# Unified Loader
# =========================================================

def load_document(
    file_path: str | Path,
    *,
    category: str,
) -> list[Document]:
    """
    统一文档加载入口。

    根据文件扩展名自动选择：

        .pdf
            → load_pdf()

        .md / .markdown
            → load_markdown()

    上层 ingestion.py 不需要关心不同文件格式的
    具体 Loader 实现。
    """

    path = Path(file_path)

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(
            path,
            category=category,
        )

    if suffix in {
        ".md",
        ".markdown",
    }:
        return load_markdown(
            path,
            category=category,
        )

    raise ValueError(
        f"当前不支持的文件类型: {suffix}"
    )


__all__ = [
    "load_pdf",
    "load_markdown",
    "load_document",
]