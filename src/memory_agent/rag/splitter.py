"""RAG 文档切分模块。

针对不同类型文档采用不同的 Chunk 策略：

PDF:
    保留页码信息
    ↓
    每页进行递归切分

Markdown:
    先按照 # / ## / ### / #### 标题切分
    ↓
    再对过长 Section 进行递归切分

最终统一输出：
    list[Document]
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# =========================================================
# 通用配置
# =========================================================

DEFAULT_CHUNK_SIZE = 1200

DEFAULT_CHUNK_OVERLAP = 150


# =========================================================
# Recursive Splitter
# =========================================================

def create_recursive_splitter(
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    创建通用递归文本切分器。

    separator 的顺序代表切分优先级。

    优先：
        段落
        ↓
        换行
        ↓
        中文句号
        ↓
        中文分号
        ↓
        中文逗号
        ↓
        空格
        ↓
        字符级切分
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        separators=[
            "\n\n",
            "\n",
            "。",
            "；",
            "，",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


# =========================================================
# PDF Splitter
# =========================================================

def split_pdf_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    切分 PDF Document。

    Loader 阶段已经保证：
        一个 Document ≈ PDF 一页

    所以这里不把不同页面拼接起来。

    这样每个 Chunk 都可以继续保留：
        page_number
        title
        source_path
        category
        等 metadata。
    """

    splitter = create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = []

    for page_document in documents:

        # -------------------------------------------------
        # 当前这一页继续细分
        # -------------------------------------------------

        page_chunks = splitter.split_documents(
            [page_document]
        )

        # -------------------------------------------------
        # 增加当前页面内部的 chunk 序号
        # -------------------------------------------------

        for page_chunk_index, chunk in enumerate(
            page_chunks
        ):

            chunk.metadata[
                "page_chunk_index"
            ] = page_chunk_index

            chunk.metadata[
                "heading_path"
            ] = None

            chunks.append(chunk)

    # -----------------------------------------------------
    # 全文统一 chunk_index
    # -----------------------------------------------------

    for chunk_index, chunk in enumerate(chunks):

        chunk.metadata[
            "chunk_index"
        ] = chunk_index

    return chunks


# =========================================================
# Markdown Splitter
# =========================================================

def split_markdown_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Markdown 两阶段切分。

    第一阶段：
        根据 Markdown 标题结构切 Section。

    第二阶段：
        如果某个 Section 太长，
        再进行 RecursiveCharacterTextSplitter 切分。
    """

    # -----------------------------------------------------
    # 1. Markdown 标题结构
    # -----------------------------------------------------

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,

        # False：
        # Chunk 正文里继续保留 Markdown 标题本身
        strip_headers=False,
    )

    recursive_splitter = (
        create_recursive_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )

    final_chunks: list[Document] = []

    # -----------------------------------------------------
    # 2. 逐个 Markdown Document 处理
    # -----------------------------------------------------

    for document in documents:

        # Loader 阶段的 metadata
        base_metadata = document.metadata.copy()

        # -------------------------------------------------
        # 第一层：
        # 根据 Markdown Header 切分
        # -------------------------------------------------

        sections = header_splitter.split_text(
            document.page_content
        )

        for section in sections:

            # ---------------------------------------------
            # MarkdownHeaderTextSplitter 会生成：
            #
            # {
            #   "h1": "...",
            #   "h2": "...",
            #   "h3": "..."
            # }
            #
            # 把它们组合成：
            #
            # LangGraph总览 > StateGraph > Node
            # ---------------------------------------------

            heading_parts = []

            for level in (
                "h1",
                "h2",
                "h3",
                "h4",
            ):
                heading = section.metadata.get(
                    level
                )

                if heading:
                    heading_parts.append(
                        heading
                    )

            heading_path = (
                " > ".join(heading_parts)
                if heading_parts
                else None
            )

            # ---------------------------------------------
            # 合并 Loader metadata + Markdown Header
            # ---------------------------------------------

            merged_metadata = {
                **base_metadata,
                **section.metadata,
                "heading_path": heading_path,
            }

            section_document = Document(
                page_content=section.page_content,
                metadata=merged_metadata,
            )

            # ---------------------------------------------
            # 第二层：
            #
            # Section 仍然可能非常长，
            # 所以继续递归切分。
            # ---------------------------------------------

            section_chunks = (
                recursive_splitter.split_documents(
                    [section_document]
                )
            )

            final_chunks.extend(
                section_chunks
            )

    # -----------------------------------------------------
    # 全文统一编号
    # -----------------------------------------------------

    for chunk_index, chunk in enumerate(
        final_chunks
    ):

        chunk.metadata[
            "chunk_index"
        ] = chunk_index

    return final_chunks


# =========================================================
# Unified Splitter
# =========================================================

def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    根据 source_type 自动选择切分策略。

    PDF：
        split_pdf_documents()

    Markdown：
        split_markdown_documents()
    """

    if not documents:
        return []

    source_type = documents[
        0
    ].metadata.get(
        "source_type"
    )

    if source_type == "pdf":

        return split_pdf_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if source_type == "markdown":

        return split_markdown_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    raise ValueError(
        f"不支持的 source_type: "
        f"{source_type}"
    )


__all__ = [
    "split_pdf_documents",
    "split_markdown_documents",
    "split_documents",
]