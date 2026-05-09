"""
校园文档数据准备模块
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .document_ingestion import load_documents as load_campus_documents
from .document_ingestion import normalize_text

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """数据准备模块 - 负责校园文档加载、清洗和分块"""

    CATEGORY_LABELS = ["规章制度", "教务教学", "校园生活", "通知公告", "其他"]
    DIFFICULTY_LABELS: List[str] = []

    def __init__(self, data_path: str):
        """
        初始化数据准备模块
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self.parent_documents_map: Dict[str, Document] = {}

    def load_documents(self) -> List[Document]:
        """
        加载完整父文档数据
        Returns:
            加载的完整父文档列表
        """
        logger.info("正在从 %s 加载校园文档...", self.data_path)

        documents = load_campus_documents(self.data_path)
        self.documents = documents
        self.chunks = []
        self.parent_child_map = {}
        self.parent_documents_map = {}

        for parent_doc in self.documents:
            self._normalize_parent_metadata(parent_doc)
            parent_id = parent_doc.metadata.get("parent_id")
            if parent_id:
                self.parent_documents_map[parent_id] = parent_doc

        logger.info("成功加载 %s 个父文档", len(self.documents))
        return documents

    def _normalize_parent_metadata(self, parent_doc: Document) -> None:
        """补齐校园文档的标准元数据，并保留少量旧字段别名。"""
        metadata = parent_doc.metadata or {}
        source = metadata.get("source", "")
        source_path = Path(source) if source else Path(self.data_path)

        doc_title = metadata.get("doc_title") or metadata.get("source_name") or source_path.stem or "未知文档"
        doc_category = metadata.get("doc_category") or "其他"
        department = metadata.get("department") or ""
        file_type = metadata.get("file_type") or source_path.suffix.lstrip(".").lower()
        parent_id = metadata.get("parent_id") or metadata.get("doc_id")
        if not parent_id:
            parent_seed = metadata.get("relative_path") or source_path.as_posix() or doc_title
            parent_id = hashlib.md5(parent_seed.encode("utf-8")).hexdigest()

        metadata.update(
            {
                "doc_title": doc_title,
                "doc_category": doc_category,
                "department": department,
                "file_type": file_type,
                "parent_id": parent_id,
                "doc_id": metadata.get("doc_id") or parent_id,
                "section": metadata.get("section") or doc_title,
                "source": source,
                "category": doc_category,
                "dish_name": doc_title,
                "difficulty": "未知",
                "doc_type": "parent",
            }
        )
        parent_doc.metadata = metadata

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """对外提供支持的分类标签列表"""
        return list(cls.CATEGORY_LABELS)

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """对外提供支持的难度标签列表"""
        return list(cls.DIFFICULTY_LABELS)

    def chunk_documents(self) -> List[Document]:
        """
        校园文档结构感知分块
        Returns:
            分块后的文档块列表
        """
        logger.info("正在进行校园文档分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        chunks: List[Document] = []
        self.parent_child_map = {}

        for parent_doc in self.documents:
            parent_chunks = self._split_parent_document(parent_doc)
            for chunk_index, raw_chunk in enumerate(parent_chunks):
                cleaned_content = normalize_text(raw_chunk.page_content or "")
                if not cleaned_content:
                    continue

                chunk_doc = self._build_chunk_document(
                    parent_doc=parent_doc,
                    raw_chunk=raw_chunk,
                    cleaned_content=cleaned_content,
                    chunk_index=chunk_index,
                    batch_index=len(chunks),
                )
                chunks.append(chunk_doc)
                self.parent_child_map[chunk_doc.metadata["chunk_id"]] = chunk_doc.metadata["parent_id"]

        self.chunks = chunks
        logger.info("校园文档分块完成，共生成 %s 个chunk", len(chunks))
        return chunks

    def _split_parent_document(self, parent_doc: Document) -> List[Document]:
        """按文件类型选择合适的切分策略。"""
        file_type = (parent_doc.metadata.get("file_type") or "").lower()

        if file_type == "md":
            return self._split_markdown_document(parent_doc)
        if file_type == "txt":
            return self._split_text_document(
                parent_doc,
                chunk_size=800,
                chunk_overlap=120,
                separators=["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""],
            )
        if file_type == "pdf":
            return self._split_text_document(
                parent_doc,
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""],
            )

        return [Document(page_content=parent_doc.page_content, metadata={})]

    def _split_markdown_document(self, parent_doc: Document) -> List[Document]:
        """Markdown 文档优先按标题切分，没有标题则退回通用分块。"""
        if not self._has_markdown_headers(parent_doc.page_content):
            return self._split_text_document(
                parent_doc,
                chunk_size=800,
                chunk_overlap=120,
                separators=["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""],
            )

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )

        try:
            markdown_chunks = splitter.split_text(parent_doc.page_content)
        except Exception as exc:
            logger.warning("Markdown 结构分割失败: %s, error=%s", parent_doc.metadata.get("source"), exc)
            markdown_chunks = []

        cleaned_chunks: List[Document] = []
        for chunk in markdown_chunks:
            cleaned_content = normalize_text(chunk.page_content or "")
            if not cleaned_content:
                continue
            cleaned_chunks.append(Document(page_content=cleaned_content, metadata=dict(chunk.metadata or {})))

        if cleaned_chunks:
            return cleaned_chunks

        return self._split_text_document(
            parent_doc,
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""],
        )

    def _split_text_document(
        self,
        parent_doc: Document,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
    ) -> List[Document]:
        """使用递归字符分割器切分普通文本。"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

        raw_chunks = splitter.split_text(parent_doc.page_content)
        chunks: List[Document] = []
        for raw_chunk in raw_chunks:
            cleaned_content = normalize_text(raw_chunk)
            if cleaned_content:
                chunks.append(Document(page_content=cleaned_content, metadata={}))

        if chunks:
            return chunks

        fallback_content = normalize_text(parent_doc.page_content)
        if fallback_content:
            return [Document(page_content=fallback_content, metadata={})]

        return []

    def _build_chunk_document(
        self,
        parent_doc: Document,
        raw_chunk: Document,
        cleaned_content: str,
        chunk_index: int,
        batch_index: int,
    ) -> Document:
        """把分块内容转成带标准元数据的子文档。"""
        parent_metadata = dict(parent_doc.metadata or {})
        chunk_metadata = dict(raw_chunk.metadata or {})
        parent_id = parent_metadata.get("parent_id") or parent_metadata.get("doc_id")
        if not parent_id:
            parent_seed = parent_metadata.get("relative_path") or parent_metadata.get("source") or cleaned_content[:128]
            parent_id = hashlib.md5(str(parent_seed).encode("utf-8")).hexdigest()

        chunk_id = self._make_chunk_id(parent_id, chunk_index)
        section = self._infer_section(parent_doc, chunk_metadata)

        metadata = {
            **parent_metadata,
            **chunk_metadata,
            "parent_id": parent_id,
            "doc_id": parent_metadata.get("doc_id") or parent_id,
            "doc_type": "child",
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "batch_index": batch_index,
            "chunk_size": len(cleaned_content),
            "section": section,
            "doc_title": parent_metadata.get("doc_title", "未知文档"),
            "doc_category": parent_metadata.get("doc_category", "其他"),
            "department": parent_metadata.get("department", ""),
            "file_type": parent_metadata.get("file_type", ""),
            "source": parent_metadata.get("source", ""),
            "relative_path": parent_metadata.get("relative_path", ""),
            "page": parent_metadata.get("page"),
            "category": parent_metadata.get("doc_category", "其他"),
            "dish_name": parent_metadata.get("doc_title", "未知文档"),
            "difficulty": "未知",
        }

        return Document(page_content=cleaned_content, metadata=metadata)

    def _infer_section(self, parent_doc: Document, chunk_metadata: Dict[str, Any]) -> str:
        """推断分块的章节标题。"""
        section_parts: List[str] = []
        for key in ("h1", "h2", "h3"):
            value = chunk_metadata.get(key)
            if value:
                section_parts.append(str(value).strip())
        if section_parts:
            return " / ".join(section_parts)

        page = parent_doc.metadata.get("page")
        if page is not None:
            return f"第{page}页"

        return parent_doc.metadata.get("doc_title") or parent_doc.metadata.get("source_name") or "正文"

    @staticmethod
    def _has_markdown_headers(text: str) -> bool:
        """判断文本中是否存在 Markdown 标题。"""
        for line in text.splitlines()[:20]:
            if line.lstrip().startswith("#"):
                return True
        return False

    @staticmethod
    def _make_chunk_id(parent_id: str, chunk_index: int) -> str:
        """生成稳定的文档块ID，便于向量索引和BM25结果去重"""
        raw_id = f"{parent_id}:{chunk_index}"
        return hashlib.md5(raw_id.encode("utf-8")).hexdigest()

    def get_parent_documents(self, retrieved_chunks: List[Document]) -> List[Document]:
        """
        根据检索召回的文档块获取对应的父文档（智能去重）
        Args:
            retrieved_chunks: 检索到的文档块列表
        Returns:
            对应的父文档列表（去重，按最强召回证据排序）
        """
        parent_rank_info: Dict[str, Dict[str, Any]] = {}
        parent_docs_map: Dict[str, Document] = {}

        for chunk_rank, chunk in enumerate(retrieved_chunks):
            chunk_metadata = chunk.metadata or {}
            parent_id = chunk_metadata.get("parent_id")
            if not parent_id:
                continue

            raw_score = chunk_metadata.get("rrf_score")
            try:
                chunk_score = float(raw_score) if raw_score is not None else 0.0
            except (TypeError, ValueError):
                chunk_score = 0.0

            if parent_id not in parent_rank_info:
                parent_rank_info[parent_id] = {
                    "first_rank": chunk_rank,
                    "best_score": chunk_score,
                    "hit_count": 1,
                }
            else:
                rank_info = parent_rank_info[parent_id]
                rank_info["hit_count"] += 1
                rank_info["first_rank"] = min(rank_info["first_rank"], chunk_rank)
                rank_info["best_score"] = max(rank_info["best_score"], chunk_score)

            if parent_id not in parent_docs_map:
                parent_doc = self.parent_documents_map.get(parent_id)
                if parent_doc is not None:
                    parent_docs_map[parent_id] = parent_doc
                    continue

                for candidate in self.documents:
                    if candidate.metadata.get("parent_id") == parent_id:
                        parent_docs_map[parent_id] = candidate
                        break

        sorted_parent_ids = sorted(
            parent_rank_info.keys(),
            key=lambda parent_id: (
                parent_rank_info[parent_id]["first_rank"],
                -parent_rank_info[parent_id]["best_score"],
                -parent_rank_info[parent_id]["hit_count"],
            ),
        )

        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        parent_info = []
        for parent_doc in parent_docs:
            doc_title = parent_doc.metadata.get("doc_title") or parent_doc.metadata.get("dish_name", "未知文档")
            parent_id = parent_doc.metadata.get("parent_id")
            rank_info = parent_rank_info.get(parent_id, {})
            relevance_count = rank_info.get("hit_count", 0)
            parent_info.append(f"{doc_title}({relevance_count}块)")

        logger.info(
            "从 %s 个文档块中找到 %s 个去重父文档: %s",
            len(retrieved_chunks),
            len(parent_docs),
            ", ".join(parent_info),
        )
        return parent_docs

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        Returns:
            统计信息字典
        """
        if not self.documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "categories": {},
                "departments": {},
                "file_types": {},
                "difficulties": {},
                "avg_chunk_size": 0,
            }

        categories: Dict[str, int] = {}
        departments: Dict[str, int] = {}
        file_types: Dict[str, int] = {}
        difficulties: Dict[str, int] = {}

        for parent_doc in self.documents:
            metadata = parent_doc.metadata or {}

            category = metadata.get("doc_category", "其他")
            categories[category] = categories.get(category, 0) + 1

            department = metadata.get("department") or "未注明"
            departments[department] = departments.get(department, 0) + 1

            file_type = metadata.get("file_type") or "unknown"
            file_types[file_type] = file_types.get(file_type, 0) + 1

            difficulty = metadata.get("difficulty", "未知")
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        avg_chunk_size = 0
        if self.chunks:
            avg_chunk_size = sum(chunk.metadata.get("chunk_size", 0) for chunk in self.chunks) / len(self.chunks)

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "departments": departments,
            "file_types": file_types,
            "difficulties": difficulties,
            "avg_chunk_size": avg_chunk_size,
        }

    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件
        Args:
            output_path: 输出文件路径
        """
        import json

        metadata_list = []
        for parent_doc in self.documents:
            metadata = parent_doc.metadata or {}
            metadata_list.append(
                {
                    "source": metadata.get("source"),
                    "relative_path": metadata.get("relative_path"),
                    "doc_id": metadata.get("doc_id"),
                    "doc_title": metadata.get("doc_title"),
                    "doc_category": metadata.get("doc_category"),
                    "department": metadata.get("department"),
                    "file_type": metadata.get("file_type"),
                    "section": metadata.get("section"),
                    "page": metadata.get("page"),
                    "category": metadata.get("category"),
                    "dish_name": metadata.get("dish_name"),
                    "difficulty": metadata.get("difficulty"),
                    "content_length": len(parent_doc.page_content),
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info("元数据已导出到: %s", output_path)
