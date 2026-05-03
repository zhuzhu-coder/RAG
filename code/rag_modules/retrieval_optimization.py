"""
检索优化模块
"""

import logging
import hashlib
import warnings
from functools import lru_cache
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def tokenize_chinese_text(text: str) -> List[str]:
    """Tokenize Chinese text for BM25, falling back to character tokens."""
    cut_for_search = _get_jieba_cut_for_search()
    tokens = cut_for_search(text) if cut_for_search else text

    return [token.strip() for token in tokens if token.strip()]


@lru_cache(maxsize=1)
def _get_jieba_cut_for_search():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jieba

        jieba.setLogLevel(logging.WARNING)
        return jieba.cut_for_search
    except Exception as exc:
        logger.warning("jieba 分词不可用，回退到字符级 BM25 分词: %s", exc)
        return None


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document], candidate_k: int = 10):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
            candidate_k: 每个检索器返回的候选文档数量
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.candidate_k = candidate_k
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.candidate_k}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.candidate_k,
            preprocess_func=tokenize_chinese_text,
        )



        logger.info("检索器设置完成")
    
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]
    
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索
        
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
            
        Returns:
            过滤后的文档列表
        """
        candidate_k = max(self.candidate_k, top_k * 3)
        faiss_filters = self._to_faiss_filters(filters)

        try:
            vector_docs = self.vectorstore.similarity_search(
                query,
                k=candidate_k,
                filter=faiss_filters,
            )
        except TypeError:
            vector_docs = [
                doc for doc in self.vector_retriever.invoke(query)
                if self._matches_filters(doc, filters)
            ]

        bm25_docs = [
            doc for doc in self.bm25_retriever.invoke(query)
            if self._matches_filters(doc, filters)
        ]

        return self._rrf_rerank(vector_docs, bm25_docs)[:top_k]

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            doc_id = self._document_key(doc)
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = self._document_key(doc)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs

    @staticmethod
    def _document_key(doc: Document) -> str:
        """Return a stable identifier for reranking and deduplication."""
        metadata = doc.metadata or {}

        parent_id = metadata.get("parent_id")
        chunk_index = metadata.get("chunk_index")
        if parent_id is not None and chunk_index is not None:
            return f"parent:{parent_id}:{chunk_index}"

        source = metadata.get("source")
        if source is not None and chunk_index is not None:
            return f"source:{source}:{chunk_index}"

        if metadata.get("chunk_id"):
            return f"chunk:{metadata['chunk_id']}"

        digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        return f"content:{digest}"

    @staticmethod
    def _matches_filters(doc: Document, filters: Dict[str, Any]) -> bool:
        metadata = doc.metadata or {}
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    @staticmethod
    def _to_faiss_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        faiss_filters = {}
        for key, value in filters.items():
            faiss_filters[key] = {"$in": value} if isinstance(value, list) else value
        return faiss_filters
