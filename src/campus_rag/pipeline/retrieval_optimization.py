"""
检索优化模块
"""

import logging
import hashlib
import warnings
from functools import lru_cache
from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)


def tokenize_chinese_text(text: str) -> List[str]:
    """
    对中文文本进行分词，使用jieba分词器
    Args:
        text: 中文文本
    Returns:
        分词后的token列表
    """
    cut_for_search = _get_jieba_cut_for_search()
    tokens = cut_for_search(text) if cut_for_search else text

    return [token.strip() for token in tokens if token.strip()]


@lru_cache(maxsize=1) # 缓存jieba分词器，避免重复加载
def _get_jieba_cut_for_search():
    """
    获取jieba分词器，缓存结果
    Returns:
        jieba 分词器
    """
    try:
        # 导入 jieba 时临时忽略 warning
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            import jieba
        # 设置 jieba 日志级别为 WARNING
        jieba.setLogLevel(logging.WARNING)
        return jieba.cut_for_search
    except Exception as exc:
        logger.warning("jieba 分词不可用，回退到字符级 BM25 分词: %s", exc)
        return None


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: Any, chunks: List[Document], candidate_k: int = 10, rrf_k: int = 60):
        """
        初始化检索优化模块
        Args:
            vectorstore: LangChain 向量存储
            chunks: 文档块列表
            candidate_k: 每个检索器返回的候选文档块数量
            rrf_k: RRF 平滑参数
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._setup_retrievers()

    def _setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器（语义相似度检索）
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", # 使用相似度检索
            search_kwargs={"k": self.candidate_k}
        )

        # BM25检索器（关键词检索）
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.candidate_k,
            preprocess_func=tokenize_chinese_text # 预处理函数：中文分词函数
        )
        logger.info("检索器设置完成")
    
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排
        Args:
            query: 查询文本
            top_k: 返回结果数量
        Returns:
            检索到的文档块列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_chunks = self.vector_retriever.invoke(query)
        bm25_chunks = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_chunks = self._rrf_rerank(vector_chunks, bm25_chunks)
        return reranked_chunks[:top_k]

    def vector_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        向量检索 - 用于评估或调试单一路径的语义检索效果
        Args:
            query: 查询文本
            top_k: 返回结果数量
        Returns:
            检索到的文档块列表
        """
        return self.vector_retriever.invoke(query)[:top_k]

    def bm25_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        BM25检索 - 用于评估或调试单一路径的关键词检索效果
        Args:
            query: 查询文本
            top_k: 返回结果数量
        Returns:
            检索到的文档块列表
        """
        return self.bm25_retriever.invoke(query)[:top_k]
    
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
        Returns:
            过滤后的文档块列表
        """
        candidate_k = max(self.candidate_k, top_k * 3) # 避免过滤后结果不够
        metadata_filters = self._to_metadata_filters(filters) # 向量库可识别的过滤条件
        filtered_chunks = [
            chunk for chunk in self.chunks
            if self._matches_filters(chunk, filters)
        ]

        try:
            vector_chunks = self.vectorstore.similarity_search(
                query,
                k=candidate_k,
                filter=metadata_filters, # 应用元数据过滤条件
            )
        except TypeError:
            # 如果向量检索器不支持元数据过滤
            vector_candidates = self.vectorstore.similarity_search(query, k=candidate_k)
            vector_chunks = [
                # 先使用扩大的候选集检索
                chunk for chunk in vector_candidates
                # 再根据元数据过滤
                if self._matches_filters(chunk, filters)
            ]
        # 过滤后的BM25检索结果
        if filtered_chunks:
            filtered_bm25_retriever = BM25Retriever.from_documents(
                filtered_chunks,
                k=candidate_k,
                preprocess_func=tokenize_chinese_text,
            )
            bm25_chunks = filtered_bm25_retriever.invoke(query)
        else:
            bm25_chunks = []
        # 使用RRF重排
        return self._rrf_rerank(vector_chunks, bm25_chunks)[:top_k]

    def _rrf_rerank(self, vector_chunks: List[Document], bm25_chunks: List[Document], k: int | None = None) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档块
        Args:
            vector_chunks: 向量检索返回的文档块
            bm25_chunks: BM25检索返回的文档块
            k: RRF参数，用于平滑排名
        Returns:
            重排后的文档块列表
        """
        rrf_k = int(k if k is not None else getattr(self, "rrf_k", 60))
        chunk_scores = {} # 文档块RRF分数映射
        chunk_objects = {} # 文档块对象映射

        # 计算向量检索结果的RRF分数
        for rank, chunk in enumerate(vector_chunks):
            # 为每个文档块生成唯一的标识符
            chunk_key = self._chunk_key(chunk)
            # 存储文档块对象
            chunk_objects[chunk_key] = chunk

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (rrf_k + rank + 1)
            # 累加RRF分数
            chunk_scores[chunk_key] = chunk_scores.get(chunk_key, 0) + rrf_score

            logger.debug(f"向量检索 - 文档块{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, chunk in enumerate(bm25_chunks):
            chunk_key = self._chunk_key(chunk)
            chunk_objects[chunk_key] = chunk

            rrf_score = 1.0 / (rrf_k + rank + 1)
            chunk_scores[chunk_key] = chunk_scores.get(chunk_key, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档块{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_chunks = []
        for chunk_key, final_score in sorted_chunks:
            if chunk_key in chunk_objects:
                chunk = chunk_objects[chunk_key]
                # 复制文档块，避免把临时RRF分数写回原始Document
                scored_chunk = chunk.model_copy(
                    deep=True, # 深拷贝
                    update={"metadata": {**(chunk.metadata or {}), "rrf_score": final_score}},
                )
                reranked_chunks.append(scored_chunk)
                logger.debug(f"最终排序 - 文档块: {chunk.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_chunks)}个文档块, BM25检索{len(bm25_chunks)}个文档块, 合并后{len(reranked_chunks)}个文档块")

        return reranked_chunks

    @staticmethod
    def _chunk_key(chunk: Document) -> str:
        """
        为文档块生成稳定的唯一标识符
        Args:
            chunk: 文档块对象
        Returns:
            稳定的唯一标识符
        """
        metadata = chunk.metadata or {}
        # 1.用parent_id和chunk_index作为key
        parent_id = metadata.get("parent_id")
        chunk_index = metadata.get("chunk_index")
        if parent_id is not None and chunk_index is not None:
            return f"parent:{parent_id}:{chunk_index}"
        # 2.用source和chunk_index作为key
        source = metadata.get("source")
        if source is not None and chunk_index is not None:
            return f"source:{source}:{chunk_index}"
        # 3.用chunk_id作为key
        if metadata.get("chunk_id"):
            return f"chunk:{metadata['chunk_id']}"
        # 4.用正文的MD5摘要作为兜底key
        digest = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
        return f"content:{digest}"

    @staticmethod
    def _matches_filters(chunk: Document, filters: Dict[str, Any]) -> bool:
        """
        检查文档块是否符合元数据过滤条件
        Args:
            chunk: 文档块对象
            filters: 元数据过滤条件字典
        Returns:
            是否符合过滤条件
        """
        metadata = chunk.metadata or {}
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
    def _to_metadata_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        将元数据过滤条件转换为 LangChain 向量库可识别的格式
        Args:
            filters: 元数据过滤条件字典
        Returns:
            向量库可识别的过滤条件
        """
        metadata_filters = {}
        for key, value in filters.items():
            # 如果值是列表，转换为 LangChain/Chroma 支持的 $in 操作符，否则直接赋值
            metadata_filters[key] = {"$in": value} if isinstance(value, list) else value
        return metadata_filters
