"""
数据准备模块
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)

class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""
    # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGORY_MAPPING = { # 分类映射
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS = list(CATEGORY_MAPPING.values()) # 分类标签
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难'] # 难度标签
    
    def __init__(self, data_path: str):
        """
        初始化数据准备模块
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整菜谱文档）
        self.chunks: List[Document] = []     # 文档块（按 Markdown 标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 文档块ID -> 父文档ID的映射
    
    def load_documents(self) -> List[Document]:
        """
        加载完整父文档数据
        Returns:
            加载的完整父文档列表
        """
        logger.info(f"正在从 {self.data_path} 加载文档...")
        
        # 直接读取Markdown文件以保持原始格式
        documents = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"): # 递归查找所有Markdown文件
            try:
                # 直接读取文件内容，保持Markdown格式
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 为每个父文档分配确定性的唯一ID（基于数据根目录的相对路径（更稳定））
                try:
                    data_root = Path(self.data_path).resolve() # 数据根目录的绝对路径
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix() # 将文档的绝对路径转换为相对于数据根目录的相对路径，并把路径统一成 / 风格
                except Exception:
                    relative_path = Path(md_file).as_posix() # 如果转换失败，直接使用文档路径本身，再把路径统一成 / 风格
                # 将相对路径转成 MD5 哈希字符串 作为父文档ID
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                # 创建Document对象
                parent_doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent"  # 标记为父文档
                    }
                )
                # 父文档列表
                documents.append(parent_doc)
            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")
        
        # 增强文档元数据
        for parent_doc in documents:
            self._enhance_metadata(parent_doc)
        
        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个父文档")
        return documents
    
    def _enhance_metadata(self, parent_doc: Document):
        """
        增强父文档元数据
        Args:
            parent_doc: 需要增强元数据的父文档
        """
        file_path = Path(parent_doc.metadata.get('source', ''))
        path_parts = file_path.parts # 将路径拆开，得到每个目录或文件名的元组
        
        # 提取菜品分类
        parent_doc.metadata['category'] = '其他'
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                parent_doc.metadata['category'] = value
                break
        
        # 提取菜品名称
        parent_doc.metadata['dish_name'] = file_path.stem # 获取文件名（不包含扩展名）作为菜品名称

        # 分析难度等级
        content = parent_doc.page_content
        if '★★★★★' in content:
            parent_doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            parent_doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            parent_doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            parent_doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            parent_doc.metadata['difficulty'] = '非常简单'
        else:
            parent_doc.metadata['difficulty'] = '未知'

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """对外提供支持的难度标签列表"""
        return cls.DIFFICULTY_LABELS
    
    def chunk_documents(self) -> List[Document]:
        """
        Markdown结构感知分块
        Returns:
            分块后的文档块列表
        """
        logger.info("正在进行Markdown结构感知分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        # 使用Markdown标题分割器
        chunks = self._markdown_header_split()

        # 为每个chunk再补充元数据
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                chunk.metadata['chunk_id'] = self._make_chunk_id(
                    chunk.metadata.get("parent_id", "unknown"),
                    chunk.metadata.get("chunk_index", i),
                )
            chunk.metadata['batch_index'] = i  # 在所有chunk中的索引
            chunk.metadata['chunk_size'] = len(chunk.page_content) # 文档块的字符数

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割
        Returns:
            按标题结构分割的文档块列表
        """
        # 定义要分割的标题层级
        headers_to_split_on = [  # 二元组，标题符号和在元数据中的显示名称
            ("#", "主标题"),      # 菜品名称
            ("##", "二级标题"),   # 必备原料、计算、操作等
            ("###", "三级标题")   # 简易版本、复杂版本等
        ]

        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, # 定义要分割的标题层级
            strip_headers=False  # 保留标题文本，便于理解上下文
        )

        all_chunks = []

        for parent_doc in self.documents:
            try:
                # 检查文档内容是否包含Markdown标题
                content_preview = parent_doc.page_content[:200]
                has_headers: bool = any(line.strip().startswith('#') for line in content_preview.split('\n'))

                if not has_headers:
                    logger.warning(f"父文档 {parent_doc.metadata.get('dish_name', '未知')} 内容中没有发现Markdown标题")
                    logger.debug(f"内容预览: {content_preview}")

                # 对每个父文档进行Markdown分割
                md_chunks: List[Document] = markdown_splitter.split_text(parent_doc.page_content)

                logger.debug(f"父文档 {parent_doc.metadata.get('dish_name', '未知')} 分割成 {len(md_chunks)} 个文档块")

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) <= 1:
                    logger.warning(f"父文档 {parent_doc.metadata.get('dish_name', '未知')} 未能按标题分割，可能缺少标题结构")

                # 为每个文档块建立与父文档的关系
                parent_id = parent_doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):
                    # 为文档块分配确定性ID，保证索引加载后能与本次BM25分块对齐
                    child_id = self._make_chunk_id(parent_id, i)

                    # 合并父文档元数据和新的标题元数据
                    chunk.metadata.update(parent_doc.metadata) # 复制父文档的元数据到文档块
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "doc_type": "child",  # 覆盖标记为文档块
                        "chunk_index": i      # 在父文档中的位置
                    })

                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)
            except Exception as e:
                logger.warning(f"父文档 {parent_doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                fallback_doc = Document(
                    page_content=parent_doc.page_content,
                    metadata={
                        **parent_doc.metadata, # 复制父文档的元数据到文档块
                        "chunk_id": self._make_chunk_id(parent_doc.metadata["parent_id"], 0),
                        "doc_type": "child",
                        "chunk_index": 0,
                    },
                )
                self.parent_child_map[fallback_doc.metadata["chunk_id"]] = parent_doc.metadata["parent_id"]
                all_chunks.append(fallback_doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

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
        # 统计每个父文档的首次命中位置、最高 RRF 分数和命中次数
        parent_rank_info = {}  # 父文档ID -> 排序信息
        parent_docs_map = {}  # 父文档ID -> 父文档对象

        # 收集所有相关的父文档ID和排序信息
        for chunk_rank, chunk in enumerate(retrieved_chunks):
            chunk_metadata = chunk.metadata or {}
            parent_id = chunk_metadata.get("parent_id")
            if parent_id:
                raw_score = chunk_metadata.get("rrf_score")
                try:
                    chunk_score = float(raw_score) if raw_score is not None else 0.0
                except (TypeError, ValueError):
                    chunk_score = 0.0

                if parent_id not in parent_rank_info:
                    parent_rank_info[parent_id] = {
                        "first_rank": chunk_rank, # 首次命中位置
                        "best_score": chunk_score,# 最高分数
                        "hit_count": 1, # 命中次数
                    }
                else:
                    rank_info = parent_rank_info[parent_id]
                    rank_info["hit_count"] += 1
                    rank_info["first_rank"] = min(rank_info["first_rank"], chunk_rank)
                    rank_info["best_score"] = max(rank_info["best_score"], chunk_score)

                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for parent_doc in self.documents:
                        if parent_doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = parent_doc
                            break

        # 按最强证据排序：先看首次命中位置，再看最高分数，最后看命中次数
        sorted_parent_ids = sorted(
            parent_rank_info.keys(),
            key=lambda parent_id: (
                parent_rank_info[parent_id]["first_rank"],
                -parent_rank_info[parent_id]["best_score"],
                -parent_rank_info[parent_id]["hit_count"],
            ),
        )

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for parent_doc in parent_docs:
            dish_name = parent_doc.metadata.get('dish_name', '未知菜品')
            parent_id = parent_doc.metadata.get('parent_id')
            rank_info = parent_rank_info.get(parent_id, {})
            relevance_count = rank_info.get("hit_count", 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(retrieved_chunks)} 个文档块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}
        categories = {}
        difficulties = {}

        for parent_doc in self.documents:
            # 统计分类
            category = parent_doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1

            # 统计难度
            difficulty = parent_doc.metadata.get('difficulty', '未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents), # 父文档总数
            'total_chunks': len(self.chunks), # 文档块数量
            'categories': categories, # 分类统计
            'difficulties': difficulties, # 难度统计
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0 # 平均文档块大小
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
            metadata_list.append({
                'source': parent_doc.metadata.get('source'), # 文件路径
                'dish_name': parent_doc.metadata.get('dish_name'), # 菜品名
                'category': parent_doc.metadata.get('category'), # 分类
                'difficulty': parent_doc.metadata.get('difficulty'), # 难度
                'content_length': len(parent_doc.page_content) # 正文长度
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 把 python 对象序列化为 JSON 字符串并写入文件，确保中文字符不被转义，缩进为 2 个空格
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")
