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
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values())) # 分类标签
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难'] # 难度标签
    
    def __init__(self, data_path: str):
        """
        初始化数据准备模块
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整文档）
        self.chunks: List[Document] = []     # 子文档（按 Markdown 标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射
    
    def load_documents(self) -> List[Document]:
        """
        加载文档数据
        Returns:
            加载的文档列表
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
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent"  # 标记为父文档
                    }
                )
                # 父文档列表
                documents.append(doc)
            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")
        
        # 增强文档元数据
        for doc in documents:
            self._enhance_metadata(doc)
        
        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据
        Args:
            doc: 需要增强元数据的文档
        """
        file_path = Path(doc.metadata.get('source', ''))
        path_parts = file_path.parts # 将路径拆开，得到每个目录或文件名的元组
        
        # 提取菜品分类
        doc.metadata['category'] = '其他'
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break
        
        # 提取菜品名称
        doc.metadata['dish_name'] = file_path.stem # 获取文件名（不包含扩展名）作为菜品名称

        # 分析难度等级
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

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
            分块后的文档列表
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
            chunk.metadata['chunk_size'] = len(chunk.page_content) # 子块的字符数

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割
        Returns:
            按标题结构分割的文档列表
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

        for doc in self.documents:
            try:
                # 检查文档内容是否包含Markdown标题
                content_preview = doc.page_content[:200]
                has_headers: bool = any(line.strip().startswith('#') for line in content_preview.split('\n'))

                if not has_headers:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 内容中没有发现Markdown标题")
                    logger.debug(f"内容预览: {content_preview}")

                # 对每个文档进行Markdown分割
                md_chunks: List[Document] = markdown_splitter.split_text(doc.page_content)

                logger.debug(f"文档 {doc.metadata.get('dish_name', '未知')} 分割成 {len(md_chunks)} 个chunk")

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) <= 1:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 未能按标题分割，可能缺少标题结构")

                # 为每个子块建立与父文档的关系
                parent_id = doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):
                    # 为子块分配确定性ID，保证索引加载后能与本次BM25分块对齐
                    child_id = self._make_chunk_id(parent_id, i)

                    # 合并原文档元数据和新的标题元数据
                    chunk.metadata.update(doc.metadata) # 复制父文档的元数据到子块
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "doc_type": "child",  # 覆盖标记为子文档
                        "chunk_index": i      # 在父文档中的位置
                    })

                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)
            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                fallback_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata, # 复制父文档的元数据到子块
                        "chunk_id": self._make_chunk_id(doc.metadata["parent_id"], 0),
                        "doc_type": "child",
                        "chunk_index": 0,
                    },
                )
                self.parent_child_map[fallback_doc.metadata["chunk_id"]] = doc.metadata["parent_id"]
                all_chunks.append(fallback_doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    @staticmethod
    def _make_chunk_id(parent_id: str, chunk_index: int) -> str:
        """生成稳定的子块ID，便于向量索引和BM25结果去重"""
        raw_id = f"{parent_id}:{chunk_index}"
        return hashlib.md5(raw_id.encode("utf-8")).hexdigest()

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        按分类过滤文档
        Args:
            category: 菜品分类
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('category') == category]
    
    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        按难度过滤文档
        Args:
            difficulty: 难度等级
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('difficulty') == difficulty]
    
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

        for doc in self.documents:
            # 统计分类
            category = doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1

            # 统计难度
            difficulty = doc.metadata.get('difficulty', '未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents), # 父文档总数
            'total_chunks': len(self.chunks), # 子块数量
            'categories': categories, # 分类统计
            'difficulties': difficulties, # 难度统计
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0 # 平均子块大小
        }
    
    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件
        Args:
            output_path: 输出文件路径
        """
        import json
        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'), # 文件路径
                'dish_name': doc.metadata.get('dish_name'), # 菜品名
                'category': doc.metadata.get('category'), # 分类
                'difficulty': doc.metadata.get('difficulty'), # 难度
                'content_length': len(doc.page_content) # 正文长度
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 把 python 对象序列化为 JSON 字符串并写入文件，确保中文字符不被转义，缩进为 2 个空格
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重）
        Args:
            child_chunks: 检索到的子块列表
        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {} # 父文档ID -> 相关性计数
        parent_docs_map = {} # 父文档ID -> 父文档对象

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                 key=lambda x: parent_relevance[x],
                                 reverse=True)

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs
