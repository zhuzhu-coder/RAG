"""
索引构建模块
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from .chunking_config import CHUNKING_CONFIG
from .document_ingestion import SUPPORTED_SUFFIXES

# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)

# manifest 文件名
MANIFEST_FILENAME = "index_manifest.json"
# manifest 结构版本
MANIFEST_SCHEMA_VERSION = 1
# 索引类型
INDEX_TYPE = "Chroma"
# Chroma collection 名称
CHROMA_COLLECTION_NAME = "campus_rag_chunks"
# embedding 向量维度
EMBEDDING_DIMENSIONS = 1024
# embedding 请求超时时间，避免网络异常时长时间挂起
EMBEDDING_TIMEOUT_SECONDS = 30
# manifest 对比键
MANIFEST_COMPARE_KEYS = [
    "schema_version", # manifest 结构版本
    "index_type", # 索引类型
    "embedding_model", # 嵌入模型名称
    "embedding_dimensions", # 向量维度
    "chunking", # 分块配置
    "source_fingerprint", # 源文件指纹
    "source_document_count", # 源文件文档数量
]


class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./storage/chroma"):
        """
        初始化索引构建模块
        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
        """
        # 初始化模型名称
        self.model_name = model_name
        # 初始化索引保存路径
        self.index_save_path = index_save_path
        # 初始化向量维度
        self.embedding_dimensions = EMBEDDING_DIMENSIONS
        # 初始化嵌入模型
        self.embeddings = None
        # 初始化向量数据库
        self.vectorstore = None
        # 初始化嵌入模型客户端
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        # 创建 embeddings 客户端对象
        self.embeddings = OpenAIEmbeddings(
            model=self.model_name,
            api_key=api_key,
            # 接口地址
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            dimensions=self.embedding_dimensions, # 向量维度
            chunk_size=10, # 一次请求处理的文档块数量
            timeout=EMBEDDING_TIMEOUT_SECONDS, # 请求超时时间
            check_embedding_ctx_length=False, # 不检查向量上下文长度
        )
        logger.info("嵌入模型初始化完成")

    def build_manifest(self, data_path: str, chunks: Optional[List[Document]] = None) -> Dict[str, Any]:
        """
        根据当前源数据和索引配置构建 manifest，用于判断本地索引是否过期
        Args:
            data_path: 校园文档数据目录
            chunks: 可选的分块结果；重建索引后用于记录 chunk_count
        Returns:
            manifest 字典
        """
        # 收集源文件记录
        source_records = self._collect_source_records(data_path)
        return {
            # manifest 结构版本
            "schema_version": MANIFEST_SCHEMA_VERSION,
            # 索引类型
            "index_type": INDEX_TYPE,
            # 嵌入模型名称
            "embedding_model": self.model_name,
            # 向量维度
            "embedding_dimensions": self.embedding_dimensions,
            # 分块配置
            "chunking": CHUNKING_CONFIG,
            # 源文件指纹
            "source_fingerprint": self._fingerprint_source_records(source_records),
            # 源文件文档数量
            "source_document_count": len(source_records),
            # 分块数量
            "chunk_count": len(chunks) if chunks is not None else None,
            # 创建时间
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        """
        读取本地索引 manifest
        Returns:
            manifest 字典；不存在或解析失败时返回 None
        """
        manifest_path = Path(self.index_save_path) / MANIFEST_FILENAME
        if not manifest_path.exists():
            logger.info(f"索引 manifest 不存在: {manifest_path}")
            return None

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f) # 从文件加载 JSON 字符串，解析为 Python 字典
        except Exception as e:
            logger.warning(f"读取索引 manifest 失败: {e}")
            return None

    def save_manifest(self, manifest: Dict[str, Any]):
        """
        保存索引 manifest 到索引目录
        Args:
            manifest: 要保存的 manifest 字典
        """
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        manifest_path = Path(self.index_save_path) / MANIFEST_FILENAME
        with open(manifest_path, "w", encoding="utf-8") as f:
            # 保存 manifest 到文件，保留中文，缩进 2 个空格
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"索引 manifest 已保存到: {manifest_path}")

    def manifest_matches(self, expected_manifest: Dict[str, Any]) -> bool:
        """
        判断本地 manifest 是否与当前源数据和索引配置匹配
        Args:
            expected_manifest: 当前运行时计算出的 manifest
        Returns:
            是否匹配
        """
        # 读取本地 manifest
        actual_manifest = self.load_manifest()
        if not actual_manifest:
            return False
        # 比较 manifest 中的每个字段是否匹配
        for key in MANIFEST_COMPARE_KEYS:
            # 如果本地 manifest 中没有该字段，或与预期 manifest 中的字段值不匹配
            if actual_manifest.get(key) != expected_manifest.get(key):
                logger.info(
                    "索引 manifest 不匹配: %s, actual=%s, expected=%s",
                    key,
                    actual_manifest.get(key),
                    expected_manifest.get(key),
                )
                return False

        return True

    def _collect_source_records(self, data_path: str) -> List[Dict[str, str]]:
        """
        收集源文档文件的稳定路径和内容哈希
        Args:
            data_path: 校园文档数据目录
        Returns:
            每个文件的相对路径和内容哈希的列表
        """
        data_root = Path(data_path).resolve()
        records = []
        # 递归查找 data_root 下的所有支持文件，再按照文件的绝对路径字符串（统一 Unix 格式）进行字典序排序
        for source_file in sorted(data_root.rglob("*"), key=lambda path: path.resolve().as_posix()):
            if not source_file.is_file():
                continue
            if source_file.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            file_path = source_file.resolve()
            # 尝试获取相对路径，如果失败则使用绝对路径
            try:
                relative_path = file_path.relative_to(data_root).as_posix()
            except ValueError:
                relative_path = file_path.as_posix()
            # 读取文件的全部二进制内容，计算并返回其 SHA256 哈希值（64 位唯一指纹字符串）
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            records.append({
                "relative_path": relative_path,
                "content_sha256": content_hash,
            })
        return records

    @staticmethod
    def _fingerprint_source_records(source_records: List[Dict[str, str]]) -> str:
        """
        根据源文件记录生成稳定 sha256 指纹
        Args:
            source_records: 每个文件的相对路径和内容哈希的列表
        Returns:
            sha256 指纹字符串
        """
        # 将源文件记录（Python 字典列表）转换为 JSON 字符串
        payload = json.dumps(
            source_records,
            ensure_ascii=False, # 中文不转义
            sort_keys=True, # 按键排序，避免键顺序不同导致hash不稳定
            separators=(",", ":"), # 去掉 JSON 字符串中多余空格，使得更紧凑
        ).encode("utf-8") # 编码为 UTF-8 字节序列
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def load_index(self, expected_manifest: Optional[Dict[str, Any]] = None):
        """
        从配置的路径加载向量索引
        Args:
            expected_manifest: 当前源数据和索引配置对应的 manifest；不匹配时不加载索引
        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None
        
        if expected_manifest is not None and not self.manifest_matches(expected_manifest):
            logger.info("索引 manifest 缺失或已过期，将构建新索引")
            return None

        try:
            # 从本地目录加载 Chroma collection
            self.vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=self.index_save_path,
            )
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None
    
    def build_vector_index(self, chunks: List[Document]) -> Chroma:
        """
        构建向量索引
        Args:
            chunks: 文档块列表
        Returns:
            Chroma向量存储对象
        """
        logger.info("正在构建Chroma向量索引...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        # 构建 Chroma 向量存储；重建时清空 collection，避免旧向量混入新索引
        self.vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.index_save_path,
        )
        self.vectorstore.reset_collection()
        self.vectorstore.add_documents(chunks, ids=self._chunk_ids(chunks))
        
        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self.vectorstore
    
    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引添加新文档块
        Args:
            new_chunks: 新的文档块列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档块到索引...")
        self.vectorstore.add_documents(new_chunks, ids=self._chunk_ids(new_chunks))
        logger.info("新文档块添加完成")

    def save_index(self):
        """
        保存向量索引到配置的路径
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        # Chroma 使用 persist_directory 自动持久化，这里保留显式入口便于编排层调用
        logger.info(f"向量索引已保存到: {self.index_save_path}")

    @staticmethod
    def _chunk_ids(chunks: List[Document]) -> List[str]:
        """提取稳定的 Chroma 文档 ID。"""
        ids = []
        for position, chunk in enumerate(chunks):
            metadata = chunk.metadata or {}
            ids.append(str(metadata.get("chunk_id") or f"chunk-{position}"))
        return ids
