"""
RAG系统配置文件
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

# 取当前文件的绝对路径的父目录
CODE_DIR = Path(__file__).resolve().parent
# 再取CODE_DIR的父目录，即项目根目录
PROJECT_ROOT = CODE_DIR.parent


def resolve_project_path(path_value: str) -> str:
    """将路径转换为绝对路径"""
    path = Path(path_value).expanduser() # 将路径中的用户~替换为用户主目录
    if not path.is_absolute(): # 如果路径不是绝对路径
        path = PROJECT_ROOT / path # 则将路径拼接到项目根目录
    return str(path.resolve())


@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "cook"))# 默认数据目录
    index_save_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "vector_index"))# 默认向量索引目录

    # 模型配置
    embedding_model: str = "text-embedding-v4"
    llm_model: str = "qwen3.5-plus"

    # 检索配置
    top_k: int = 3
    retrieval_candidate_k: int = 10 # 检索候选数量

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 初始化后时自动调用
    def __post_init__(self):
        """初始化后的处理"""
        # 路径统一，将所有路径转换为绝对路径
        self.data_path = resolve_project_path(self.data_path)
        self.index_save_path = resolve_project_path(self.index_save_path)
        # 类型转换，将从环境变量中获取的字符串转换为整数或浮点数
        self.top_k = int(self.top_k)
        self.retrieval_candidate_k = int(self.retrieval_candidate_k)
        self.temperature = float(self.temperature)
        self.max_tokens = int(self.max_tokens)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig': # 当前类， 字典
        """从字典创建配置对象"""
        return cls(**config_dict) # 字典解包

    @classmethod
    def from_env(cls) -> 'RAGConfig': # 当前类  
        """从环境变量创建配置对象"""
        env_mapping = {
            "data_path": "RAG_DATA_PATH",
            "index_save_path": "RAG_INDEX_SAVE_PATH",
            "embedding_model": "RAG_EMBEDDING_MODEL",
            "llm_model": "RAG_LLM_MODEL",
            "top_k": "RAG_TOP_K",
            "retrieval_candidate_k": "RAG_RETRIEVAL_CANDIDATE_K",
            "temperature": "RAG_TEMPERATURE",
            "max_tokens": "RAG_MAX_TOKENS",
        }
        # 字典推导式，将环境变量中的值赋值给配置类的属性
        config_dict = {
            field_name: os.environ[env_name]
            for field_name, env_name in env_mapping.items()
            if env_name in os.environ
        }
        return cls(**config_dict) # 字典解包
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'retrieval_candidate_k': self.retrieval_candidate_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

# 默认配置实例
DEFAULT_CONFIG = RAGConfig.from_env()
