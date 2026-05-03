"""
RAG系统配置文件
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent


def resolve_project_path(path_value: str) -> str:
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "cook"))
    index_save_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "vector_index"))

    # 模型配置
    embedding_model: str = "text-embedding-v4"
    llm_model: str = "qwen-plus"

    # 检索配置
    top_k: int = 3
    retrieval_candidate_k: int = 10

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        """初始化后的处理"""
        self.data_path = resolve_project_path(self.data_path)
        self.index_save_path = resolve_project_path(self.index_save_path)
        self.top_k = int(self.top_k)
        self.retrieval_candidate_k = int(self.retrieval_candidate_k)
        self.temperature = float(self.temperature)
        self.max_tokens = int(self.max_tokens)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """从 RAG_* 环境变量创建配置对象"""
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
        config_dict = {
            field_name: os.environ[env_name]
            for field_name, env_name in env_mapping.items()
            if env_name in os.environ
        }
        return cls(**config_dict)
    
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
