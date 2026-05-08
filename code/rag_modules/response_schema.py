"""
响应模式定义
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True) # 数据类，不可变
class RetrievedSource:
    """与答案引用编号关联的召回文档块"""

    source_id: int # 答案引用编号，同一菜谱的多个文档块可以共享编号
    dish_name: str # 菜品名称
    category: str # 分类
    difficulty: str # 难度
    section: str # 章节
    source: str # 来源文件路径
    chunk_index: int # 文档块索引
    rrf_score: Optional[float] # RRF分数
    snippet: str # 文档块摘要

    def to_dict(self) -> Dict[str, Any]:
        """将 RetrievedSource 对象转换为字典"""
        return asdict(self)


@dataclass(frozen=True)
class RAGResponse:
    """完整RAG响应"""

    question: str # 原始问题
    route_type: str # 路由类型
    rewritten_query: str # 重写后的问题
    answer: str # 最终答案
    sources: List[RetrievedSource] # 与答案引用编号对齐的召回文档块列表

    def to_dict(self) -> Dict[str, Any]:
        """将 RAGResponse 对象转换为字典"""
        return {
            "question": self.question,
            "route_type": self.route_type,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            # 将对齐后的召回文档块列表转换为字典列表
            "sources": [source.to_dict() for source in self.sources],
        }
