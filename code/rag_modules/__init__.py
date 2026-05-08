from .data_preparation import DataPreparationModule
from .index_construction import IndexConstructionModule
from .retrieval_optimization import RetrievalOptimizationModule
from .generation_integration import GenerationIntegrationModule
from .response_schema import RAGResponse, RetrievedSource

# 对外暴露的模块
__all__ = [
    'DataPreparationModule',
    'IndexConstructionModule',
    'RetrievalOptimizationModule',
    'GenerationIntegrationModule',
    'RAGResponse',
    'RetrievedSource',
]

# 包版本号
__version__ = "1.0.0"
