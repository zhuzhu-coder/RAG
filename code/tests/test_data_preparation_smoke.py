from pathlib import Path

from config import PROJECT_ROOT, RAGConfig
from langchain_core.documents import Document
from rag_modules.data_preparation import DataPreparationModule


def test_default_sample_data_can_be_loaded_and_chunked():
    config = RAGConfig()
    sample_file = PROJECT_ROOT / "data" / "cook" / "vegetable_dish" / "番茄炒蛋.md"

    assert Path(config.data_path).exists()
    assert sample_file.exists()

    data_module = DataPreparationModule(config.data_path)
    documents = data_module.load_documents()
    chunks = data_module.chunk_documents()
    stats = data_module.get_statistics()

    assert documents
    assert chunks
    assert stats["total_documents"] >= 1
    assert "素菜" in stats["categories"]


def test_chunk_ids_are_stable_across_loads():
    config = RAGConfig()

    first_module = DataPreparationModule(config.data_path)
    first_module.load_documents()
    first_chunks = first_module.chunk_documents()

    second_module = DataPreparationModule(config.data_path)
    second_module.load_documents()
    second_chunks = second_module.chunk_documents()

    assert [chunk.metadata["chunk_id"] for chunk in first_chunks] == [
        chunk.metadata["chunk_id"] for chunk in second_chunks
    ]


def test_get_parent_documents_prefers_stronger_first_match_over_chunk_count():
    data_module = DataPreparationModule.__new__(DataPreparationModule)
    data_module.documents = [
        Document(
            page_content="# 番茄炒蛋\n完整菜谱",
            metadata={"parent_id": "parent-a", "dish_name": "番茄炒蛋"},
        ),
        Document(
            page_content="# 清炒油麦菜\n完整菜谱",
            metadata={"parent_id": "parent-b", "dish_name": "清炒油麦菜"},
        ),
    ]

    retrieved_chunks = [
        Document(
            page_content="清炒油麦菜 食材",
            metadata={"parent_id": "parent-b", "rrf_score": 0.05},
        ),
        Document(
            page_content="番茄炒蛋 食材",
            metadata={"parent_id": "parent-a", "rrf_score": 0.04},
        ),
        Document(
            page_content="番茄炒蛋 步骤",
            metadata={"parent_id": "parent-a", "rrf_score": 0.03},
        ),
    ]

    parent_docs = data_module.get_parent_documents(retrieved_chunks)

    assert [doc.metadata["dish_name"] for doc in parent_docs] == [
        "清炒油麦菜",
        "番茄炒蛋",
    ]
