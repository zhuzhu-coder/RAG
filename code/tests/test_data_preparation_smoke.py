from config import PROJECT_ROOT
from rag_modules.data_preparation import DataPreparationModule


def test_campus_documents_can_be_loaded_and_chunked():
    module = DataPreparationModule(str(PROJECT_ROOT / "data" / "campus"))

    documents = module.load_documents()
    chunks = module.chunk_documents()
    stats = module.get_statistics()

    assert documents
    assert chunks
    assert stats["total_documents"] >= 1
    assert "规章制度" in stats["categories"]
    assert set(stats["file_types"]) >= {"md", "txt", "pdf"}
    assert all("doc_title" in doc.metadata for doc in documents)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
    assert all("doc_title" in chunk.metadata for chunk in chunks)
