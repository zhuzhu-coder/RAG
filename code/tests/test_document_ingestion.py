from config import PROJECT_ROOT
from rag_modules.document_ingestion import load_documents


def test_load_documents_returns_supported_file_types():
    docs = load_documents(PROJECT_ROOT / "data" / "campus")

    assert docs
    assert {doc.metadata["file_type"] for doc in docs} >= {"md", "txt", "pdf"}
    assert all(doc.metadata["doc_title"] for doc in docs)
