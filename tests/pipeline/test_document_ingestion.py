from pathlib import Path

from campus_rag.config import PROJECT_ROOT
from langchain_core.documents import Document
from campus_rag.pipeline import document_ingestion
from campus_rag.pipeline.document_ingestion import infer_doc_category, load_documents, load_pdf_document


def test_load_documents_returns_supported_file_types():
    docs = load_documents(PROJECT_ROOT / "data" / "knowledge_base" / "campus")

    assert docs
    assert {doc.metadata["file_type"] for doc in docs} >= {"md", "txt", "pdf"}
    assert all(doc.metadata["doc_title"] for doc in docs)


def test_infer_doc_category_uses_path_only():
    assert infer_doc_category(Path("data/knowledge_base/campus/teaching/exams/sample.txt")) == "教务教学"
    assert infer_doc_category(Path("data/knowledge_base/campus/life/dormitory/sample.txt")) == "校园生活"
    assert infer_doc_category(Path("data/knowledge_base/campus/unknown/sample.txt")) == "其他"


def test_load_documents_can_read_utf16_text(tmp_path):
    campus_root = tmp_path / "data" / "campus"
    target_dir = campus_root / "teaching" / "exams"
    target_dir.mkdir(parents=True)

    source_path = target_dir / "utf16_notice.txt"
    source_path.write_text("关于期末考试安排", encoding="utf-16")

    docs = load_documents(campus_root)

    assert len(docs) == 1
    assert docs[0].page_content == "关于期末考试安排"
    assert docs[0].metadata["file_type"] == "txt"
    assert docs[0].metadata["doc_category"] == "教务教学"
    assert "section" not in docs[0].metadata


def test_load_pdf_document_uses_document_title_without_parent_section(monkeypatch, tmp_path):
    pdf_path = tmp_path / "data" / "campus" / "regulations" / "student_handbook.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    class FakePyPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [
                Document(page_content="学生手册\n第一章 总则", metadata={"page": 0}),
                Document(page_content="第二章 学籍管理\n学生应按规定注册。", metadata={"page": 1}),
            ]

    monkeypatch.setattr(document_ingestion, "PyPDFLoader", FakePyPDFLoader)

    docs = load_pdf_document(pdf_path, pdf_path.parents[2])

    assert [doc.metadata["doc_title"] for doc in docs] == ["学生手册", "学生手册"]
    assert [doc.metadata["page"] for doc in docs] == [1, 2]
    assert all("section" not in doc.metadata for doc in docs)

