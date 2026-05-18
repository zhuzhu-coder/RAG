from campus_rag.config import PROJECT_ROOT
from langchain_core.documents import Document
from campus_rag.pipeline.data_preparation import DataPreparationModule

DATA_ROOT = PROJECT_ROOT / "data" / "knowledge_base" / "campus"


def _chunk(parent_id, chunk_index, content):
    return Document(
        page_content=content,
        metadata={
            "parent_id": parent_id,
            "chunk_index": chunk_index,
            "chunk_id": f"{parent_id}:{chunk_index}",
            "doc_title": "宿舍晚归登记说明",
            "doc_category": "校园生活",
            "department": "学生处",
            "file_type": "md",
            "source": "data/knowledge_base/campus/life/dorm/宿舍晚归登记说明.md",
            "section": f"第{chunk_index}段",
        },
    )


def test_campus_documents_can_be_loaded_and_chunked():
    module = DataPreparationModule(str(DATA_ROOT))

    documents = module.load_documents()
    chunks = module.chunk_documents()
    stats = module.get_statistics()

    assert documents
    assert chunks
    assert stats["total_documents"] >= 1
    assert "规章制度" in stats["categories"]
    assert set(stats["file_types"]) >= {"md", "txt", "pdf"}
    assert all("doc_title" in doc.metadata for doc in documents)
    assert all("section" not in doc.metadata for doc in documents)
    assert all("dish_name" not in doc.metadata for doc in documents)
    assert all("difficulty" not in doc.metadata for doc in documents)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
    assert all("doc_title" in chunk.metadata for chunk in chunks)
    assert all("section" in chunk.metadata for chunk in chunks)
    assert all("dish_name" not in chunk.metadata for chunk in chunks)
    assert all("difficulty" not in chunk.metadata for chunk in chunks)
    assert "difficulties" not in stats


def test_normalize_parent_metadata_does_not_add_parent_section():
    module = DataPreparationModule(str(DATA_ROOT))
    parent_doc = Document(
        page_content="学生请假管理办法",
        metadata={
            "source": "data/knowledge_base/campus/regulations/student_affairs/leave.md",
            "relative_path": "regulations/student_affairs/leave.md",
            "doc_title": "学生请假管理办法",
            "doc_id": "stable-parent-id",
        },
    )

    module._normalize_parent_metadata(parent_doc)

    assert parent_doc.metadata["doc_id"] == "stable-parent-id"
    assert parent_doc.metadata["parent_id"] == "stable-parent-id"
    assert parent_doc.metadata["doc_type"] == "parent"
    assert "section" not in parent_doc.metadata


def test_chunk_documents_normalizes_raw_splitter_output_once(monkeypatch):
    module = DataPreparationModule(str(DATA_ROOT))
    parent_doc = Document(
        page_content="第一行\r\n\r\n   第二行",
        metadata={
            "doc_title": "测试文档",
            "doc_id": "parent-1",
            "parent_id": "parent-1",
            "doc_type": "parent",
            "file_type": "txt",
        },
    )
    module.documents = [parent_doc]

    def fake_split_parent_document(_parent_doc):
        return [
            Document(page_content=" 第一行\r\n\r\n   第二行 ", metadata={}),
            Document(page_content="   \n\t ", metadata={}),
        ]

    monkeypatch.setattr(module, "_split_parent_document", fake_split_parent_document)

    chunks = module.chunk_documents()

    assert len(chunks) == 1
    assert chunks[0].page_content == "第一行\n\n第二行"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["batch_index"] == 0


def test_get_context_documents_backfills_neighbor_chunks_and_parent_metadata():
    module = DataPreparationModule(str(DATA_ROOT))
    parent_doc = Document(
        page_content="完整父文档不应直接进入生成上下文。",
        metadata={
            "parent_id": "parent-dorm",
            "doc_title": "宿舍晚归登记说明",
            "doc_category": "校园生活",
            "department": "学生处",
            "file_type": "md",
            "source": "data/knowledge_base/campus/life/dorm/宿舍晚归登记说明.md",
            "doc_type": "parent",
        },
    )
    chunks = [
        _chunk("parent-dorm", 0, "片段0：宿舍门禁开放时间。"),
        _chunk("parent-dorm", 1, "片段1：晚归需要登记。"),
        _chunk("parent-dorm", 2, "片段2：多次晚归会被提醒。"),
        _chunk("parent-dorm", 3, "片段3：违规电器另行处理。"),
    ]
    module.documents = [parent_doc]
    module.parent_documents_map = {"parent-dorm": parent_doc}
    module.chunks = chunks

    context_docs = module.get_context_documents([chunks[1]], window_size=1)

    assert len(context_docs) == 1
    context_doc = context_docs[0]
    assert context_doc.metadata["parent_id"] == "parent-dorm"
    assert context_doc.metadata["doc_title"] == "宿舍晚归登记说明"
    assert context_doc.metadata["doc_category"] == "校园生活"
    assert context_doc.metadata["department"] == "学生处"
    assert context_doc.metadata["file_type"] == "md"
    assert context_doc.metadata["source"].endswith("宿舍晚归登记说明.md")
    assert context_doc.metadata["doc_type"] == "context"
    assert context_doc.metadata["context_chunk_indices"] == [0, 1, 2]
    assert "片段0：宿舍门禁开放时间。" in context_doc.page_content
    assert "片段1：晚归需要登记。" in context_doc.page_content
    assert "片段2：多次晚归会被提醒。" in context_doc.page_content
    assert "片段3：违规电器另行处理。" not in context_doc.page_content
    assert "完整父文档不应直接进入生成上下文。" not in context_doc.page_content


def test_get_context_documents_deduplicates_overlapping_windows_in_chunk_order():
    module = DataPreparationModule(str(DATA_ROOT))
    parent_doc = Document(
        page_content="完整父文档",
        metadata={"parent_id": "parent-dorm", "doc_title": "宿舍晚归登记说明"},
    )
    chunks = [
        _chunk("parent-dorm", 0, "片段0"),
        _chunk("parent-dorm", 1, "片段1"),
        _chunk("parent-dorm", 2, "片段2"),
        _chunk("parent-dorm", 3, "片段3"),
    ]
    module.documents = [parent_doc]
    module.parent_documents_map = {"parent-dorm": parent_doc}
    module.chunks = chunks

    context_docs = module.get_context_documents([chunks[1], chunks[2]], window_size=1)

    assert len(context_docs) == 1
    assert context_docs[0].metadata["context_chunk_indices"] == [0, 1, 2, 3]
    assert context_docs[0].page_content.count("片段1") == 1
    assert context_docs[0].page_content.index("片段0") < context_docs[0].page_content.index("片段3")


def test_get_context_documents_handles_missing_neighbors_without_error():
    module = DataPreparationModule(str(DATA_ROOT))
    chunk = _chunk("parent-single", 0, "唯一命中片段")
    module.documents = []
    module.parent_documents_map = {}
    module.chunks = [chunk]

    context_docs = module.get_context_documents([chunk], window_size=2)

    assert len(context_docs) == 1
    assert context_docs[0].metadata["parent_id"] == "parent-single"
    assert context_docs[0].metadata["context_chunk_indices"] == [0]
    assert context_docs[0].page_content == "[片段 0]\n唯一命中片段"

