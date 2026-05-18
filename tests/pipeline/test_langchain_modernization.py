import importlib
import sys
import types
from pathlib import Path

from langchain_core.documents import Document


def _install_openai_stub(monkeypatch):
    fake_openai = types.ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeOpenAIEmbeddings:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_openai.ChatOpenAI = FakeChatOpenAI
    fake_openai.OpenAIEmbeddings = FakeOpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai)


def _reload_module(monkeypatch, module_name):
    _install_openai_stub(monkeypatch)
    for name in list(sys.modules):
        if name == "campus_rag.pipeline" or name.startswith("campus_rag.pipeline."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(module_name)


def test_build_context_truncates_long_first_document_instead_of_dropping_it(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    doc = Document(
        page_content="说明：" + "按流程提交材料。" * 40,
        metadata={"doc_title": "长文档", "doc_category": "规章制度", "department": "学生处"},
    )

    context = generator._build_context([doc], max_length=90)

    assert "长文档" in context
    assert "说明" in context
    assert len(context) <= 90


def test_build_context_labels_parent_documents_with_citation_numbers(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    doc = Document(
        page_content="## 办理流程\n先提交申请，再等待审批。",
        metadata={
            "doc_title": "学生请假管理办法",
            "doc_category": "规章制度",
            "department": "学生处",
            "source": "data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md",
            "二级标题": "办理流程",
        },
    )

    context = generator._build_context([doc])

    assert "[1] 校园文档: 学生请假管理办法" in context
    assert "来源: data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md" in context


def test_build_context_does_not_fall_back_to_legacy_domain_fields(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    doc = Document(
        page_content="## 请假流程\n先提交申请，再学院审批。",
        metadata={"source": "data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md"},
    )

    context = generator._build_context([doc])

    assert "[1] 校园文档: 未知文档" in context


def test_grounded_answer_rules_require_citations_and_no_hallucination(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")

    rules = module.GenerationIntegrationModule._grounded_answer_rules()

    assert "只能基于" in rules
    assert "不要编造" in rules
    assert "资料不足" in rules
    assert "[1]" in rules
    assert "参考来源" in rules
    assert "系统自动追加" in rules
    assert "不要自行编写参考来源小节" in rules


def test_normalize_route_type_handles_model_wrapped_outputs(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")

    assert module.GenerationIntegrationModule._normalize_route_type("detail") == "detail"
    assert module.GenerationIntegrationModule._normalize_route_type("分类结果：list。") == "list"
    assert module.GenerationIntegrationModule._normalize_route_type("答案是 general") == "general"
    assert module.GenerationIntegrationModule._normalize_route_type("无法判断") == "general"


def test_parse_query_analysis_handles_plain_json(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")

    analysis = module.GenerationIntegrationModule._parse_query_analysis(
        '{"route_type":"detail","rewritten_query":"学生晚归处理规定"}',
        "我晚归了会怎么样",
    )

    assert analysis == module.QueryAnalysis(
        route_type="detail",
        rewritten_query="学生晚归处理规定",
    )


def test_parse_query_analysis_handles_fenced_json(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")

    analysis = module.GenerationIntegrationModule._parse_query_analysis(
        '```json\n{"route_type":"general","rewritten_query":"校园卡补办第一步"}\n```',
        "校园卡丢了第一步怎么办",
    )

    assert analysis.route_type == "general"
    assert analysis.rewritten_query == "校园卡补办第一步"


def test_parse_query_analysis_falls_back_to_original_query_when_json_is_invalid(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")

    analysis = module.GenerationIntegrationModule._parse_query_analysis(
        "分类结果可能是 detail，但没有返回 JSON",
        "我晚归了会怎么样",
    )

    assert analysis == module.QueryAnalysis(
        route_type="general",
        rewritten_query="我晚归了会怎么样",
    )


def test_parse_query_analysis_keeps_list_query_original(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")

    analysis = module.GenerationIntegrationModule._parse_query_analysis(
        '{"route_type":"list","rewritten_query":"校园通知公告"}',
        "有哪些校园通知",
    )

    assert analysis == module.QueryAnalysis(
        route_type="list",
        rewritten_query="有哪些校园通知",
    )


def test_generation_module_no_longer_exposes_legacy_router_and_rewrite_methods(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")

    assert not hasattr(module.GenerationIntegrationModule, "query_router")
    assert not hasattr(module.GenerationIntegrationModule, "query_rewrite")


def test_append_reference_lines_adds_references_once(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    doc = Document(
        page_content="## 办理流程\n先提交申请。",
        metadata={
            "doc_title": "学生请假管理办法",
            "source": "data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md",
        },
    )

    answer = "请假超过三天需要学院审批。[1]"
    with_references = generator._append_reference_lines(answer, [doc])

    assert with_references == (
        "请假超过三天需要学院审批。[1]\n\n"
        "参考来源:\n"
        "[1] 学生请假管理办法 - data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md"
    )
    assert generator._append_reference_lines(with_references, [doc]) == with_references


def test_stream_with_reference_lines_adds_references_once(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    doc = Document(
        page_content="## 办理流程\n先提交申请。",
        metadata={
            "doc_title": "学生请假管理办法",
            "source": "data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md",
        },
    )

    streamed_answer = "".join(
        generator._stream_with_reference_lines(["请假超过三天需要学院审批。[1]"], [doc])
    )

    assert streamed_answer == (
        "请假超过三天需要学院审批。[1]\n\n"
        "参考来源:\n"
        "[1] 学生请假管理办法 - data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md"
    )

    already_has_references = "".join(
        generator._stream_with_reference_lines(
            [
                "请假超过三天需要学院审批。[1]\n\n",
                "参考来源:\n[1] 学生请假管理办法",
            ],
            [doc],
        )
    )

    assert already_has_references.count("参考来源") == 1
    assert already_has_references.endswith("[1] 学生请假管理办法")


def test_generate_list_answer_includes_citation_numbers(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    parent_docs = [
        Document(
            page_content="## 办理流程\n先提交申请。",
            metadata={
                "doc_title": "学生请假管理办法",
                "source": "data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md",
            },
        )
    ]

    answer = generator.generate_list_answer("列出请假相关规定", parent_docs)

    assert "学生请假管理办法 [1]" in answer
    assert "参考来源" in answer
    assert "[1] 学生请假管理办法 - data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md" in answer


def test_generation_module_default_model_matches_config_default(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.generation_integration")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    generator = module.GenerationIntegrationModule()

    assert generator.model_name == "qwen3.6-plus"
    assert generator.llm.kwargs["model"] == "qwen3.6-plus"


def test_rrf_rerank_keeps_distinct_chunks_with_same_content(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")
    retriever = module.RetrievalOptimizationModule.__new__(module.RetrievalOptimizationModule)

    vector_chunk = Document(page_content="相同内容", metadata={"chunk_id": "chunk-a"})
    bm25_chunk = Document(page_content="相同内容", metadata={"chunk_id": "chunk-b"})

    reranked = retriever._rrf_rerank([vector_chunk], [bm25_chunk])

    assert [chunk.metadata["chunk_id"] for chunk in reranked] == ["chunk-a", "chunk-b"]


def test_rrf_rerank_deduplicates_same_parent_chunk_from_different_retrievers(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")
    retriever = module.RetrievalOptimizationModule.__new__(module.RetrievalOptimizationModule)

    vector_chunk = Document(
        page_content="请假审批流程",
        metadata={"chunk_id": "vector-id", "parent_id": "parent-1", "chunk_index": 2},
    )
    bm25_chunk = Document(
        page_content="请假审批流程",
        metadata={"chunk_id": "bm25-id", "parent_id": "parent-1", "chunk_index": 2},
    )

    reranked = retriever._rrf_rerank([vector_chunk], [bm25_chunk])

    assert len(reranked) == 1
    assert reranked[0].metadata["rrf_score"] > 0.03


def test_rrf_rerank_does_not_mutate_input_chunk_metadata(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")
    retriever = module.RetrievalOptimizationModule.__new__(module.RetrievalOptimizationModule)

    original_chunk = Document(page_content="请假审批流程", metadata={"chunk_id": "chunk-a"})

    reranked = retriever._rrf_rerank([original_chunk], [])

    assert reranked[0].metadata["rrf_score"] > 0
    assert "rrf_score" not in original_chunk.metadata


def test_rrf_rerank_uses_instance_rrf_k_when_not_overridden(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")
    retriever = module.RetrievalOptimizationModule.__new__(module.RetrievalOptimizationModule)
    retriever.rrf_k = 10

    chunk = Document(page_content="考试证件要求", metadata={"chunk_id": "chunk-a"})

    reranked = retriever._rrf_rerank([chunk], [])

    assert reranked[0].metadata["rrf_score"] == 1 / 11


def test_retrievers_use_candidate_count_and_chinese_preprocess(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")
    captured = {}

    class FakeVectorStore:
        def as_retriever(self, **kwargs):
            captured["vector_kwargs"] = kwargs
            return object()

    class FakeBM25Retriever:
        @classmethod
        def from_documents(cls, documents, **kwargs):
            captured["bm25_documents"] = documents
            captured["bm25_kwargs"] = kwargs
            return object()

    monkeypatch.setattr(module, "BM25Retriever", FakeBM25Retriever)
    chunks = [Document(page_content="学生请假流程", metadata={"chunk_id": "1"})]

    module.RetrievalOptimizationModule(FakeVectorStore(), chunks, candidate_k=8)

    assert captured["vector_kwargs"]["search_kwargs"]["k"] == 8
    assert captured["bm25_kwargs"]["k"] == 8
    assert callable(captured["bm25_kwargs"]["preprocess_func"])
    assert captured["bm25_kwargs"]["preprocess_func"]("学生请假")


def test_retrieval_module_exposes_vector_and_bm25_search_methods(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")

    class FakeVectorRetriever:
        def invoke(self, query):
            return [
                Document(page_content="向量命中一", metadata={"chunk_id": "v1"}),
                Document(page_content="向量命中二", metadata={"chunk_id": "v2"}),
            ]

    class FakeBM25Retriever:
        def invoke(self, query):
            return [
                Document(page_content="关键词命中一", metadata={"chunk_id": "b1"}),
                Document(page_content="关键词命中二", metadata={"chunk_id": "b2"}),
            ]

    retriever = module.RetrievalOptimizationModule.__new__(module.RetrievalOptimizationModule)
    retriever.vector_retriever = FakeVectorRetriever()
    retriever.bm25_retriever = FakeBM25Retriever()

    assert [doc.metadata["chunk_id"] for doc in retriever.vector_search("学生请假", top_k=1)] == ["v1"]
    assert [doc.metadata["chunk_id"] for doc in retriever.bm25_search("学生请假", top_k=2)] == ["b1", "b2"]


def test_jieba_cut_for_search_imports_without_pkg_resources_shim(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")

    assert not hasattr(module, "_build_pkg_resources_shim")
    assert not hasattr(module, "_temporary_pkg_resources_shim")

    for name in list(sys.modules):
        if name == "jieba" or name.startswith("jieba.") or name == "pkg_resources":
            monkeypatch.delitem(sys.modules, name, raising=False)

    module._get_jieba_cut_for_search.cache_clear()

    assert callable(module._get_jieba_cut_for_search())


def test_metadata_filtered_search_rebuilds_bm25_from_all_filtered_chunks(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")
    captured = {"factory_calls": []}

    class FakeVectorRetriever:
        def invoke(self, query):
            return [Document(page_content="向量命中", metadata={"chunk_id": "vector"})]

    class FakeVectorStore:
        def as_retriever(self, **kwargs):
            captured["vector_kwargs"] = kwargs
            return FakeVectorRetriever()

        def similarity_search(self, query, k=5, filter=None):
            captured["similarity_search"] = {"query": query, "k": k, "filter": filter}
            return [Document(page_content="向量命中", metadata={"chunk_id": "vector"})]

    class FakeBM25Retriever:
        def __init__(self, documents):
            self.documents = documents

        def invoke(self, query):
            return list(self.documents)

    class FakeBM25RetrieverFactory:
        @classmethod
        def from_documents(cls, documents, **kwargs):
            captured["factory_calls"].append({"documents": documents, "kwargs": kwargs})
            return FakeBM25Retriever(documents)

    monkeypatch.setattr(module, "BM25Retriever", FakeBM25RetrieverFactory)

    chunks = [
        Document(page_content="学生请假管理办法一", metadata={"doc_category": "规章制度", "chunk_id": "a"}),
        Document(page_content="学生请假管理办法二", metadata={"doc_category": "规章制度", "chunk_id": "b"}),
        Document(page_content="学生请假管理办法三", metadata={"doc_category": "规章制度", "chunk_id": "c"}),
        Document(page_content="校园卡补办说明", metadata={"doc_category": "校园生活", "chunk_id": "d"}),
    ]

    retriever = module.RetrievalOptimizationModule(FakeVectorStore(), chunks, candidate_k=2)
    retriever.metadata_filtered_search("规章制度有哪些", {"doc_category": "规章制度"}, top_k=2)

    assert captured["similarity_search"]["k"] == 6
    assert captured["similarity_search"]["filter"] == {"doc_category": "规章制度"}
    assert len(captured["factory_calls"]) == 2
    assert [doc.metadata["chunk_id"] for doc in captured["factory_calls"][1]["documents"]] == ["a", "b", "c"]


def test_retrieval_module_uses_generic_metadata_filters(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")

    assert module.RetrievalOptimizationModule._to_metadata_filters(
        {"doc_category": ["规章制度", "教务教学"], "department": "学生处"}
    ) == {
        "doc_category": {"$in": ["规章制度", "教务教学"]},
        "department": "学生处",
    }


def test_metadata_filtered_search_fallback_uses_expanded_candidate_count(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")
    captured = {"similarity_calls": []}

    chunks = [
        Document(page_content="校园卡补办说明", metadata={"doc_category": "校园生活", "chunk_id": "life"}),
        Document(page_content="学生请假管理办法一", metadata={"doc_category": "规章制度", "chunk_id": "rule-a"}),
        Document(page_content="学生请假管理办法二", metadata={"doc_category": "规章制度", "chunk_id": "rule-b"}),
    ]

    class FakeVectorRetriever:
        def invoke(self, query):
            return [chunks[0]]

    class FakeVectorStore:
        def as_retriever(self, **kwargs):
            return FakeVectorRetriever()

        def similarity_search(self, query, k=5, filter=None):
            captured["similarity_calls"].append({"k": k, "filter": filter})
            if filter is not None:
                raise TypeError("filter is not supported")
            return list(chunks)

    class FakeBM25Retriever:
        def invoke(self, query):
            return []

    class FakeBM25RetrieverFactory:
        @classmethod
        def from_documents(cls, documents, **kwargs):
            return FakeBM25Retriever()

    monkeypatch.setattr(module, "BM25Retriever", FakeBM25RetrieverFactory)

    retriever = module.RetrievalOptimizationModule(FakeVectorStore(), chunks, candidate_k=2)
    results = retriever.metadata_filtered_search("规章制度有哪些", {"doc_category": "规章制度"}, top_k=2)

    assert captured["similarity_calls"] == [
        {"k": 6, "filter": {"doc_category": "规章制度"}},
        {"k": 6, "filter": None},
    ]
    assert [doc.metadata["chunk_id"] for doc in results] == ["rule-a", "rule-b"]


def test_bm25_setup_is_compatible_with_installed_langchain_community(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.retrieval_optimization")

    class FakeVectorStore:
        def as_retriever(self, **kwargs):
            return object()

    chunks = [Document(page_content="学生请假流程", metadata={"chunk_id": "1"})]

    retriever = module.RetrievalOptimizationModule(FakeVectorStore(), chunks, candidate_k=1)

    assert retriever.bm25_retriever.invoke("请假")


def test_source_fingerprint_changes_when_source_records_change(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.index_construction")

    first_fingerprint = module.IndexConstructionModule._fingerprint_source_records([
        {"relative_path": "regulations/student_affairs/学生请假管理办法.md", "content_sha256": "first"},
    ])
    second_fingerprint = module.IndexConstructionModule._fingerprint_source_records([
        {"relative_path": "regulations/student_affairs/学生请假管理办法.md", "content_sha256": "second"},
    ])

    assert first_fingerprint.startswith("sha256:")
    assert second_fingerprint.startswith("sha256:")
    assert first_fingerprint != second_fingerprint


def test_setup_embeddings_configures_request_timeout(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.index_construction")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    indexer = module.IndexConstructionModule("text-embedding-v4", index_save_path="unused-index")

    assert indexer.embeddings.kwargs["timeout"] == module.EMBEDDING_TIMEOUT_SECONDS


def test_build_manifest_tracks_md_txt_and_pdf_sources(monkeypatch, tmp_path):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.index_construction")
    indexer = module.IndexConstructionModule.__new__(module.IndexConstructionModule)
    indexer.model_name = "text-embedding-v4"
    indexer.embedding_dimensions = module.EMBEDDING_DIMENSIONS

    data_root = tmp_path / "data" / "campus"
    (data_root / "regulations").mkdir(parents=True)
    (data_root / "teaching").mkdir(parents=True)
    (data_root / "life").mkdir(parents=True)

    md_file = data_root / "regulations" / "student_affairs.md"
    txt_file = data_root / "teaching" / "exam_notice.txt"
    pdf_file = data_root / "life" / "campus_notice.pdf"

    md_file.write_text("# 学生事务管理办法", encoding="utf-8")
    txt_file.write_text("期末考试安排", encoding="utf-8")
    pdf_file.write_bytes(b"%PDF-1.4\n%%EOF")

    first_manifest = indexer.build_manifest(str(data_root))
    txt_file.write_text("期末考试安排已更新", encoding="utf-8")
    second_manifest = indexer.build_manifest(str(data_root))

    assert first_manifest["source_document_count"] == 3
    assert second_manifest["source_document_count"] == 3
    assert first_manifest["source_fingerprint"] != second_manifest["source_fingerprint"]


def test_manifest_matching_ignores_diagnostic_counts(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.index_construction")
    indexer = module.IndexConstructionModule.__new__(module.IndexConstructionModule)

    expected_manifest = {
        "schema_version": 1,
        "index_type": "Chroma",
        "embedding_model": "text-embedding-v4",
        "embedding_dimensions": 1024,
        "chunking": {"splitter": "MarkdownHeaderTextSplitter"},
        "source_fingerprint": "sha256:abc",
        "source_document_count": 2,
        "chunk_count": 20,
        "created_at": "2026-05-18T00:00:00+00:00",
    }
    cached_manifest = {
        **expected_manifest,
        "source_document_count": 1,
        "chunk_count": 10,
        "created_at": "2026-05-17T00:00:00+00:00",
    }
    monkeypatch.setattr(indexer, "load_manifest", lambda: cached_manifest)

    assert "source_document_count" not in module.MANIFEST_COMPARE_KEYS
    assert indexer.manifest_matches(expected_manifest)


def test_build_manifest_records_chunking_strategy_per_file_type(monkeypatch, tmp_path):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.index_construction")
    indexer = module.IndexConstructionModule.__new__(module.IndexConstructionModule)
    indexer.model_name = "text-embedding-v4"
    indexer.embedding_dimensions = module.EMBEDDING_DIMENSIONS

    data_root = tmp_path / "data" / "campus"
    data_root.mkdir(parents=True)

    manifest = indexer.build_manifest(str(data_root))

    assert manifest["index_type"] == "Chroma"
    assert set(manifest["chunking"]) >= {"md", "txt", "pdf"}
    assert manifest["chunking"]["md"]["splitter"] == "MarkdownHeaderTextSplitter"
    assert manifest["chunking"]["md"]["fallback"]["splitter"] == "RecursiveCharacterTextSplitter"
    assert manifest["chunking"]["txt"]["chunk_size"] == 800
    assert manifest["chunking"]["txt"]["chunk_overlap"] == 120
    assert manifest["chunking"]["pdf"]["chunk_size"] == 1000
    assert manifest["chunking"]["pdf"]["chunk_overlap"] == 150


def test_load_index_returns_none_when_manifest_is_missing(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.index_construction")
    indexer = module.IndexConstructionModule.__new__(module.IndexConstructionModule)
    indexer.index_save_path = str(Path(__file__).resolve().parent)
    indexer.embeddings = object()
    indexer.vectorstore = None

    def fail_chroma_load(*args, **kwargs):
        raise AssertionError("Chroma should not be loaded without a matching manifest")

    monkeypatch.setattr(module, "Chroma", fail_chroma_load)

    assert indexer.load_index({"schema_version": 1}) is None


def test_load_index_uses_chroma_when_manifest_matches(monkeypatch):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.index_construction")
    indexer = module.IndexConstructionModule.__new__(module.IndexConstructionModule)
    index_path = Path(__file__).resolve().parent
    indexer.index_save_path = str(index_path)
    indexer.embeddings = object()
    indexer.vectorstore = None
    expected_manifest = {
        "schema_version": 1,
        "index_type": "Chroma",
        "embedding_model": "text-embedding-v4",
        "embedding_dimensions": 1024,
        "chunking": {"splitter": "MarkdownHeaderTextSplitter"},
        "source_fingerprint": "sha256:abc",
        "source_document_count": 1,
    }
    monkeypatch.setattr(indexer, "load_manifest", lambda: expected_manifest)
    sentinel_vectorstore = object()

    def fake_chroma(collection_name, embedding_function, persist_directory):
        assert collection_name == module.CHROMA_COLLECTION_NAME
        assert embedding_function is indexer.embeddings
        assert persist_directory == str(index_path)
        return sentinel_vectorstore

    monkeypatch.setattr(module, "Chroma", fake_chroma)

    assert indexer.load_index(expected_manifest) is sentinel_vectorstore


def test_build_vector_index_resets_chroma_collection_and_uses_chunk_ids(monkeypatch, tmp_path):
    module = _reload_module(monkeypatch, "campus_rag.pipeline.index_construction")
    indexer = module.IndexConstructionModule.__new__(module.IndexConstructionModule)
    indexer.index_save_path = str(tmp_path / "storage" / "chroma")
    indexer.embeddings = object()
    indexer.vectorstore = None

    captured = {}

    class FakeChroma:
        def __init__(self, collection_name, embedding_function, persist_directory):
            captured["init"] = {
                "collection_name": collection_name,
                "embedding_function": embedding_function,
                "persist_directory": persist_directory,
            }
            self.reset_calls = 0
            self.added = None

        def reset_collection(self):
            self.reset_calls += 1

        def add_documents(self, documents, ids):
            self.added = {"documents": documents, "ids": ids}
            return ids

    monkeypatch.setattr(module, "Chroma", FakeChroma)
    chunks = [
        Document(page_content="学生请假流程", metadata={"chunk_id": "chunk-a"}),
        Document(page_content="校园卡补办", metadata={"chunk_id": "chunk-b"}),
    ]

    vectorstore = indexer.build_vector_index(chunks)

    assert captured["init"] == {
        "collection_name": module.CHROMA_COLLECTION_NAME,
        "embedding_function": indexer.embeddings,
        "persist_directory": str(tmp_path / "storage" / "chroma"),
    }
    assert vectorstore.reset_calls == 1
    assert vectorstore.added["documents"] == chunks
    assert vectorstore.added["ids"] == ["chunk-a", "chunk-b"]

