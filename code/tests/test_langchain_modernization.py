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
        if name == "rag_modules" or name.startswith("rag_modules."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(module_name)


def test_build_context_truncates_long_first_document_instead_of_dropping_it(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    doc = Document(
        page_content="做法：" + "先煎后炒。" * 40,
        metadata={"dish_name": "长菜谱", "category": "荤菜", "difficulty": "简单"},
    )

    context = generator._build_context([doc], max_length=90)

    assert "长菜谱" in context
    assert "做法" in context
    assert len(context) <= 90


def test_build_context_labels_parent_documents_with_citation_numbers(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    doc = Document(
        page_content="## 操作步骤\n先炒鸡蛋，再炒番茄。",
        metadata={
            "dish_name": "番茄炒蛋",
            "category": "素菜",
            "difficulty": "简单",
            "source": "data/cook/vegetable_dish/番茄炒蛋.md",
            "二级标题": "操作步骤",
        },
    )

    context = generator._build_context([doc])

    assert "[1] 菜谱: 番茄炒蛋" in context
    assert "章节: 操作步骤" not in context
    assert "来源: data/cook/vegetable_dish/番茄炒蛋.md" in context


def test_grounded_answer_rules_require_citations_and_no_hallucination(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.generation_integration")

    rules = module.GenerationIntegrationModule._grounded_answer_rules()

    assert "只能基于" in rules
    assert "不要编造" in rules
    assert "资料不足" in rules
    assert "[1]" in rules
    assert "参考来源" in rules
    assert "系统自动追加" in rules
    assert "不要自行编写参考来源小节" in rules


def test_normalize_route_type_handles_model_wrapped_outputs(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.generation_integration")

    assert module.GenerationIntegrationModule._normalize_route_type("detail") == "detail"
    assert module.GenerationIntegrationModule._normalize_route_type("分类结果：list。") == "list"
    assert module.GenerationIntegrationModule._normalize_route_type("答案是 general") == "general"
    assert module.GenerationIntegrationModule._normalize_route_type("无法判断") == "general"


def test_append_reference_lines_adds_references_once(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    doc = Document(
        page_content="## 操作步骤\n先炒鸡蛋。",
        metadata={
            "dish_name": "番茄炒蛋",
            "source": "data/cook/vegetable_dish/番茄炒蛋.md",
        },
    )

    answer = "番茄炒蛋需要鸡蛋和番茄。[1]"
    with_references = generator._append_reference_lines(answer, [doc])

    assert with_references == (
        "番茄炒蛋需要鸡蛋和番茄。[1]\n\n"
        "参考来源:\n"
        "[1] 番茄炒蛋 - data/cook/vegetable_dish/番茄炒蛋.md"
    )
    assert generator._append_reference_lines(with_references, [doc]) == with_references


def test_stream_with_reference_lines_adds_references_once(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    doc = Document(
        page_content="## 操作步骤\n先炒鸡蛋。",
        metadata={
            "dish_name": "番茄炒蛋",
            "source": "data/cook/vegetable_dish/番茄炒蛋.md",
        },
    )

    streamed_answer = "".join(
        generator._stream_with_reference_lines(["番茄炒蛋需要鸡蛋和番茄。[1]"], [doc])
    )

    assert streamed_answer == (
        "番茄炒蛋需要鸡蛋和番茄。[1]\n\n"
        "参考来源:\n"
        "[1] 番茄炒蛋 - data/cook/vegetable_dish/番茄炒蛋.md"
    )

    already_has_references = "".join(
        generator._stream_with_reference_lines(
            [
                "番茄炒蛋需要鸡蛋和番茄。[1]\n\n",
                "参考来源:\n[1] 番茄炒蛋",
            ],
            [doc],
        )
    )

    assert already_has_references.count("参考来源") == 1
    assert already_has_references.endswith("[1] 番茄炒蛋")


def test_generate_list_answer_includes_citation_numbers(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.generation_integration")
    generator = module.GenerationIntegrationModule.__new__(module.GenerationIntegrationModule)

    parent_docs = [
        Document(
            page_content="## 操作步骤\n先炒鸡蛋。",
            metadata={
                "dish_name": "番茄炒蛋",
                "source": "data/cook/vegetable_dish/番茄炒蛋.md",
            },
        )
    ]

    answer = generator.generate_list_answer("推荐一个素菜", parent_docs)

    assert "番茄炒蛋 [1]" in answer
    assert "参考来源" in answer
    assert "[1] 番茄炒蛋 - data/cook/vegetable_dish/番茄炒蛋.md" in answer


def test_rrf_rerank_keeps_distinct_chunks_with_same_content(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")
    retriever = module.RetrievalOptimizationModule.__new__(module.RetrievalOptimizationModule)

    vector_chunk = Document(page_content="相同内容", metadata={"chunk_id": "chunk-a"})
    bm25_chunk = Document(page_content="相同内容", metadata={"chunk_id": "chunk-b"})

    reranked = retriever._rrf_rerank([vector_chunk], [bm25_chunk])

    assert [chunk.metadata["chunk_id"] for chunk in reranked] == ["chunk-a", "chunk-b"]


def test_rrf_rerank_deduplicates_same_parent_chunk_from_different_retrievers(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")
    retriever = module.RetrievalOptimizationModule.__new__(module.RetrievalOptimizationModule)

    vector_chunk = Document(
        page_content="番茄炒蛋操作步骤",
        metadata={"chunk_id": "vector-id", "parent_id": "parent-1", "chunk_index": 2},
    )
    bm25_chunk = Document(
        page_content="番茄炒蛋操作步骤",
        metadata={"chunk_id": "bm25-id", "parent_id": "parent-1", "chunk_index": 2},
    )

    reranked = retriever._rrf_rerank([vector_chunk], [bm25_chunk])

    assert len(reranked) == 1
    assert reranked[0].metadata["rrf_score"] > 0.03


def test_retrievers_use_candidate_count_and_chinese_preprocess(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")
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
    chunks = [Document(page_content="宫保鸡丁做法", metadata={"chunk_id": "1"})]

    module.RetrievalOptimizationModule(FakeVectorStore(), chunks, candidate_k=8)

    assert captured["vector_kwargs"]["search_kwargs"]["k"] == 8
    assert captured["bm25_kwargs"]["k"] == 8
    assert callable(captured["bm25_kwargs"]["preprocess_func"])
    assert captured["bm25_kwargs"]["preprocess_func"]("宫保鸡丁")


def test_retrieval_module_exposes_vector_and_bm25_search_methods(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")

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

    assert [doc.metadata["chunk_id"] for doc in retriever.vector_search("番茄炒蛋", top_k=1)] == ["v1"]
    assert [doc.metadata["chunk_id"] for doc in retriever.bm25_search("番茄炒蛋", top_k=2)] == ["b1", "b2"]


def test_pkg_resources_shim_allows_jieba_import(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")

    for name in list(sys.modules):
        if name == "jieba" or name.startswith("jieba.") or name == "pkg_resources":
            monkeypatch.delitem(sys.modules, name, raising=False)

    with module._temporary_pkg_resources_shim():
        import jieba

    assert callable(jieba.cut_for_search)


def test_metadata_filtered_search_rebuilds_bm25_from_all_filtered_chunks(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")
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
        Document(page_content="番茄炒蛋一", metadata={"difficulty": "非常简单", "chunk_id": "a"}),
        Document(page_content="番茄炒蛋二", metadata={"difficulty": "非常简单", "chunk_id": "b"}),
        Document(page_content="番茄炒蛋三", metadata={"difficulty": "非常简单", "chunk_id": "c"}),
        Document(page_content="红烧肉", metadata={"difficulty": "困难", "chunk_id": "d"}),
    ]

    retriever = module.RetrievalOptimizationModule(FakeVectorStore(), chunks, candidate_k=2)
    retriever.metadata_filtered_search("非常简单有哪些菜", {"difficulty": "非常简单"}, top_k=2)

    assert captured["similarity_search"]["k"] == 6
    assert captured["similarity_search"]["filter"] == {"difficulty": "非常简单"}
    assert len(captured["factory_calls"]) == 2
    assert [doc.metadata["chunk_id"] for doc in captured["factory_calls"][1]["documents"]] == ["a", "b", "c"]


def test_bm25_setup_is_compatible_with_installed_langchain_community(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")

    class FakeVectorStore:
        def as_retriever(self, **kwargs):
            return object()

    chunks = [Document(page_content="宫保鸡丁做法", metadata={"chunk_id": "1"})]

    retriever = module.RetrievalOptimizationModule(FakeVectorStore(), chunks, candidate_k=1)

    assert retriever.bm25_retriever.invoke("鸡丁")


def test_source_fingerprint_changes_when_source_records_change(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.index_construction")

    first_fingerprint = module.IndexConstructionModule._fingerprint_source_records([
        {"relative_path": "vegetable_dish/番茄炒蛋.md", "content_sha256": "first"},
    ])
    second_fingerprint = module.IndexConstructionModule._fingerprint_source_records([
        {"relative_path": "vegetable_dish/番茄炒蛋.md", "content_sha256": "second"},
    ])

    assert first_fingerprint.startswith("sha256:")
    assert second_fingerprint.startswith("sha256:")
    assert first_fingerprint != second_fingerprint


def test_load_index_returns_none_when_manifest_is_missing(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.index_construction")
    indexer = module.IndexConstructionModule.__new__(module.IndexConstructionModule)
    indexer.index_save_path = str(Path(__file__).resolve().parent)
    indexer.embeddings = object()
    indexer.vectorstore = None

    def fail_load_local(*args, **kwargs):
        raise AssertionError("FAISS.load_local should not be called without a matching manifest")

    monkeypatch.setattr(module.FAISS, "load_local", fail_load_local)

    assert indexer.load_index({"schema_version": 1}) is None


def test_load_index_uses_faiss_when_manifest_matches(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.index_construction")
    indexer = module.IndexConstructionModule.__new__(module.IndexConstructionModule)
    index_path = Path(__file__).resolve().parent
    indexer.index_save_path = str(index_path)
    indexer.embeddings = object()
    indexer.vectorstore = None
    expected_manifest = {
        "schema_version": 1,
        "index_type": "FAISS",
        "embedding_model": "text-embedding-v4",
        "embedding_dimensions": 1024,
        "chunking": {"splitter": "MarkdownHeaderTextSplitter"},
        "source_fingerprint": "sha256:abc",
        "source_document_count": 1,
    }
    monkeypatch.setattr(indexer, "load_manifest", lambda: expected_manifest)
    sentinel_vectorstore = object()

    def fake_load_local(path, embeddings, allow_dangerous_deserialization):
        assert path == str(index_path)
        assert embeddings is indexer.embeddings
        assert allow_dangerous_deserialization is True
        return sentinel_vectorstore

    monkeypatch.setattr(module.FAISS, "load_local", fake_load_local)

    assert indexer.load_index(expected_manifest) is sentinel_vectorstore
