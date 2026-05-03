import importlib
import sys
import types

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


def test_rrf_rerank_keeps_distinct_chunks_with_same_content(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")
    retriever = module.RetrievalOptimizationModule.__new__(module.RetrievalOptimizationModule)

    vector_doc = Document(page_content="相同内容", metadata={"chunk_id": "chunk-a"})
    bm25_doc = Document(page_content="相同内容", metadata={"chunk_id": "chunk-b"})

    reranked = retriever._rrf_rerank([vector_doc], [bm25_doc])

    assert [doc.metadata["chunk_id"] for doc in reranked] == ["chunk-a", "chunk-b"]


def test_rrf_rerank_deduplicates_same_parent_chunk_from_different_retrievers(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")
    retriever = module.RetrievalOptimizationModule.__new__(module.RetrievalOptimizationModule)

    vector_doc = Document(
        page_content="番茄炒蛋操作步骤",
        metadata={"chunk_id": "vector-id", "parent_id": "parent-1", "chunk_index": 2},
    )
    bm25_doc = Document(
        page_content="番茄炒蛋操作步骤",
        metadata={"chunk_id": "bm25-id", "parent_id": "parent-1", "chunk_index": 2},
    )

    reranked = retriever._rrf_rerank([vector_doc], [bm25_doc])

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
    docs = [Document(page_content="宫保鸡丁做法", metadata={"chunk_id": "1"})]

    module.RetrievalOptimizationModule(FakeVectorStore(), docs, candidate_k=8)

    assert captured["vector_kwargs"]["search_kwargs"]["k"] == 8
    assert captured["bm25_kwargs"]["k"] == 8
    assert callable(captured["bm25_kwargs"]["preprocess_func"])
    assert captured["bm25_kwargs"]["preprocess_func"]("宫保鸡丁")


def test_bm25_setup_is_compatible_with_installed_langchain_community(monkeypatch):
    module = _reload_module(monkeypatch, "rag_modules.retrieval_optimization")

    class FakeVectorStore:
        def as_retriever(self, **kwargs):
            return object()

    docs = [Document(page_content="宫保鸡丁做法", metadata={"chunk_id": "1"})]

    retriever = module.RetrievalOptimizationModule(FakeVectorStore(), docs, candidate_k=1)

    assert retriever.bm25_retriever.invoke("鸡丁")
