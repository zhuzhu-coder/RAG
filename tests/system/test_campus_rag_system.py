import os
import builtins
from types import SimpleNamespace

from campus_rag.config import RAGConfig
from langchain_core.documents import Document
from campus_rag.system import CampusRAGSystem
from campus_rag.pipeline import RAGResponse
from campus_rag.pipeline.document_ingestion import SUPPORTED_SUFFIXES


class FakeVectorRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def invoke(self, query):
        return self.chunks


class FakeVectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def as_retriever(self, **kwargs):
        self.retriever_kwargs = kwargs
        return FakeVectorRetriever(self.chunks)

    def similarity_search(self, query, k=5, filter=None):
        chunks = self.chunks
        if filter:
            chunks = [
                chunk for chunk in chunks
                if all(chunk.metadata.get(key) == value for key, value in filter.items())
            ]
        return chunks[:k]


class FakeIndexConstructionModule:
    def __init__(self, model_name, index_save_path):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.vectorstore = None
        self.saved_manifest = None
        self.received_manifest = None

    def build_manifest(self, data_path, chunks=None):
        source_count = sum(
            1
            for path in __import__("pathlib").Path(data_path).rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        return {
            "source_document_count": source_count,
            "chunk_count": len(chunks) if chunks is not None else None,
        }

    def load_index(self, expected_manifest):
        self.received_manifest = expected_manifest
        return None

    def build_vector_index(self, chunks):
        self.vectorstore = FakeVectorStore(chunks)
        return self.vectorstore

    def save_manifest(self, manifest):
        self.saved_manifest = manifest


class FakeGenerationIntegrationModule:
    def __init__(self, model_name, temperature, max_tokens):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.generated_docs = None
        self.analysis_calls = []

    def analyze_query(self, question):
        self.analysis_calls.append(question)
        return SimpleNamespace(route_type="detail", rewritten_query=question)

    def generate_step_by_step_answer(self, question, parent_docs):
        self.generated_docs = parent_docs
        doc_titles = ",".join(parent_doc.metadata["doc_title"] for parent_doc in parent_docs)
        return f"answer:{question}:{doc_titles}"


def test_campus_rag_system_runs_end_to_end_without_network(monkeypatch):
    import campus_rag.system as system_module

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(system_module, "IndexConstructionModule", FakeIndexConstructionModule)
    monkeypatch.setattr(system_module, "GenerationIntegrationModule", FakeGenerationIntegrationModule)

    config = RAGConfig(top_k=2, retrieval_candidate_k=4)
    system = CampusRAGSystem(config)

    system.initialize_system()
    system.build_knowledge_base()
    answer = system.ask_question("学生请假超过三天需要谁审批？")

    assert answer.startswith("answer:学生请假超过三天需要谁审批？:")
    assert "学生请假管理办法" in answer
    assert system.index_module.saved_manifest["chunk_count"] > 0
    assert system.index_module.received_manifest["source_document_count"] >= 1
    assert system.retrieval_module.candidate_k == 4


def test_ask_question_can_return_structured_sources(monkeypatch):
    import campus_rag.system as system_module

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(system_module, "IndexConstructionModule", FakeIndexConstructionModule)
    monkeypatch.setattr(system_module, "GenerationIntegrationModule", FakeGenerationIntegrationModule)

    config = RAGConfig(top_k=2, retrieval_candidate_k=4)
    system = CampusRAGSystem(config)

    system.initialize_system()
    system.build_knowledge_base()
    response = system.ask_question("学生请假超过三天需要谁审批？", return_sources=True)

    assert isinstance(response, RAGResponse)
    assert response.question == "学生请假超过三天需要谁审批？"
    assert response.route_type == "detail"
    assert response.rewritten_query == "学生请假超过三天需要谁审批？"
    assert response.answer.startswith("answer:学生请假超过三天需要谁审批？:")
    assert response.sources

    source = response.sources[0]
    assert source.source_id == 1
    assert source.doc_title
    assert source.doc_category
    assert source.department is not None
    assert source.file_type
    assert source.section
    assert source.source.endswith(".md")
    assert source.page is None or isinstance(source.page, int)
    assert source.chunk_index >= 0
    assert source.rrf_score is not None
    assert source.snippet


def test_generation_uses_context_docs_while_sources_keep_retrieved_chunks(monkeypatch):
    import campus_rag.system as system_module

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(system_module, "IndexConstructionModule", FakeIndexConstructionModule)
    monkeypatch.setattr(system_module, "GenerationIntegrationModule", FakeGenerationIntegrationModule)

    config = RAGConfig(top_k=2, retrieval_candidate_k=4)
    system = CampusRAGSystem(config)

    system.initialize_system()
    system.build_knowledge_base()
    response = system.ask_question("学生请假超过三天需要谁审批？", return_sources=True)

    generated_docs = system.generation_module.generated_docs
    assert generated_docs
    assert response.sources
    assert all(doc.metadata.get("doc_type") == "context" for doc in generated_docs)
    assert all(source.chunk_index >= 0 for source in response.sources)
    assert {
        doc.metadata.get("doc_title") for doc in generated_docs
    } == {
        source.doc_title for source in response.sources
    }


def test_ask_question_uses_context_window_documents_for_generation():
    system = CampusRAGSystem.__new__(CampusRAGSystem)
    system.config = RAGConfig(top_k=2, context_window_size=2)
    retrieved_chunks = [
        Document(
            page_content="命中片段：晚归需要登记。",
            metadata={
                "parent_id": "parent-dorm",
                "chunk_index": 1,
                "rrf_score": 0.04,
                "section": "晚归登记",
            },
        )
    ]
    context_docs = [
        Document(
            page_content="[片段 0]\n前文\n\n[片段 1]\n命中片段：晚归需要登记。\n\n[片段 2]\n后文",
            metadata={
                "parent_id": "parent-dorm",
                "doc_title": "宿舍晚归登记说明",
                "doc_category": "校园生活",
                "department": "学生处",
                "file_type": "md",
                "source": "data/knowledge_base/campus/life/dorm/宿舍晚归登记说明.md",
                "doc_type": "context",
                "context_window_size": 2,
                "context_chunk_indices": [0, 1, 2],
            },
        )
    ]

    class FakeRetrievalModule:
        def hybrid_search(self, query, top_k):
            self.query = query
            self.top_k = top_k
            return retrieved_chunks

    class FakeDataModule:
        def get_context_documents(self, chunks, window_size):
            self.received_chunks = chunks
            self.received_window_size = window_size
            return context_docs

    class FakeGenerationModule:
        def __init__(self):
            self.generated_docs = None
            self.analysis_calls = []

        def analyze_query(self, question):
            self.analysis_calls.append(question)
            return SimpleNamespace(route_type="detail", rewritten_query="宿舍晚归处理规定")

        def generate_step_by_step_answer(self, question, docs):
            self.generated_docs = docs
            return "基于证据窗口回答。[1]"

    system.retrieval_module = FakeRetrievalModule()
    system.data_module = FakeDataModule()
    system.generation_module = FakeGenerationModule()

    response = system.ask_question("晚归会怎么样", return_sources=True, return_trace=True)

    assert system.generation_module.analysis_calls == ["晚归会怎么样"]
    assert system.retrieval_module.query == "宿舍晚归处理规定"
    assert system.data_module.received_chunks == retrieved_chunks
    assert system.data_module.received_window_size == 2
    assert system.generation_module.generated_docs == context_docs
    assert response.rewritten_query == "宿舍晚归处理规定"
    assert response.answer == "基于证据窗口回答。[1]"
    assert response.sources[0].source_id == 1
    assert response.sources[0].doc_title == "宿舍晚归登记说明"
    assert response.trace.retrieval_strategy == "hybrid"
    assert response.trace.filters == {}
    assert set(response.trace.timings_ms) == {
        "analysis",
        "retrieval",
        "context_build",
        "generation",
        "total",
    }
    assert all(value >= 0 for value in response.trace.timings_ms.values())
    assert response.trace.retrieval_params == {
        "top_k": 2,
        "candidate_k": 10,
        "rrf_k": 60,
        "context_window_size": 2,
    }
    assert response.trace.retrieved_chunks == [
        {
            "rank": 1,
            "doc_title": "未知文档",
            "section": "晚归登记",
            "chunk_index": 1,
            "rrf_score": 0.04,
        }
    ]
    assert response.trace.context_documents == [
        {
            "source_id": 1,
            "doc_title": "宿舍晚归登记说明",
            "context_window_size": 2,
            "context_chunk_indices": [0, 1, 2],
        }
    ]
    assert response.trace.source_count == 1


def test_ask_question_returns_trace_for_empty_retrieval():
    system = CampusRAGSystem.__new__(CampusRAGSystem)
    system.config = RAGConfig(top_k=3)

    class FakeRetrievalModule:
        def hybrid_search(self, query, top_k):
            return []

    class FakeGenerationModule:
        def analyze_query(self, question):
            return SimpleNamespace(route_type="detail", rewritten_query=question)

    system.retrieval_module = FakeRetrievalModule()
    system.data_module = object()
    system.generation_module = FakeGenerationModule()

    response = system.ask_question("不存在的问题", return_sources=True, return_trace=True)

    assert response.answer == "抱歉，没有找到相关的校园文档信息。请尝试其他标题或关键词。"
    assert response.sources == []
    assert response.trace.retrieval_strategy == "hybrid"
    assert response.trace.retrieved_chunks == []
    assert response.trace.context_documents == []
    assert response.trace.source_count == 0


def test_aligned_sources_keep_all_chunks_with_parent_doc_source_ids():
    system = CampusRAGSystem.__new__(CampusRAGSystem)
    parent_docs = [
        Document(
            page_content="# 学生请假管理办法\n完整文档",
            metadata={
                "parent_id": "parent-a",
                "doc_title": "学生请假管理办法",
                "doc_category": "规章制度",
                "department": "学生处",
                "file_type": "md",
                "source": "data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md",
            },
        ),
        Document(
            page_content="# 期末考试安排\n完整文档",
            metadata={
                "parent_id": "parent-b",
                "doc_title": "期末考试安排",
                "doc_category": "教务教学",
                "department": "教务处",
                "file_type": "txt",
                "source": "data/knowledge_base/campus/teaching/exams/期末考试安排.txt",
            },
        ),
    ]
    chunks = [
        Document(
            page_content="# 请假审批\n超过三天需审批。",
            metadata={
                "parent_id": "parent-a",
                "chunk_index": 0,
                "rrf_score": 0.03,
            },
        ),
        Document(
            page_content="# 补充说明\n需要学院审核。",
            metadata={
                "parent_id": "parent-a",
                "chunk_index": 2,
                "rrf_score": 0.02,
            },
        ),
        Document(
            page_content="# 缓考申请\n通过教务系统。",
            metadata={
                "parent_id": "parent-b",
                "chunk_index": 0,
                "rrf_score": 0.01,
            },
        ),
    ]

    sources = system._build_aligned_sources(parent_docs, chunks)

    assert [source.source_id for source in sources] == [1, 1, 2]
    assert [source.doc_title for source in sources] == ["学生请假管理办法", "学生请假管理办法", "期末考试安排"]
    assert [source.section for source in sources] == ["请假审批", "补充说明", "缓考申请"]
    assert [source.chunk_index for source in sources] == [0, 2, 0]


def test_campus_rag_system_default_config_reads_current_environment(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_RETRIEVAL_CANDIDATE_K", "14")

    system = CampusRAGSystem()

    assert system.config.top_k == 7
    assert system.config.retrieval_candidate_k == 14


def test_environment_loader_does_not_search_parent_directories(monkeypatch, tmp_path):
    import campus_rag.system as system_module

    project_root = tmp_path / "repo" / ".worktrees" / "feature"
    project_root.mkdir(parents=True)
    parent_env = tmp_path / "repo" / ".env"
    parent_env.write_text("RAG_DATA_PATH=parent_data\n", encoding="utf-8")

    monkeypatch.chdir(project_root)
    monkeypatch.delenv("RAG_DATA_PATH", raising=False)

    system_module._load_environment(project_root)

    assert "RAG_DATA_PATH" not in os.environ


def test_interactive_empty_question_prompts_again(monkeypatch, capsys):
    system = CampusRAGSystem()
    monkeypatch.setattr(system, "initialize_system", lambda: None)
    monkeypatch.setattr(system, "build_knowledge_base", lambda: None)

    answers = iter(["", "退出"])
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(builtins, "input", fake_input)

    system.run_interactive()

    output = capsys.readouterr().out
    assert len(prompts) == 2
    assert "请输入问题内容" in output


def test_interactive_eof_exits_with_clear_message(monkeypatch, capsys):
    system = CampusRAGSystem()
    monkeypatch.setattr(system, "initialize_system", lambda: None)
    monkeypatch.setattr(system, "build_knowledge_base", lambda: None)

    def fake_input(prompt):
        raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)

    system.run_interactive()

    output = capsys.readouterr().out
    assert "输入流已结束" in output

