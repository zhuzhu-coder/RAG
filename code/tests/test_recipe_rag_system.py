import os

from config import RAGConfig
from langchain_core.documents import Document
from main import RecipeRAGSystem
from rag_modules import RAGResponse


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
        self.saved = False
        self.saved_manifest = None
        self.received_manifest = None

    def build_manifest(self, data_path, chunks=None):
        return {
            "source_document_count": len(list(__import__("pathlib").Path(data_path).rglob("*.md"))),
            "chunk_count": len(chunks) if chunks is not None else None,
        }

    def load_index(self, expected_manifest):
        self.received_manifest = expected_manifest
        return None

    def build_vector_index(self, chunks):
        self.vectorstore = FakeVectorStore(chunks)
        return self.vectorstore

    def save_index(self):
        self.saved = True

    def save_manifest(self, manifest):
        self.saved_manifest = manifest


class FakeGenerationIntegrationModule:
    def __init__(self, model_name, temperature, max_tokens):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.generated_docs = None

    def query_router(self, question):
        return "detail"

    def query_rewrite(self, question):
        return question

    def generate_step_by_step_answer(self, question, parent_docs):
        self.generated_docs = parent_docs
        dish_names = ",".join(parent_doc.metadata["dish_name"] for parent_doc in parent_docs)
        return f"answer:{question}:{dish_names}"


def test_recipe_rag_system_runs_end_to_end_without_network(monkeypatch):
    import main as main_module

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "IndexConstructionModule", FakeIndexConstructionModule)
    monkeypatch.setattr(main_module, "GenerationIntegrationModule", FakeGenerationIntegrationModule)

    config = RAGConfig(top_k=2, retrieval_candidate_k=4)
    system = RecipeRAGSystem(config)

    system.initialize_system()
    system.build_knowledge_base()
    answer = system.ask_question("番茄炒蛋怎么做？")

    assert answer.startswith("answer:番茄炒蛋怎么做？:")
    assert "番茄炒蛋" in answer
    assert system.index_module.saved is True
    assert system.index_module.saved_manifest["chunk_count"] > 0
    assert system.index_module.received_manifest["source_document_count"] >= 1
    assert system.retrieval_module.candidate_k == 4


def test_ask_question_can_return_structured_sources(monkeypatch):
    import main as main_module

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "IndexConstructionModule", FakeIndexConstructionModule)
    monkeypatch.setattr(main_module, "GenerationIntegrationModule", FakeGenerationIntegrationModule)

    config = RAGConfig(top_k=2, retrieval_candidate_k=4)
    system = RecipeRAGSystem(config)

    system.initialize_system()
    system.build_knowledge_base()
    response = system.ask_question("番茄炒蛋怎么做？", return_sources=True)

    assert isinstance(response, RAGResponse)
    assert response.question == "番茄炒蛋怎么做？"
    assert response.route_type == "detail"
    assert response.rewritten_query == "番茄炒蛋怎么做？"
    assert response.answer.startswith("answer:番茄炒蛋怎么做？:")
    assert response.sources

    source = response.sources[0]
    assert source.source_id == 1
    assert source.dish_name
    assert source.category
    assert source.difficulty
    assert source.section
    assert source.source.endswith(".md")
    assert source.chunk_index >= 0
    assert source.rrf_score is not None
    assert source.snippet


def test_generation_uses_parent_docs_while_sources_keep_retrieved_chunks(monkeypatch):
    import main as main_module

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "IndexConstructionModule", FakeIndexConstructionModule)
    monkeypatch.setattr(main_module, "GenerationIntegrationModule", FakeGenerationIntegrationModule)

    config = RAGConfig(top_k=2, retrieval_candidate_k=4)
    system = RecipeRAGSystem(config)

    system.initialize_system()
    system.build_knowledge_base()
    response = system.ask_question("番茄炒蛋怎么做？", return_sources=True)

    generated_docs = system.generation_module.generated_docs
    assert generated_docs
    assert response.sources
    assert all(doc.metadata.get("doc_type") == "parent" for doc in generated_docs)
    assert all(source.chunk_index >= 0 for source in response.sources)
    assert {
        doc.metadata.get("dish_name") for doc in generated_docs
    } == {
        source.dish_name for source in response.sources
    }


def test_aligned_sources_keep_all_chunks_with_parent_doc_source_ids():
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    parent_docs = [
        Document(
            page_content="# 清炒油麦菜\n完整菜谱",
            metadata={
                "parent_id": "parent-a",
                "dish_name": "清炒油麦菜",
                "category": "素菜",
                "difficulty": "非常简单",
                "source": "data/cook/vegetable_dish/清炒油麦菜.md",
            },
        ),
        Document(
            page_content="# 蒜蓉西兰花\n完整菜谱",
            metadata={
                "parent_id": "parent-b",
                "dish_name": "蒜蓉西兰花",
                "category": "素菜",
                "difficulty": "非常简单",
                "source": "data/cook/vegetable_dish/蒜蓉西兰花.md",
            },
        ),
    ]
    chunks = [
        Document(
            page_content="# 食材\n油麦菜、蒜。",
            metadata={
                "parent_id": "parent-a",
                "chunk_index": 0,
                "rrf_score": 0.03,
            },
        ),
        Document(
            page_content="# 小贴士\n大火快炒。",
            metadata={
                "parent_id": "parent-a",
                "chunk_index": 2,
                "rrf_score": 0.02,
            },
        ),
        Document(
            page_content="# 食材\n西兰花、蒜。",
            metadata={
                "parent_id": "parent-b",
                "chunk_index": 0,
                "rrf_score": 0.01,
            },
        ),
    ]

    sources = system._build_aligned_sources(parent_docs, chunks)

    assert [source.source_id for source in sources] == [1, 1, 2]
    assert [source.dish_name for source in sources] == ["清炒油麦菜", "清炒油麦菜", "蒜蓉西兰花"]
    assert [source.section for source in sources] == ["食材", "小贴士", "食材"]
    assert [source.chunk_index for source in sources] == [0, 2, 0]


def test_recipe_rag_system_default_config_reads_current_environment(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_RETRIEVAL_CANDIDATE_K", "14")

    system = RecipeRAGSystem()

    assert system.config.top_k == 7
    assert system.config.retrieval_candidate_k == 14
