import os

from config import RAGConfig
from main import RecipeRAGSystem


class FakeVectorRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def as_retriever(self, **kwargs):
        self.retriever_kwargs = kwargs
        return FakeVectorRetriever(self.docs)

    def similarity_search(self, query, k=5, filter=None):
        docs = self.docs
        if filter:
            docs = [
                doc for doc in docs
                if all(doc.metadata.get(key) == value for key, value in filter.items())
            ]
        return docs[:k]


class FakeIndexConstructionModule:
    def __init__(self, model_name, index_save_path):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.vectorstore = None
        self.saved = False

    def load_index(self):
        return None

    def build_vector_index(self, chunks):
        self.vectorstore = FakeVectorStore(chunks)
        return self.vectorstore

    def save_index(self):
        self.saved = True


class FakeGenerationIntegrationModule:
    def __init__(self, model_name, temperature, max_tokens):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query_router(self, question):
        return "detail"

    def query_rewrite(self, question):
        return question

    def generate_step_by_step_answer(self, question, docs):
        dish_names = ",".join(doc.metadata["dish_name"] for doc in docs)
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
    assert system.retrieval_module.candidate_k == 4


def test_recipe_rag_system_default_config_reads_current_environment(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_RETRIEVAL_CANDIDATE_K", "14")

    system = RecipeRAGSystem()

    assert system.config.top_k == 7
    assert system.config.retrieval_candidate_k == 14
