from pathlib import Path

from config import CODE_DIR, PROJECT_ROOT, RAGConfig


def test_default_paths_are_absolute_and_project_relative():
    config = RAGConfig()

    assert Path(config.data_path).is_absolute()
    assert Path(config.index_save_path).is_absolute()
    assert Path(config.data_path) == PROJECT_ROOT / "data" / "cook"
    assert Path(config.index_save_path) == PROJECT_ROOT / "vector_index"


def test_relative_paths_are_resolved_from_project_root():
    config = RAGConfig(
        data_path="custom_data",
        index_save_path="custom_index",
    )

    assert Path(config.data_path) == PROJECT_ROOT / "custom_data"
    assert Path(config.index_save_path) == PROJECT_ROOT / "custom_index"


def test_absolute_paths_are_preserved():
    data_path = PROJECT_ROOT / "absolute-recipes"
    index_path = PROJECT_ROOT / "absolute-index"

    config = RAGConfig(
        data_path=str(data_path),
        index_save_path=str(index_path),
    )

    assert Path(config.data_path) == data_path
    assert Path(config.index_save_path) == index_path
    assert CODE_DIR == PROJECT_ROOT / "code"


def test_config_can_be_created_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_DATA_PATH", "env_data")
    monkeypatch.setenv("RAG_INDEX_SAVE_PATH", "env_index")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-embedding")
    monkeypatch.setenv("RAG_LLM_MODEL", "env-llm")
    monkeypatch.setenv("RAG_TOP_K", "5")
    monkeypatch.setenv("RAG_TEMPERATURE", "0.2")
    monkeypatch.setenv("RAG_MAX_TOKENS", "512")
    monkeypatch.setenv("RAG_RETRIEVAL_CANDIDATE_K", "12")

    config = RAGConfig.from_env()

    assert Path(config.data_path) == PROJECT_ROOT / "env_data"
    assert Path(config.index_save_path) == PROJECT_ROOT / "env_index"
    assert config.embedding_model == "env-embedding"
    assert config.llm_model == "env-llm"
    assert config.top_k == 5
    assert config.temperature == 0.2
    assert config.max_tokens == 512
    assert config.retrieval_candidate_k == 12
