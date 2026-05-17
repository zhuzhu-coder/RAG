from pathlib import Path


def test_public_package_imports_are_available():
    from campus_rag.api import create_app
    from campus_rag.config import PROJECT_ROOT, RAGConfig
    from campus_rag.pipeline import RAGResponse
    from campus_rag.system import CampusRAGSystem

    assert create_app is not None
    assert CampusRAGSystem is not None
    assert RAGResponse is not None
    assert Path(RAGConfig().data_path) == PROJECT_ROOT / "data" / "knowledge_base" / "campus"
    assert Path(RAGConfig().index_save_path) == PROJECT_ROOT / "storage" / "chroma"


def test_project_directories_are_professionally_grouped():
    from campus_rag.config import PROJECT_ROOT

    assert (PROJECT_ROOT / "src" / "campus_rag").is_dir()
    assert (PROJECT_ROOT / "frontend" / "src" / "app").is_dir()
    assert (PROJECT_ROOT / "frontend" / "src" / "api").is_dir()
    assert (PROJECT_ROOT / "frontend" / "src" / "components").is_dir()
    assert (PROJECT_ROOT / "frontend" / "src" / "styles").is_dir()
    assert (PROJECT_ROOT / "frontend" / "src" / "types").is_dir()
    assert (PROJECT_ROOT / "evaluations" / "datasets").is_dir()
    assert (PROJECT_ROOT / "evaluations" / "results").is_dir()
    assert (PROJECT_ROOT / "tests" / "api").is_dir()
    assert (PROJECT_ROOT / "tests" / "evaluation").is_dir()
    assert (PROJECT_ROOT / "tests" / "pipeline").is_dir()
    assert (PROJECT_ROOT / "tests" / "system").is_dir()
    assert not (PROJECT_ROOT / "evals").exists()

