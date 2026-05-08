from langchain_core.documents import Document

from evals.run_retrieval_eval import (
    configure_utf8_stdio,
    _build_bm25_only_search_fn,
    evaluate_retrieval_cases,
    extract_dish_names,
    reciprocal_rank,
)
from config import RAGConfig


def test_extract_dish_names_prefers_metadata_over_content():
    docs = [
        Document(page_content="# 番茄炒蛋\n做法", metadata={"dish_name": "番茄炒蛋"}),
        Document(page_content="# 番茄炒蛋\n步骤", metadata={"dish_name": "番茄炒蛋"}),
        Document(page_content="# 红烧肉\n做法", metadata={}),
    ]

    assert extract_dish_names(docs) == ["番茄炒蛋", "红烧肉"]


def test_reciprocal_rank_returns_first_expected_dish_rank():
    ranked_dishes = ["蒜蓉西兰花", "番茄炒蛋", "红烧肉"]

    assert reciprocal_rank(ranked_dishes, ["红烧肉", "番茄炒蛋"]) == 0.5
    assert reciprocal_rank(ranked_dishes, ["不存在的菜"]) == 0.0


def test_evaluate_retrieval_cases_computes_strategy_metrics():
    cases = [
        {
            "id": "case-1",
            "question": "番茄炒蛋怎么做？",
            "expected_dishes": ["番茄炒蛋"],
        },
        {
            "id": "case-2",
            "question": "推荐一个排骨菜",
            "expected_dishes": ["糖醋排骨"],
        },
    ]

    strategy_results = {
        "vector": {
            "番茄炒蛋怎么做？": [
                Document(page_content="# 番茄炒蛋\n做法", metadata={"dish_name": "番茄炒蛋"})
            ],
            "推荐一个排骨菜": [
                Document(page_content="# 红烧肉\n做法", metadata={"dish_name": "红烧肉"})
            ],
        },
        "hybrid": {
            "番茄炒蛋怎么做？": [
                Document(page_content="# 番茄炒蛋\n做法", metadata={"dish_name": "番茄炒蛋"})
            ],
            "推荐一个排骨菜": [
                Document(page_content="# 红烧肉\n做法", metadata={"dish_name": "红烧肉"}),
                Document(page_content="# 糖醋排骨\n做法", metadata={"dish_name": "糖醋排骨"}),
            ],
        },
    }

    def search_fn(strategy_name, question, top_k):
        return strategy_results[strategy_name][question][:top_k]

    report = evaluate_retrieval_cases(
        cases,
        strategies=["vector", "hybrid"],
        search_fn=search_fn,
        top_k=3,
    )

    assert report["summary"]["case_count"] == 2
    assert report["strategies"]["vector"]["hit_at_1"] == 0.5
    assert report["strategies"]["vector"]["hit_at_3"] == 0.5
    assert report["strategies"]["vector"]["mrr"] == 0.5
    assert report["strategies"]["hybrid"]["hit_at_1"] == 0.5
    assert report["strategies"]["hybrid"]["hit_at_3"] == 1.0
    assert report["strategies"]["hybrid"]["mrr"] == 0.75
    assert report["cases"][1]["results"]["hybrid"]["ranked_dishes"] == ["红烧肉", "糖醋排骨"]


def test_configure_utf8_stdio_reconfigures_streams_when_supported():
    class FakeStream:
        def __init__(self):
            self.received_encoding = None

        def reconfigure(self, encoding):
            self.received_encoding = encoding

    stdout = FakeStream()
    stderr = FakeStream()

    configure_utf8_stdio(stdout, stderr)

    assert stdout.received_encoding == "utf-8"
    assert stderr.received_encoding == "utf-8"


def test_build_bm25_only_search_fn_does_not_require_dashscope_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    search_fn = _build_bm25_only_search_fn(RAGConfig())
    docs = search_fn("bm25", "番茄炒蛋怎么做？", top_k=1)

    assert docs
    assert docs[0].metadata["dish_name"] == "番茄炒蛋"
