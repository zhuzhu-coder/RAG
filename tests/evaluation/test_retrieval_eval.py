from langchain_core.documents import Document

from evaluations.run_retrieval_eval import (
    DEFAULT_EVAL_SET,
    DEFAULT_STRATEGIES,
    configure_utf8_stdio,
    _build_bm25_only_search_fn,
    evaluate_retrieval_cases,
    extract_doc_titles,
    keyword_coverage,
    reciprocal_rank,
    load_eval_cases,
)
from campus_rag.config import RAGConfig
from pathlib import Path
import json


def test_extract_doc_titles_prefers_metadata_over_content():
    docs = [
        Document(page_content="# 学生请假管理办法\n请假审批流程", metadata={"doc_title": "学生请假管理办法"}),
        Document(page_content="# 学生请假管理办法\n补充说明", metadata={"doc_title": "学生请假管理办法"}),
        Document(page_content="# 期末考试安排\n考试通知", metadata={}),
    ]

    assert extract_doc_titles(docs) == ["学生请假管理办法", "期末考试安排"]


def test_reciprocal_rank_returns_first_expected_doc_title():
    ranked_doc_titles = ["校园卡补办说明", "学生请假管理办法", "期末考试安排"]

    assert reciprocal_rank(ranked_doc_titles, ["期末考试安排", "学生请假管理办法"]) == 0.5
    assert reciprocal_rank(ranked_doc_titles, ["不存在的文档"]) == 0.0


def test_keyword_coverage_counts_expected_keywords_in_returned_docs():
    docs = [
        Document(page_content="校园卡补办需要身份证和学生证。", metadata={"doc_title": "校园卡补办说明"}),
        Document(page_content="请到综合服务大厅办理。", metadata={"doc_title": "综合服务大厅办事指南"}),
    ]

    assert keyword_coverage(docs, ["身份证", "学生证", "照片"]) == 2 / 3
    assert keyword_coverage(docs, []) == 0.0


def test_evaluate_retrieval_cases_computes_strategy_metrics():
    cases = [
        {
            "id": "case-1",
            "question": "学生请假超过三天需要谁审批？",
            "expected_doc_titles": ["学生请假管理办法"],
            "expected_keywords": ["辅导员", "学院"],
        },
        {
            "id": "case-2",
            "question": "校园卡补办需要什么材料？",
            "expected_doc_titles": ["校园卡补办说明"],
            "expected_keywords": ["身份证", "学生证"],
        },
    ]

    strategy_results = {
        "vector": {
            "学生请假超过三天需要谁审批？": [
                Document(page_content="# 学生请假管理办法\n请假审批", metadata={"doc_title": "学生请假管理办法"})
            ],
            "校园卡补办需要什么材料？": [
                Document(page_content="# 期末考试安排\n考试通知", metadata={"doc_title": "期末考试安排"})
            ],
        },
        "hybrid": {
            "学生请假超过三天需要谁审批？": [
                Document(page_content="# 学生请假管理办法\n辅导员和学院审批", metadata={"doc_title": "学生请假管理办法"})
            ],
            "校园卡补办需要什么材料？": [
                Document(page_content="# 期末考试安排\n考试通知", metadata={"doc_title": "期末考试安排"}),
                Document(page_content="# 校园卡补办说明\n身份证和学生证", metadata={"doc_title": "校园卡补办说明"}),
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
    assert report["strategies"]["vector"]["keyword_coverage"] == 0.0
    assert report["strategies"]["hybrid"]["hit_at_1"] == 0.5
    assert report["strategies"]["hybrid"]["hit_at_3"] == 1.0
    assert report["strategies"]["hybrid"]["mrr"] == 0.75
    assert report["strategies"]["hybrid"]["keyword_coverage"] == 1.0
    assert report["cases"][1]["results"]["hybrid"]["ranked_doc_titles"] == ["期末考试安排", "校园卡补办说明"]
    assert report["cases"][1]["results"]["hybrid"]["keyword_coverage"] == 1.0
    assert "expected_dishes" not in report["cases"][0]
    assert "ranked_dishes" not in report["cases"][1]["results"]["hybrid"]


def test_default_eval_set_has_expanded_campus_schema():
    cases = load_eval_cases(DEFAULT_EVAL_SET)

    assert len(cases) >= 30
    assert set(DEFAULT_STRATEGIES) == {"vector", "bm25", "hybrid"}
    for case in cases:
        assert case["expected_doc_titles"]
        assert case["expected_keywords"]
        assert case["category"] in {"regulations", "teaching", "life", "notices"}
        assert case["query_type"] in {"precise", "fuzzy", "procedure", "list"}


def test_load_eval_cases_rejects_legacy_expected_dishes_field(tmp_path):
    legacy_eval = tmp_path / "legacy.jsonl"
    legacy_eval.write_text(
        json.dumps(
            {
                "id": "legacy",
                "question": "学生请假超过三天需要谁审批？",
                "expected_dishes": ["学生请假管理办法"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    try:
        load_eval_cases(legacy_eval)
        raised = False
    except ValueError:
        raised = True

    assert raised


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
    docs = search_fn("bm25", "学生请假超过三天需要谁审批？", top_k=1)

    assert docs
    assert docs[0].metadata["doc_title"] == "学生请假管理办法"

