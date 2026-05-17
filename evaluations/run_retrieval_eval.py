"""
RAG检索评估入口

运行方式:
    python evaluations/run_retrieval_eval.py
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from campus_rag.config import RAGConfig
from campus_rag.pipeline.data_preparation import DataPreparationModule
from campus_rag.pipeline.retrieval_optimization import tokenize_chinese_text

# 默认评估集文件路径
DEFAULT_EVAL_SET = Path(__file__).resolve().parent / "datasets" / "campus_smoke_eval_set.jsonl"
# 默认评估的检索策略
DEFAULT_STRATEGIES = ("vector", "bm25", "hybrid")


def configure_utf8_stdio(stdout=None, stderr=None):
    """将标准输出和标准错误流配置为 UTF-8 编码"""
    import sys

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    for stream in (stdout, stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def load_eval_cases(path: Path) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件读取评估问题集
    """
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if not case.get("id"):
                raise ValueError(f"{path}:{line_number} 缺少 id")
            if not case.get("question"):
                raise ValueError(f"{path}:{line_number} 缺少 question")

            expected_doc_titles = case.get("expected_doc_titles")
            if not expected_doc_titles:
                raise ValueError(f"{path}:{line_number} 缺少 expected_doc_titles")

            case["expected_doc_titles"] = expected_doc_titles
            cases.append(case)
    return cases


def extract_doc_titles(docs: Iterable[Document]) -> List[str]:
    """
    从检索结果中提取文档标题，用于和 expected_doc_titles 对齐
    """
    doc_titles = []
    seen = set()
    for doc in docs:
        metadata = doc.metadata or {}
        doc_title = metadata.get("doc_title")
        if not doc_title:
            first_line = (doc.page_content or "").strip().splitlines()[0:1]
            doc_title = first_line[0].replace("#", "").strip() if first_line else ""
        doc_title = doc_title or "未知文档"
        if doc_title in seen:
            continue
        doc_titles.append(doc_title)
        seen.add(doc_title)
    return doc_titles


def reciprocal_rank(ranked_doc_titles: List[str], expected_doc_titles: List[str]) -> float:
    """
    计算第一个命中标准文档标题的倒数排名
    """
    expected = set(expected_doc_titles)
    for rank, doc_title in enumerate(ranked_doc_titles, 1):
        if doc_title in expected:
            return 1.0 / rank
    return 0.0


def keyword_coverage(docs: Iterable[Document], expected_keywords: List[str]) -> float:
    """
    计算检索结果正文对 expected_keywords 的覆盖率
    """
    if not expected_keywords:
        return 0.0

    combined_text = "\n".join(doc.page_content or "" for doc in docs)
    matched_count = sum(1 for keyword in expected_keywords if keyword in combined_text)
    return matched_count / len(expected_keywords)


def _hit_at(ranked_doc_titles: List[str], expected_doc_titles: List[str], k: int) -> float:
    """计算 hit@k 指标"""
    expected = set(expected_doc_titles)
    return 1.0 if any(doc_title in expected for doc_title in ranked_doc_titles[:k]) else 0.0


def evaluate_retrieval_cases(
    cases: List[Dict[str, Any]],
    strategies: Iterable[str],
    search_fn: Callable[[str, str, int], List[Document]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    对多种检索策略计算 hit@1、hit@k 和 MRR
    """
    strategies = list(strategies)
    metric_sums = {
        strategy: {"hit_at_1": 0.0, f"hit_at_{top_k}": 0.0, "mrr": 0.0, "keyword_coverage": 0.0}
        for strategy in strategies
    }
    case_reports = []

    for case in cases:
        question = case["question"]
        expected_doc_titles = case["expected_doc_titles"]
        expected_keywords = case.get("expected_keywords", [])
        case_report = {
            "id": case["id"],
            "question": question,
            "expected_doc_titles": expected_doc_titles,
            "expected_keywords": expected_keywords,
            "results": {},
        }

        for strategy in strategies:
            docs = search_fn(strategy, question, top_k)
            ranked_doc_titles = extract_doc_titles(docs)
            hit_at_1 = _hit_at(ranked_doc_titles, expected_doc_titles, 1)
            hit_at_k = _hit_at(ranked_doc_titles, expected_doc_titles, top_k)
            mrr = reciprocal_rank(ranked_doc_titles, expected_doc_titles)
            coverage = keyword_coverage(docs, expected_keywords)

            metric_sums[strategy]["hit_at_1"] += hit_at_1
            metric_sums[strategy][f"hit_at_{top_k}"] += hit_at_k
            metric_sums[strategy]["mrr"] += mrr
            metric_sums[strategy]["keyword_coverage"] += coverage
            case_report["results"][strategy] = {
                "ranked_doc_titles": ranked_doc_titles,
                "hit_at_1": hit_at_1,
                f"hit_at_{top_k}": hit_at_k,
                "mrr": mrr,
                "keyword_coverage": coverage,
            }

        case_reports.append(case_report)

    case_count = len(cases)
    strategy_metrics = {}
    for strategy, sums in metric_sums.items():
        strategy_metrics[strategy] = {
            metric_name: round(metric_sum / case_count, 4) if case_count else 0.0
            for metric_name, metric_sum in sums.items()
        }

    return {
        "summary": {
            "case_count": case_count,
            "top_k": top_k,
            "strategies": strategies,
        },
        "strategies": strategy_metrics,
        "cases": case_reports,
    }


def _build_search_fn(system: Any):
    """
    构建一个检索函数，根据系统配置选择合适的检索模块
    """
    retrieval_module = system.retrieval_module

    def search(strategy: str, question: str, top_k: int) -> List[Document]:
        if strategy == "vector":
            return retrieval_module.vector_search(question, top_k=top_k)
        if strategy == "bm25":
            return retrieval_module.bm25_search(question, top_k=top_k)
        if strategy == "hybrid":
            return retrieval_module.hybrid_search(question, top_k=top_k)
        raise ValueError(f"未知检索策略: {strategy}")

    return search


def _build_bm25_only_search_fn(config: RAGConfig):
    """
    构建只依赖本地文档和 BM25 的检索函数，适合纯检索评估
    """
    data_module = DataPreparationModule(config.data_path)
    data_module.load_documents()
    chunks = data_module.chunk_documents()
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        k=3,
        preprocess_func=tokenize_chinese_text,
    )

    def search(strategy: str, question: str, top_k: int) -> List[Document]:
        if strategy != "bm25":
            raise ValueError("bm25-only 模式只支持 bm25 策略")
        bm25_retriever.k = top_k
        return bm25_retriever.invoke(question)

    return search


def _build_full_search_fn(config: RAGConfig):
    """
    构建需要完整 RAG 系统的检索函数
    """
    from campus_rag.system import CampusRAGSystem

    system = CampusRAGSystem(config)
    system.initialize_system()
    system.build_knowledge_base()
    return _build_search_fn(system)


def print_report(report: Dict[str, Any]):
    """打印适合命令行阅读的评估报告"""
    summary = report["summary"]
    top_k = summary["top_k"]
    print(f"评估问题数: {summary['case_count']}")
    print(f"Top K: {top_k}")
    print("\n策略指标:")
    for strategy, metrics in report["strategies"].items():
        print(
            f"- {strategy}: "
            f"hit@1={metrics['hit_at_1']:.4f}, "
            f"hit@{top_k}={metrics[f'hit_at_{top_k}']:.4f}, "
            f"mrr={metrics['mrr']:.4f}, "
            f"keyword_coverage={metrics['keyword_coverage']:.4f}"
        )

    print("\n逐题结果:")
    for case in report["cases"]:
        print(f"- {case['id']} {case['question']}")
        print(f"  expected: {', '.join(case['expected_doc_titles'])}")
        for strategy, result in case["results"].items():
            ranked = ", ".join(result["ranked_doc_titles"])
            print(
                f"  {strategy}: hit@1={result['hit_at_1']:.0f}, "
                f"hit@{top_k}={result[f'hit_at_{top_k}']:.0f}, "
                f"mrr={result['mrr']:.4f}, "
                f"keyword_coverage={result['keyword_coverage']:.4f}, ranked=[{ranked}]"
            )


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="运行校园RAG检索评估")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=list(DEFAULT_STRATEGIES),
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_set)
    config = RAGConfig.from_env()
    if set(args.strategies) == {"bm25"}:
        search_fn = _build_bm25_only_search_fn(config)
    else:
        search_fn = _build_full_search_fn(config)
    report = evaluate_retrieval_cases(
        cases,
        strategies=args.strategies,
        search_fn=search_fn,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
