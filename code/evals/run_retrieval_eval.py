"""
RAG检索评估入口

运行方式:
    python evals/run_retrieval_eval.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

# 导入项目根目录
CODE_DIR = Path(__file__).resolve().parents[1] # 当前文件绝对路径的上两层目录 
# 加入模块搜索路径
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from config import RAGConfig
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.retrieval_optimization import tokenize_chinese_text

# 默认评估集文件路径
DEFAULT_EVAL_SET = Path(__file__).resolve().with_name("recipe_eval_set.jsonl")
# 默认评估的检索策略
DEFAULT_STRATEGIES = ("vector", "bm25", "hybrid")


def configure_utf8_stdio(stdout=sys.stdout, stderr=sys.stderr):
    """将标准输出和标准错误流配置为 UTF-8 编码"""
    # 分别处理输出流和错误流
    for stream in (stdout, stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def load_eval_cases(path: Path) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件读取评估问题集
    Args:
        path: JSONL 文件路径
    Returns:
        评估问题集列表
    """
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # 解析 JSON 为字典格式
            case = json.loads(line)
            if not case.get("id"):
                raise ValueError(f"{path}:{line_number} 缺少 id")
            if not case.get("question"):
                raise ValueError(f"{path}:{line_number} 缺少 question")
            if not case.get("expected_dishes"):
                raise ValueError(f"{path}:{line_number} 缺少 expected_dishes")
            cases.append(case)
    return cases


def extract_dish_names(docs: Iterable[Document]) -> List[str]:
    """
    从检索结果中提取菜名，用于和 expected_dishes 对齐
    Args:
        docs: 检索结果文档列表
    Returns:
        菜名列表
    """
    dish_names = []
    seen = set()
    for doc in docs:
        metadata = doc.metadata or {}
        dish_name = metadata.get("dish_name")
        if not dish_name:
            # 如果元数据中没有菜名，尝试从文档内容的第一行提取菜名
            first_line = (doc.page_content or "").strip().splitlines()[0:1]
            dish_name = first_line[0].replace("#", "").strip() if first_line else ""
        dish_name = dish_name or "未知菜品"
        if dish_name in seen:
            continue
        dish_names.append(dish_name)
        seen.add(dish_name)
    return dish_names


def reciprocal_rank(ranked_dishes: List[str], expected_dishes: List[str]) -> float:
    """
    计算第一个命中标准菜名的倒数排名
    Args:
        ranked_dishes: 检索结果菜名列表
        expected_dishes: 标准菜名列表
    Returns:
        倒数排名
    """
    expected = set(expected_dishes)
    for rank, dish_name in enumerate(ranked_dishes, 1):
        if dish_name in expected:
            return 1.0 / rank
    return 0.0


def _hit_at(ranked_dishes: List[str], expected_dishes: List[str], k: int) -> float:
    """
    计算 hit@k 指标
    Args:
        ranked_dishes: 检索结果菜名列表
        expected_dishes: 标准菜名列表
        k: 要计算的 hit@k 指标
    Returns:
        hit@k 指标值
    """
    expected = set(expected_dishes)
    # 只要检索结果中包含一个标准菜名，hit@k 就为 1
    return 1.0 if any(dish_name in expected for dish_name in ranked_dishes[:k]) else 0.0


def evaluate_retrieval_cases(
    cases: List[Dict[str, Any]],
    strategies: Iterable[str],
    search_fn: Callable[[str, str, int], List[Document]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    对多种检索策略计算 hit@1、hit@k 和 MRR
    Args:
        cases: 评估问题列表
        strategies: 需要评估的策略名
        search_fn: 接收 strategy、question、top_k 并返回文档列表的函数
        top_k: 每个策略返回的检索结果数量
    Returns:
        包含总体指标和逐题结果的报告字典
    """
    strategies = list(strategies)
    # 给每个策略准备一组指标累计值
    metric_sums = {
        strategy: {"hit_at_1": 0.0, f"hit_at_{top_k}": 0.0, "mrr": 0.0}
        for strategy in strategies
    }
    # 逐题结果列表
    case_reports = []

    for case in cases:
        question = case["question"]
        expected_dishes = case["expected_dishes"]
        case_report = {
            "id": case["id"],
            "question": question,
            "expected_dishes": expected_dishes,
            "results": {},
        }
        # 遍历每个策略
        for strategy in strategies:
            docs = search_fn(strategy, question, top_k)
            ranked_dishes = extract_dish_names(docs)
            hit_at_1 = _hit_at(ranked_dishes, expected_dishes, 1)  # 第一个结果是否命中标准答案
            hit_at_k = _hit_at(ranked_dishes, expected_dishes, top_k)  # top_k 个结果是否命中标准答案
            mrr = reciprocal_rank(ranked_dishes, expected_dishes)  # 第一个命中标准菜名的倒数排名
            # 累加到指标累计值中
            metric_sums[strategy]["hit_at_1"] += hit_at_1
            metric_sums[strategy][f"hit_at_{top_k}"] += hit_at_k
            metric_sums[strategy]["mrr"] += mrr
            # 记录逐题结果
            case_report["results"][strategy] = {
                "ranked_dishes": ranked_dishes,
                "hit_at_1": hit_at_1,
                f"hit_at_{top_k}": hit_at_k,
                "mrr": mrr,
            }

        case_reports.append(case_report)
    # 把累计值转换为平均值
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
    Args:
        system: 食谱RAG系统对象
    Returns:
        检索函数，接收 strategy、question、top_k 并返回文档列表
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
    Args:
        config: RAG 配置
    Returns:
        检索函数，接收 strategy、question、top_k 并返回文档列表
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
    Args:
        config: RAG 配置
    Returns:
        检索函数，接收 strategy、question、top_k 并返回文档列表
    """
    from main import RecipeRAGSystem

    system = RecipeRAGSystem(config)
    system.initialize_system()
    system.build_knowledge_base()
    return _build_search_fn(system)


def print_report(report: Dict[str, Any]):
    """
    打印适合命令行阅读的评估报告
    Args:
        report: 包含总体指标和逐题结果的报告字典
    """
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
            f"mrr={metrics['mrr']:.4f}"
        )

    print("\n逐题结果:")
    for case in report["cases"]:
        print(f"- {case['id']} {case['question']}")
        print(f"  expected: {', '.join(case['expected_dishes'])}")
        for strategy, result in case["results"].items():
            ranked = ", ".join(result["ranked_dishes"])
            print(
                f"  {strategy}: hit@1={result['hit_at_1']:.0f}, "
                f"hit@{top_k}={result[f'hit_at_{top_k}']:.0f}, "
                f"mrr={result['mrr']:.4f}, ranked=[{ranked}]"
            )


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="运行食谱RAG检索评估")
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
