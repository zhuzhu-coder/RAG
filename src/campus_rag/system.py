"""Campus knowledge RAG orchestration."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from .config import PROJECT_ROOT, RAGConfig
from .pipeline import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RAGResponse,
    RAGTrace,
    RetrievedSource,
    RetrievalOptimizationModule,
)

def _load_environment(project_root: Path = PROJECT_ROOT) -> None:
    """加载环境变量，只读取项目根目录或当前目录的 .env"""
    project_env = project_root / ".env"
    if project_env.exists():
        load_dotenv(project_env)
        return

    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env)


_load_environment()

# 配置日志
logging.basicConfig(
    # 设置最低日志级别
    level=logging.INFO,
    # 设置日志格式，包含时间、模块名、日志级别、日志消息内容
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# 设置jieba日志级别为WARNING，避免打印详细日志
logging.getLogger("jieba").setLevel(logging.WARNING)
# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)

class CampusRAGSystem:
    """校园知识库RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统
        Args:
            config: RAG系统配置，默认从环境变量读取
        """
        self.config = config or RAGConfig.from_env()
        # 数据准备模块
        self.data_module = None
        # 向量索引模块
        self.index_module = None
        # 检索模块
        self.retrieval_module = None
        # 回答生成模块
        self.generation_module = None

        # 检查数据路径是否存在；如果配置路径不可用，尝试回退到默认校园知识库目录
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            campus_data_path = PROJECT_ROOT / "data" / "knowledge_base" / "campus"
            if campus_data_path.exists():
                logger.warning("数据路径不存在，已回退到校园知识库目录: %s -> %s", data_path, campus_data_path)
                self.config.data_path = str(campus_data_path)
            else:
                raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 在环境变量中检查API密钥是否存在
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 用源文件指纹和索引配置判断 Chroma 本地缓存是否仍然可用
        print("检查向量索引缓存...")
        # 构建预期的缓存校验 manifest
        expected_manifest = self.index_module.build_manifest(self.config.data_path)
        # 加载索引
        vectorstore = self.index_module.load_index(expected_manifest)

        # 2. 无论向量索引是否命中缓存，BM25 和父文档回答都需要当前文档与分块
        print("加载校园文档...")
        self.data_module.load_documents()
        print("进行文本分块...")
        chunks = self.data_module.chunk_documents()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
        else:
            print("未找到可用索引或索引已过期，开始构建新索引...")

            # 3. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 4. 保存索引缓存校验元数据；Chroma 会通过 persist_directory 自动持久化
            print("保存索引缓存元数据...")
            self.index_module.save_manifest(
                self.index_module.build_manifest(self.config.data_path, chunks)
            )

        # 5. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore,
            chunks,
            candidate_k=self.config.retrieval_candidate_k,
            rrf_k=self.config.rrf_k,
        )

        # 6. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   文档分类: {list(stats['categories'].keys())}")
        print(f"   部门分布: {list(stats['departments'].keys())}")
        print(f"   文件类型: {list(stats['file_types'].keys())}")

        print("✅ 知识库构建完成！")
    
    def ask_question(
        self,
        question: str,
        stream: bool = False,
        return_sources: bool = False,
        return_trace: bool = False,
    ):
        """
        回答用户问题
        Args:
            question: 用户问题
            stream: 是否使用流式输出
            return_sources: 是否返回结构化回答和检索来源 （默认否）
            return_trace: 是否返回查询链路调试信息（默认否）
        Returns:
            生成的回答、生成器或结构化RAG响应
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        if stream and (return_sources or return_trace):
            raise ValueError("结构化来源或trace返回暂不支持流式输出")

        total_start = time.perf_counter()
        timings_ms = {
            "analysis": 0.0, # 合并查询分析耗时
            "retrieval": 0.0, # 检索耗时
            "context_build": 0.0, # 上下文构建耗时
            "generation": 0.0, # 回答生成耗时
            "total": 0.0, # 总耗时
        }
        
        print(f"\n❓ 用户问题: {question}")

        # 1. 合并查询分析：一次 LLM 调用同时完成路由和改写
        print("🤖 智能分析查询...")
        analysis_start = time.perf_counter()
        query_analysis = self.generation_module.analyze_query(question)
        timings_ms["analysis"] = self._elapsed_ms(analysis_start)
        route_type = query_analysis.route_type
        rewritten_query = query_analysis.rewritten_query
        print(f"🎯 查询类型: {route_type}")
        if rewritten_query == question:
            print(f"📝 查询保持原样: {question}")
        else:
            print(f"📝 检索查询: {rewritten_query}")
        
        # 2. 检索相关文档块（自动应用元数据过滤）
        print("🔍 检索相关文档块...")
        retrieval_start = time.perf_counter()
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            retrieval_strategy = "metadata_filtered"
            # 应用元数据过滤
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=self.config.top_k)
        else:
            retrieval_strategy = "hybrid"
            # 无元数据过滤，使用混合检索
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)
        timings_ms["retrieval"] = self._elapsed_ms(retrieval_start)

        # 显示检索到的文档块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                doc_title = chunk.metadata.get('doc_title', '未知文档')
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    # 如果是标题开头，提取标题（仅取第一行）
                    title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace('#', '').strip() # 移除标题前的#符号并去空格
                    chunk_info.append(f"{doc_title}({section_title})")
                else:
                    chunk_info.append(f"{doc_title}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 结构化来源需要等父文档确定后再构建，确保 source_id 与答案引用编号对齐
        sources = []
        relevant_context_docs = []

        # 5. 检查是否找到相关内容
        if not relevant_chunks:
            answer = "抱歉，没有找到相关的校园文档信息。请尝试其他标题或关键词。"
            timings_ms["total"] = self._elapsed_ms(total_start)
            trace = self._build_trace(
                retrieval_strategy,
                filters,
                timings_ms,
                relevant_chunks,
                relevant_context_docs,
                sources,
            ) if return_trace else None
            if return_sources or return_trace:
                return self._build_rag_response(question, route_type, rewritten_query, answer, sources, trace)
            return answer

        # 6. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回文档标题列表
            print("📋 生成文档列表...")
            # 基于召回块构建证据上下文，避免把完整父文档直接塞进生成阶段
            context_start = time.perf_counter()
            relevant_context_docs = self.data_module.get_context_documents(
                relevant_chunks,
                window_size=self.config.context_window_size,
            )
            timings_ms["context_build"] = self._elapsed_ms(context_start)

            # 显示找到的上下文文档名称
            context_doc_names = []
            for context_doc in relevant_context_docs:
                doc_title = context_doc.metadata.get('doc_title', '未知文档')
                context_doc_names.append(doc_title)

            if context_doc_names:
                print(f"找到证据上下文: {', '.join(context_doc_names)}")

            sources = self._build_aligned_sources(relevant_context_docs, relevant_chunks) if (return_sources or return_trace) else []
            generation_start = time.perf_counter()
            answer = self.generation_module.generate_list_answer(question, relevant_context_docs)
            timings_ms["generation"] = self._elapsed_ms(generation_start)
            timings_ms["total"] = self._elapsed_ms(total_start)
            trace = self._build_trace(
                retrieval_strategy,
                filters,
                timings_ms,
                relevant_chunks,
                relevant_context_docs,
                sources,
            ) if return_trace else None
            if return_sources or return_trace:
                return self._build_rag_response(
                    question,
                    route_type,
                    rewritten_query,
                    answer,
                    sources if return_sources else [],
                    trace,
                )
            return answer
        else:
            # 详细查询：获取证据上下文并生成详细回答
            print("获取证据上下文...")
            context_start = time.perf_counter()
            relevant_context_docs = self.data_module.get_context_documents(
                relevant_chunks,
                window_size=self.config.context_window_size,
            )
            timings_ms["context_build"] = self._elapsed_ms(context_start)

            # 显示找到的上下文文档名称
            context_doc_names = []
            for context_doc in relevant_context_docs:
                doc_title = context_doc.metadata.get('doc_title', '未知文档')
                context_doc_names.append(doc_title)

            if context_doc_names:
                print(f"找到证据上下文: {', '.join(context_doc_names)}")
            else:
                print(f"对应 {len(relevant_context_docs)} 个证据上下文")

            sources = self._build_aligned_sources(relevant_context_docs, relevant_chunks) if (return_sources or return_trace) else []
            print("✍️ 生成详细回答...")

            # 根据路由类型自动选择回答模式
            generation_start = time.perf_counter()
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_context_docs)
                answer = self.generation_module.generate_step_by_step_answer(question, relevant_context_docs)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_context_docs)
                answer = self.generation_module.generate_basic_answer(question, relevant_context_docs)
            timings_ms["generation"] = self._elapsed_ms(generation_start)
            timings_ms["total"] = self._elapsed_ms(total_start)

            trace = self._build_trace(
                retrieval_strategy,
                filters,
                timings_ms,
                relevant_chunks,
                relevant_context_docs,
                sources,
            ) if return_trace else None
            if return_sources or return_trace:
                return self._build_rag_response(
                    question,
                    route_type,
                    rewritten_query,
                    answer,
                    sources if return_sources else [],
                    trace,
                )
            return answer

    def _build_aligned_sources(self, parent_docs: List, retrieved_chunks: List) -> List[RetrievedSource]:
        """
        将召回文档块转换为与答案引用编号对齐的结构化来源
        Args:
            parent_docs: 生成回答使用的父文档列表
            retrieved_chunks: 检索召回的文档块列表
        Returns:
            结构化来源列表 同一父文档的多个文档块会共享同一个 source_id
        """
        source_id_by_parent_id = {}
        parent_metadata_by_parent_id = {}

        for source_id, parent_doc in enumerate(parent_docs, 1):
            metadata = parent_doc.metadata or {}
            parent_id = metadata.get("parent_id")
            if parent_id:
                source_id_by_parent_id[parent_id] = source_id
                parent_metadata_by_parent_id[parent_id] = metadata

        sources = []
        for chunk in retrieved_chunks:
            chunk_metadata = chunk.metadata or {}
            parent_id = chunk_metadata.get("parent_id")
            source_id = source_id_by_parent_id.get(parent_id)

            if source_id is None:
                continue

            parent_metadata = parent_metadata_by_parent_id.get(parent_id, chunk_metadata)
            page_value = parent_metadata.get("page")
            if page_value is None:
                page_value = chunk_metadata.get("page")
            sources.append(
                RetrievedSource(
                    source_id=source_id,
                    doc_title=parent_metadata.get("doc_title", "未知文档"),
                    doc_category=parent_metadata.get("doc_category", "未知"),
                    department=parent_metadata.get("department", ""),
                    file_type=parent_metadata.get("file_type", ""),
                    section=self._extract_section_title(chunk),
                    source=parent_metadata.get("source", ""),
                    page=self._safe_int(page_value, None) if page_value is not None else None,
                    chunk_index=self._safe_int(chunk_metadata.get("chunk_index"), source_id - 1),
                    rrf_score=self._safe_float(chunk_metadata.get("rrf_score")),
                    snippet=self._build_snippet(chunk.page_content),
                )
            )
        return sources

    def _build_trace(
        self,
        retrieval_strategy: str,
        filters: Dict[str, Any],
        timings_ms: Dict[str, float],
        retrieved_chunks: List,
        context_docs: List,
        sources: List[RetrievedSource],
    ) -> RAGTrace:
        """构造 RAG 查询链路调试信息"""
        return RAGTrace(
            retrieval_strategy=retrieval_strategy,
            filters=dict(filters or {}),
            timings_ms={key: round(value, 2) for key, value in timings_ms.items()},
            retrieval_params={
                "top_k": self.config.top_k,
                "candidate_k": self.config.retrieval_candidate_k,
                "rrf_k": self.config.rrf_k,
                "context_window_size": self.config.context_window_size,
            },
            retrieved_chunks=[
                self._build_retrieved_chunk_trace(rank, chunk)
                for rank, chunk in enumerate(retrieved_chunks, 1)
            ],
            context_documents=[
                self._build_context_document_trace(source_id, context_doc)
                for source_id, context_doc in enumerate(context_docs, 1)
            ],
            source_count=len(sources),
        )

    def _build_retrieved_chunk_trace(self, rank: int, chunk) -> Dict[str, Any]:
        """构造单个召回块的调试摘要"""
        metadata = chunk.metadata or {}
        return {
            "rank": rank,
            "doc_title": metadata.get("doc_title", "未知文档"),
            "section": self._extract_section_title(chunk),
            "chunk_index": self._safe_int(metadata.get("chunk_index"), None),
            "rrf_score": self._safe_float(metadata.get("rrf_score")),
        }

    def _build_context_document_trace(self, source_id: int, context_doc) -> Dict[str, Any]:
        """构造单个证据上下文文档的调试摘要"""
        metadata = context_doc.metadata or {}
        return {
            "source_id": source_id,
            "doc_title": metadata.get("doc_title", "未知文档"),
            "context_window_size": self._safe_int(
                metadata.get("context_window_size"),
                self.config.context_window_size,
            ),
            "context_chunk_indices": list(metadata.get("context_chunk_indices") or []),
        }

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        """计算从 start_time 到当前的毫秒耗时"""
        return (time.perf_counter() - start_time) * 1000

    @staticmethod
    def _build_rag_response(
        question: str,
        route_type: str,
        rewritten_query: str,
        answer: str,
        sources: List[RetrievedSource],
        trace: RAGTrace = None,
    ) -> RAGResponse:
        """创建结构化RAG响应"""
        return RAGResponse(
            question=question,
            route_type=route_type,
            rewritten_query=rewritten_query,
            answer=answer,
            sources=sources,
            trace=trace,
        )

    @staticmethod
    def _extract_section_title(chunk) -> str:
        """从文档块元数据或正文标题中提取章节名"""
        metadata = chunk.metadata or {}
        if metadata.get("section"):
            return str(metadata["section"])
        for key in ("三级标题", "二级标题", "主标题"):
            if metadata.get(key):
                return str(metadata[key])
        # 尝试从正文提取标题
        content_preview = chunk.page_content.strip()
        # 检查是否以标题开头
        if content_preview.startswith("#"):
            first_line = content_preview.splitlines()[0]
            title = first_line.replace("#", "").strip()
            if title:
                return title

        return "内容片段"

    @staticmethod
    def _build_snippet(content: str, max_length: int = 200) -> str:
        """生成适合展示的召回片段摘要"""
        snippet = " ".join(content.strip().split())
        if len(snippet) <= max_length:
            return snippet
        return snippet[: max_length - 3].rstrip() + "..."

    @staticmethod
    def _safe_int(value, default: int) -> int:
        """安全转换整数，避免元数据缺失或格式异常影响响应生成"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value):
        """安全转换浮点数，缺失时返回None"""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        Args:
            query: 用户查询
        Returns:
            过滤条件字典
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['doc_category'] = cat
                break

        return filters
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("📚  校园知识库RAG系统 - 交互式问答  📚")
        print("=" * 60)
        print("💡 方便查规章、找流程、看通知。")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if not user_input:
                    print("请输入问题内容，或输入'退出'结束。")
                    continue
                if user_input.lower() in ['退出', 'quit', 'exit']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (是/否, 默认是): ").strip().lower()
                use_stream = stream_choice != '否'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for text_chunk in self.ask_question(user_input, stream=True):
                        print(text_chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                print("\n输入流已结束，退出交互式问答。")
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用校园知识库RAG系统！")

