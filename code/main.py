"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 将当前文件目录加入模块搜索路径，方便导入其他模块
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import PROJECT_ROOT, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
    RAGResponse,
    RetrievedSource,
)

# 拼接路径，加载环境变量
load_dotenv(PROJECT_ROOT / ".env")
# 兜底加载
load_dotenv()

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

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统
        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
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

        # 检查数据路径是否存在
        if not Path(self.config.data_path).exists():
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

        # 1. 用源文件指纹和索引配置判断本地向量索引是否仍然可用
        print("检查向量索引缓存...")
        expected_manifest = self.index_module.build_manifest(self.config.data_path)
        vectorstore = self.index_module.load_index(expected_manifest)

        # 2. 无论向量索引是否命中缓存，BM25 和父文档回答都需要当前文档与分块
        print("加载食谱文档...")
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

            # 4. 保存索引与 manifest
            print("保存向量索引...")
            self.index_module.save_index()
            self.index_module.save_manifest(
                self.index_module.build_manifest(self.config.data_path, chunks)
            )

        # 5. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore,
            chunks,
            candidate_k=self.config.retrieval_candidate_k,
        )

        # 6. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")
    
    def ask_question(self, question: str, stream: bool = False, return_sources: bool = False):
        """
        回答用户问题
        Args:
            question: 用户问题
            stream: 是否使用流式输出
            return_sources: 是否返回结构化回答和检索来源
        Returns:
            生成的回答、生成器或结构化RAG响应
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        if stream and return_sources:
            raise ValueError("结构化来源返回暂不支持流式输出")
        
        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            # 详细查询和一般查询使用智能重写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)
        
        # 3. 检索相关文档块（自动应用元数据过滤）
        print("🔍 检索相关文档块...")
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            # 应用元数据过滤
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=self.config.top_k)
        else:
            # 无元数据过滤，使用混合检索
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        # 显示检索到的文档块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    # 如果是标题开头，提取标题（仅取第一行）
                    title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace('#', '').strip() # 移除标题前的#符号并去空格
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 结构化来源需要等父文档确定后再构建，确保 source_id 与答案引用编号对齐
        sources = []

        # 5. 检查是否找到相关内容
        if not relevant_chunks:
            answer = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
            if return_sources:
                return self._build_rag_response(question, route_type, rewritten_query, answer, sources)
            return answer

        # 6. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回菜品名称列表
            print("📋 生成菜品列表...")
            # 从检索到的文档块中映射出完整父文档
            relevant_parent_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的完整菜谱名称
            parent_doc_names = []
            for parent_doc in relevant_parent_docs:
                dish_name = parent_doc.metadata.get('dish_name', '未知菜品')
                parent_doc_names.append(dish_name)

            if parent_doc_names:
                print(f"找到完整菜谱: {', '.join(parent_doc_names)}")

            sources = self._build_aligned_sources(relevant_parent_docs, relevant_chunks) if return_sources else []
            answer = self.generation_module.generate_list_answer(question, relevant_parent_docs)
            if return_sources:
                return self._build_rag_response(question, route_type, rewritten_query, answer, sources)
            return answer
        else:
            # 详细查询：获取完整父文档并生成详细回答
            print("获取完整父文档...")
            relevant_parent_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的完整菜谱名称
            parent_doc_names = []
            for parent_doc in relevant_parent_docs:
                dish_name = parent_doc.metadata.get('dish_name', '未知菜品')
                parent_doc_names.append(dish_name)

            if parent_doc_names:
                print(f"找到完整菜谱: {', '.join(parent_doc_names)}")
            else:
                print(f"对应 {len(relevant_parent_docs)} 个完整父文档")

            sources = self._build_aligned_sources(relevant_parent_docs, relevant_chunks) if return_sources else []
            print("✍️ 生成详细回答...")

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_parent_docs)
                answer = self.generation_module.generate_step_by_step_answer(question, relevant_parent_docs)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_parent_docs)
                answer = self.generation_module.generate_basic_answer(question, relevant_parent_docs)

            if return_sources:
                return self._build_rag_response(question, route_type, rewritten_query, answer, sources)
            return answer

    def _build_aligned_sources(self, parent_docs: List, retrieved_chunks: List) -> List[RetrievedSource]:
        """
        将召回文档块转换为与答案引用编号对齐的结构化来源
        Args:
            parent_docs: 生成回答使用的父文档列表
            retrieved_chunks: 检索召回的文档块列表
        Returns:
            结构化来源列表。同一父文档的多个文档块会共享同一个 source_id
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
            sources.append(
                RetrievedSource(
                    source_id=source_id,
                    dish_name=parent_metadata.get("dish_name", "未知菜品"),
                    category=parent_metadata.get("category", "未知"),
                    difficulty=parent_metadata.get("difficulty", "未知"),
                    section=self._extract_section_title(chunk),
                    source=parent_metadata.get("source", ""),
                    chunk_index=self._safe_int(chunk_metadata.get("chunk_index"), source_id - 1),
                    rrf_score=self._safe_float(chunk_metadata.get("rrf_score")),
                    snippet=self._build_snippet(chunk.page_content),
                )
            )
        return sources

    @staticmethod
    def _build_rag_response(
        question: str,
        route_type: str,
        rewritten_query: str,
        answer: str,
        sources: List[RetrievedSource],
    ) -> RAGResponse:
        """创建结构化RAG响应"""
        return RAGResponse(
            question=question,
            route_type=route_type,
            rewritten_query=rewritten_query,
            answer=answer,
            sources=sources,
        )

    @staticmethod
    def _extract_section_title(chunk) -> str:
        """从文档块元数据或正文标题中提取章节名"""
        metadata = chunk.metadata or {}
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
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        # 按难度关键词长度排序，避免先匹配到较短的关键词
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
    
    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息
        Args:
            dish_name: 菜品名称
        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 检索相关文档块，并映射到完整父文档
        retrieved_chunks = self.retrieval_module.hybrid_search(dish_name, top_k=3)
        parent_docs = self.data_module.get_parent_documents(retrieved_chunks)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", parent_docs)

        return answer
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
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
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")



def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()


