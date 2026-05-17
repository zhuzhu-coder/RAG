"""
生成集成模块
"""

import os
import logging
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryAnalysis:
    """一次 LLM 查询分析的结构化结果"""

    route_type: str
    rewritten_query: str


class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "qwen3.6-plus", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        # 初始化模型名称
        self.model_name = model_name
        # 初始化生成温度
        self.temperature = temperature
        # 初始化最大token数
        self.max_tokens = max_tokens
        # 初始化LLM模型
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """初始化LLM模型"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
        logger.info("LLM初始化完成")
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        合并查询路由和查询重写，减少一次 LLM 调用
        Args:
            query: 用户查询
        Returns:
            查询类型与检索用改写问题
        """
        prompt = ChatPromptTemplate.from_template("""
你是一个校园知识库 RAG 查询分析助手。请一次性完成：
1. 判断用户问题类型；
2. 生成更适合检索的查询。

问题类型只能是以下三种之一：
- list：用户想要获取文档列表、通知列表或推荐，只需要文档标题。
- detail：用户想要具体办理方法、流程、条件、材料、时间、处罚或后果。
- general：其他一般解释性问题。

改写规则：
- 必须保持原意，不要扩写成用户没有问到的内容。
- 必须保留用户原始问题中的核心事件词、对象词和行为词。
- 如果问题中出现“晚归”“校园卡”“缓考”“补考”“图书馆闭馆”“网络维护”“宿舍”“考试证件”等词，改写后必须保留。
- list 类型问题不要改写，rewritten_query 直接返回原问题。
- detail/general 类型可以补足缺失的对象、流程或条件，但保持简洁、通顺。

请只返回 JSON，不要添加解释、Markdown 或多余文本：
{{"route_type":"detail","rewritten_query":"学生晚归处理规定"}}

用户问题: {query}
""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        analysis = self._parse_query_analysis(response, query)

        if analysis.rewritten_query != query:
            logger.info("查询已重写: '%s' → '%s'", query, analysis.rewritten_query)
        else:
            logger.info("查询无需重写: '%s'", query)
        return analysis

    @classmethod
    def _parse_query_analysis(cls, response: str, original_query: str) -> QueryAnalysis:
        """
        解析合并查询分析的 JSON 输出，失败时回退到原问题
        Args:
            response: 模型原始输出
            original_query: 用户原始问题
        Returns:
            规范化后的 QueryAnalysis
        """
        payload = cls._extract_json_payload(response)
        if payload is None:
            return QueryAnalysis(route_type="general", rewritten_query=original_query)

        route_type = cls._normalize_route_type(str(payload.get("route_type", "")))
        rewritten_query = str(payload.get("rewritten_query") or "").strip()
        if not rewritten_query or route_type == "list":
            rewritten_query = original_query

        return QueryAnalysis(route_type=route_type, rewritten_query=rewritten_query)

    @staticmethod
    def _extract_json_payload(response: str) -> Optional[Dict[str, Any]]:
        """从模型输出中提取 JSON 对象"""
        text = (response or "").strip()
        if not text:
            return None

        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not object_match:
            return None

        try:
            payload = json.loads(object_match.group(0))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _normalize_route_type(result: str) -> str:
        """
        从模型输出中提取稳定的路由类型
        Args:
            result: 模型原始输出
        Returns:
            list / detail / general
        """
        # 规范化输入，转换为小写并移除首尾空格
        normalized = (result or "").strip().lower()
        # 检查是否包含路由类型
        for route_type in ("list", "detail", "general"):
            if re.search(rf"(?<![a-z]){route_type}(?![a-z])", normalized):
                return route_type
        return "general"

    def _build_context(self, parent_docs: List[Document], max_length: int = 2000) -> str:
        """
        构建上下文字符串
        Args:
            parent_docs: 用于生成回答的完整父文档列表
            max_length: 最大长度
        Returns:
            格式化的上下文字符串
        """
        if not parent_docs:
            return "暂无相关校园文档信息。"

        separator = "\n" + "=" * 50 + "\n"

        context_parts = [] # 保存每个父文档格式化后的文本
        current_length = len(separator) # 已占用的字符长度
        # 遍历每个父文档，格式化并添加到上下文
        for i, parent_doc in enumerate(parent_docs, 1):
            doc_text = self._format_context_doc(i, parent_doc, include_optional_metadata=True)

            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                # 尝试不包含可选元数据的格式化
                compact_doc_text = self._format_context_doc(i, parent_doc, include_optional_metadata=False)
                if len(compact_doc_text) < len(doc_text):
                    doc_text = compact_doc_text
                remaining_length = max_length - current_length
                # 判断是否还有剩余长度
                if remaining_length > 0:
                    # 截断文档文本，确保不超过最大长度
                    suffix = "..." if remaining_length > 3 else ""
                    truncated_text = doc_text[:remaining_length - len(suffix)].rstrip()
                    context_parts.append(truncated_text + suffix)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return (separator + "\n".join(context_parts))[:max_length]

    def _format_context_doc(
        self,
        source_id: int,
        parent_doc: Document,
        include_optional_metadata: bool = True,
    ) -> str:
        """
        将单个完整父文档格式化为带引用编号的上下文文本
        Args:
            source_id: 来源ID
            parent_doc: 完整父文档
            include_optional_metadata: 是否包含可选元数据
        Returns:
            格式化的文档字符串
        """
        metadata = parent_doc.metadata or {}
        doc_title = GenerationIntegrationModule._get_doc_title(metadata)
        metadata_info = f"[{source_id}] 校园文档: {doc_title}"

        if include_optional_metadata:
            if metadata.get("doc_category"):
                metadata_info += f" | 分类: {metadata['doc_category']}"
            if metadata.get("department"):
                metadata_info += f" | 部门: {metadata['department']}"
            if metadata.get("file_type"):
                metadata_info += f" | 类型: {metadata['file_type']}"
        if metadata.get("source"):
            metadata_info += f"\n来源: {metadata['source']}"

        return f"{metadata_info}\n内容:\n{parent_doc.page_content}\n"

    @staticmethod
    def _grounded_answer_rules() -> str:
        """生成回答时使用的检索约束和引用规则。"""
        return """
回答规则：
1. 只能基于“相关校园文档信息”回答，不要使用未检索到的外部知识。
2. 不要编造制度、流程、日期、地点、材料或来源。
3. 如果相关校园文档信息不足以回答问题，明确说明“当前资料不足”，不要猜测。
4. 每个关键要求、步骤、提示或结论后面必须标注来源编号，例如 [1]。
5. 参考来源列表会由系统自动追加，不要自行编写参考来源小节。
"""

    @staticmethod
    def _build_reference_lines(parent_docs: List[Document]) -> List[str]:
        """
        根据上下文父文档生成参考来源行
        Args:
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            参考来源行列表
        """
        reference_lines = []
        seen_titles = set()
        for source_id, parent_doc in enumerate(parent_docs, 1):
            metadata = parent_doc.metadata or {}
            doc_title = GenerationIntegrationModule._get_doc_title(metadata)
            if doc_title in seen_titles:
                continue
            line = f"[{source_id}] {doc_title}"
            if metadata.get("source"):
                line += f" - {metadata['source']}"
            reference_lines.append(line)
            seen_titles.add(doc_title)
        return reference_lines

    def _append_reference_lines(self, answer: str, parent_docs: List[Document]) -> str:
        """
        在答案末尾追加稳定的参考来源列表
        Args:
            answer: 模型生成的答案
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            追加参考来源后的答案
        """
        if "参考来源" in answer:
            return answer

        reference_lines = self._build_reference_lines(parent_docs)
        if not reference_lines:
            return answer

        return answer.rstrip() + "\n\n参考来源:\n" + "\n".join(reference_lines)

    def _stream_with_reference_lines(
        self,
        text_chunks: Iterable[str],
        parent_docs: List[Document],
    ) -> Iterator[str]:
        """
        流式输出模型文本，并在末尾按需追加稳定的参考来源列表
        Args:
            text_chunks: 模型流式生成的文本片段
            parent_docs: 用于生成回答的完整父文档列表
        Yields:
            模型文本片段，以及可能追加的参考来源片段
        """
        generated_parts = []
        for text_chunk in text_chunks:
            generated_parts.append(text_chunk)
            yield text_chunk

        generated_text = "".join(generated_parts)
        if "参考来源" in generated_text:
            return

        reference_lines = self._build_reference_lines(parent_docs)
        if reference_lines:
            yield "\n\n参考来源:\n" + "\n".join(reference_lines)

    def generate_list_answer(self, query: str, parent_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            列表式回答
        """
        if not parent_docs:
            return "抱歉，没有找到相关的校园文档。"
        # 构造文档引用列表
        doc_refs = []
        seen_doc_titles = set()
        for source_id, parent_doc in enumerate(parent_docs, 1):
            doc_title = self._get_doc_title(parent_doc.metadata or {})
            if doc_title not in seen_doc_titles:
                doc_refs.append((doc_title, source_id))
                seen_doc_titles.add(doc_title)

        # 构建简洁的列表回答
        if len(doc_refs) == 1:
            answer = f"为您推荐：{doc_refs[0][0]} [{doc_refs[0][1]}]"
        elif len(doc_refs) <= 3:
            answer = "为您推荐以下文档：\n" + "\n".join(
                [f"{i+1}. {name} [{source_id}]" for i, (name, source_id) in enumerate(doc_refs)]
            )
        else:
            answer = "为您推荐以下文档：\n" + "\n".join(
                [f"{i+1}. {name} [{source_id}]" for i, (name, source_id) in enumerate(doc_refs[:3])]
            )
            answer += f"\n\n还有其他 {len(doc_refs)-3} 份文档可供选择。"
        # 追加参考来源
        reference_lines = self._build_reference_lines(parent_docs)
        if reference_lines:
            answer += "\n\n参考来源:\n" + "\n".join(reference_lines)
        return answer

    def generate_basic_answer(self, query: str, parent_docs: List[Document]) -> str:
        """
        生成基础回答
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            生成的回答
        """
        context = self._build_context(parent_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的校园知识库助手。请根据以下校园文档信息回答用户的问题。

用户问题: {question}

相关校园文档信息:
{context}

""" + self._grounded_answer_rules() + """

请在遵守以上规则的前提下，提供详细、实用的回答。

回答:""")

        # 构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context} # 可运行的映射规则
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return self._append_reference_lines(response, parent_docs)

    def generate_step_by_step_answer(self, query: str, parent_docs: List[Document]) -> str:
        """
        生成分步骤详细回答
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            分步骤的详细回答
        """
        context = self._build_context(parent_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的校园事务助手。请根据校园文档信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关校园文档信息:
{context}

""" + self._grounded_answer_rules() + """

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 文档概览
[简要介绍文档主题和适用范围]

## 关键要求
[列出主要条件、要求或材料]

## 办理步骤
[详细的分步骤说明，每步包含具体操作和注意事项]

## 注意事项
[仅在有实用提醒时包含。优先使用原文中的要求与提示，必要时可以基于正文总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的注意事项要分享，可以省略该部分

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return self._append_reference_lines(response, parent_docs)


    def generate_basic_answer_stream(self, query: str, parent_docs: List[Document]):
        """
        生成基础回答 - 流式输出
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Yields:
            生成的回答片段
        """
        context = self._build_context(parent_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的校园知识库助手。请根据以下校园文档信息回答用户的问题。

用户问题: {question}

相关校园文档信息:
{context}

""" + self._grounded_answer_rules() + """

请在遵守以上规则的前提下，提供详细、实用的回答。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        yield from self._stream_with_reference_lines(chain.stream(query), parent_docs)

    @staticmethod
    def _get_doc_title(metadata: dict) -> str:
        """从元数据中提取稳定的文档标题"""
        return (
            metadata.get("doc_title")
            or metadata.get("source_name")
            or "未知文档"
        )

    def generate_step_by_step_answer_stream(self, query: str, parent_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Yields:
            详细步骤回答片段
        """
        context = self._build_context(parent_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的校园事务助手。请根据校园文档信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关校园文档信息:
{context}

""" + self._grounded_answer_rules() + """

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 文档概览
[简要介绍文档主题和适用范围]

## 关键要求
[列出主要条件、要求或材料]

## 办理步骤
[详细的分步骤说明，每步包含具体操作和注意事项]

## 注意事项
[仅在有实用提醒时包含。如果原文内容与办理无关或为空，可以基于正文总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        yield from self._stream_with_reference_lines(chain.stream(query), parent_docs)
