"""
生成集成模块
"""

import os
import logging
import re
from typing import Iterable, Iterator, List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
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
    
    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式
        Args:
            query: 用户查询
        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query)
        # 规范化路由类型
        return self._normalize_route_type(result)

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

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询
        Args:
            query: 原始查询
        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate.from_template("""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

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
            return "暂无相关食谱信息。"

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
        dish_name = metadata.get("dish_name", "未知菜品")
        metadata_info = f"[{source_id}] 菜谱: {dish_name}"

        if include_optional_metadata:
            if "category" in metadata:
                metadata_info += f" | 分类: {metadata['category']}"
            if "difficulty" in metadata:
                metadata_info += f" | 难度: {metadata['difficulty']}"
        if metadata.get("source"):
            metadata_info += f"\n来源: {metadata['source']}"

        return f"{metadata_info}\n内容:\n{parent_doc.page_content}\n"

    @staticmethod
    def _grounded_answer_rules() -> str:
        """生成回答时使用的检索约束和引用规则。"""
        return """
回答规则：
1. 只能基于“相关食谱信息”回答，不要使用未检索到的外部知识。
2. 不要编造食材、用量、步骤、时间、功效或来源。
3. 如果相关食谱信息不足以回答问题，明确说明“当前资料不足”，不要猜测。
4. 每个关键食材、步骤、技巧或结论后面必须标注来源编号，例如 [1]。
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
        seen_dishes = set()
        for source_id, parent_doc in enumerate(parent_docs, 1):
            metadata = parent_doc.metadata or {}
            dish_name = metadata.get("dish_name", "未知菜品")
            if dish_name in seen_dishes:
                continue
            line = f"[{source_id}] {dish_name}"
            if metadata.get("source"):
                line += f" - {metadata['source']}"
            reference_lines.append(line)
            seen_dishes.add(dish_name)
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
            return "抱歉，没有找到相关的菜品信息。"

        # 提取菜品名称和来源编号
        dish_refs = []
        seen_dish_names = set()
        for source_id, parent_doc in enumerate(parent_docs, 1):
            dish_name = parent_doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in seen_dish_names:
                dish_refs.append((dish_name, source_id))
                seen_dish_names.add(dish_name)

        # 构建简洁的列表回答
        if len(dish_refs) == 1:
            answer = f"为您推荐：{dish_refs[0][0]} [{dish_refs[0][1]}]"
        elif len(dish_refs) <= 3:
            answer = "为您推荐以下菜品：\n" + "\n".join(
                [f"{i+1}. {name} [{source_id}]" for i, (name, source_id) in enumerate(dish_refs)]
            )
        else:
            answer = "为您推荐以下菜品：\n" + "\n".join(
                [f"{i+1}. {name} [{source_id}]" for i, (name, source_id) in enumerate(dish_refs[:3])]
            )
            answer += f"\n\n还有其他 {len(dish_refs)-3} 道菜品可供选择。"

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
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
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
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

""" + self._grounded_answer_rules() + """

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分

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
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
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
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

""" + self._grounded_answer_rules() + """

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

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
