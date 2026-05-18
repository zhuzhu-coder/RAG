"""
校园文档分块配置
"""

from typing import Any, Dict, List

# 通用中文文本递归切分分隔符
TEXT_SEPARATORS: List[str] = ["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""]

# Markdown 标题层级配置，使用 JSON 友好的列表，便于写入缓存校验 manifest
MARKDOWN_HEADERS_TO_SPLIT_ON: List[List[str]] = [["#", "h1"], ["##", "h2"], ["###", "h3"]]

# 数据准备和索引缓存校验 manifest 共用的分块配置
CHUNKING_CONFIG: Dict[str, Dict[str, Any]] = {
    "md": {
        "splitter": "MarkdownHeaderTextSplitter",  # 按 Markdown 标题切分
        "strip_headers": False,  # 保留标题内容
        "headers_to_split_on": MARKDOWN_HEADERS_TO_SPLIT_ON,  # 切分标题层级
        # 通用分块配置
        "fallback": {
            "splitter": "RecursiveCharacterTextSplitter",  # 按字符长度和分隔符切分
            "chunk_size": 800,  # 每个分块最大字符数
            "chunk_overlap": 120,  # 分块重叠字符数
            "separators": TEXT_SEPARATORS,  # 分隔符列表
        },
    },
    "txt": {
        "splitter": "RecursiveCharacterTextSplitter",  # 按字符长度和分隔符切分
        "chunk_size": 800,  # 每个分块最大字符数
        "chunk_overlap": 120,  # 分块重叠字符数
        "separators": TEXT_SEPARATORS,  # 分隔符列表
    },
    "pdf": {
        "splitter": "RecursiveCharacterTextSplitter",  # 按字符长度和分隔符切分
        "chunk_size": 1000,  # 每个分块最大字符数
        "chunk_overlap": 150,  # 分块重叠字符数
        "separators": TEXT_SEPARATORS,  # 分隔符列表
    },
}
