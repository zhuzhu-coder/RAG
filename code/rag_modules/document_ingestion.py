"""
校园文档接入层
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
CATEGORY_MAPPING = {
    "regulations": "规章制度",
    "teaching": "教务教学",
    "life": "校园生活",
    "notices": "通知公告",
}
FALLBACK_CATEGORY = "其他"
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def load_documents(data_path: str | Path) -> List[Document]:
    """递归加载校园语料中的支持文件。"""
    root = Path(data_path)
    if not root.exists():
        logger.warning("文档目录不存在: %s", root)
        return []

    documents: List[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix == ".md":
            doc = load_markdown_document(path, root)
            if doc is not None:
                documents.append(doc)
        elif suffix == ".txt":
            doc = load_text_document(path, root)
            if doc is not None:
                documents.append(doc)
        elif suffix == ".pdf":
            documents.extend(load_pdf_document(path, root))

    return documents


def load_markdown_document(path: Path, data_root: Optional[Path] = None) -> Optional[Document]:
    """读取 Markdown 文档并生成父文档对象。"""
    text = _read_text_file(path)
    if not text:
        logger.warning("Markdown 文档为空: %s", path)
        return None

    normalized_text = normalize_text(text)
    if not normalized_text:
        logger.warning("Markdown 文档清洗后为空: %s", path)
        return None

    return Document(
        page_content=normalized_text,
        metadata=build_base_metadata(path, normalized_text, "md", data_root=data_root),
    )


def load_text_document(path: Path, data_root: Optional[Path] = None) -> Optional[Document]:
    """读取 TXT 文档并生成父文档对象。"""
    text = _read_text_file(path)
    if not text:
        logger.warning("TXT 文档为空: %s", path)
        return None

    normalized_text = normalize_text(text)
    if not normalized_text:
        logger.warning("TXT 文档清洗后为空: %s", path)
        return None

    return Document(
        page_content=normalized_text,
        metadata=build_base_metadata(path, normalized_text, "txt", data_root=data_root),
    )


def load_pdf_document(path: Path, data_root: Optional[Path] = None) -> List[Document]:
    """按页读取可直接抽取文本的 PDF。"""
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pragma: no cover - 解析失败属于外部输入问题
        logger.warning("PDF 读取失败: %s, error=%s", path, exc)
        return []

    documents: List[Document] = []
    for page_number, page in enumerate(reader.pages, 1):
        try:
            text = normalize_text(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - 解析失败属于外部输入问题
            logger.warning("PDF 页面抽取失败: %s, page=%s, error=%s", path, page_number, exc)
            continue

        if not text:
            logger.info("PDF 页面无可抽取文本，已跳过: %s, page=%s", path, page_number)
            continue

        metadata = build_base_metadata(path, text, "pdf", page=page_number, data_root=data_root)
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        logger.warning("PDF 没有可抽取文本，已标记为不支持: %s", path)

    return documents


def build_base_metadata(
    path: Path,
    text: str,
    file_type: str,
    page: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> dict:
    """构建通用文档元数据。"""
    normalized_path = path.resolve()
    relative_path = _relative_path(normalized_path, data_root)
    doc_title = infer_doc_title(path, text, file_type)
    doc_category = infer_doc_category(path, text)
    department = infer_department(text)
    doc_id_seed = f"{relative_path}#{page}" if page is not None else relative_path
    doc_id = hashlib.md5(doc_id_seed.encode("utf-8")).hexdigest()

    return {
        "source": str(path),
        "source_name": path.name,
        "relative_path": relative_path,
        "file_type": file_type,
        "doc_title": doc_title,
        "doc_category": doc_category,
        "department": department,
        "doc_id": doc_id,
        "parent_id": doc_id,
        "doc_type": "parent",
        "section": doc_title,
        "page": page,
        "content_length": len(text),
        "line_count": len(text.splitlines()),
        "file_size": path.stat().st_size if path.exists() else 0,
    }


def infer_doc_title(path: Path, text: str, file_type: str) -> str:
    """从正文或文件名推断标题。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return path.stem or "未知文档"

    if file_type == "md":
        for line in lines[:10]:
            if line.startswith("#"):
                candidate = _clean_title(line)
                if candidate:
                    return candidate

    for line in lines[:5]:
        candidate = _clean_title(line)
        if _looks_like_title(candidate):
            return candidate

    return path.stem or lines[0] or "未知文档"


def infer_doc_category(path: Path, text: str) -> str:
    """根据路径和内容推断校园文档分类。"""
    path_parts = {part.lower() for part in path.parts}
    for key, value in CATEGORY_MAPPING.items():
        if key in path_parts:
            return value

    lowered_text = text.lower()
    if "教务" in text or "教学" in text or "考试" in text:
        return "教务教学"
    if "宿舍" in text or "校园卡" in text or "生活" in text:
        return "校园生活"
    if "通知" in text or "公告" in text:
        return "通知公告"

    if any(key in lowered_text for key in ("academic", "exam", "student")):
        return "教务教学"

    return FALLBACK_CATEGORY


def infer_department(text: str) -> str:
    """尝试从正文中提取发布部门。"""
    department_patterns = ("教务处", "学生处", "后勤处", "图书馆", "保卫处", "学院", "综合服务大厅")
    for pattern in department_patterns:
        if pattern in text:
            return pattern
    return ""


def normalize_text(text: str) -> str:
    """统一换行、空白和空行。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")

    lines: List[str] = []
    blank_line_emitted = False
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).rstrip()
        if line.strip():
            lines.append(line.strip())
            blank_line_emitted = False
        elif not blank_line_emitted:
            lines.append("")
            blank_line_emitted = True

    return "\n".join(lines).strip()


def _read_text_file(path: Path) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # pragma: no cover - 外部文件问题
            logger.warning("读取文本文件失败: %s, error=%s", path, exc)
            return ""

    logger.warning("文本文件编码无法识别: %s", path)
    return ""


def _clean_title(text: str) -> str:
    cleaned = re.sub(r"^[#\-\*\s]+", "", text).strip()
    cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned).strip()
    return cleaned


def _looks_like_title(text: str) -> bool:
    if not text:
        return False
    if len(text) > 80:
        return False
    if text[-1] in "。.!！？?":
        return False
    if re.match(r"^\d+[\.\)]", text):
        return False
    return True


def _relative_path(path: Path, data_root: Optional[Path]) -> str:
    if data_root is not None:
        try:
            return path.relative_to(data_root.resolve()).as_posix()
        except Exception:
            pass
    return path.as_posix()
