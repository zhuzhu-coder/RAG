"""
校园知识库 RAG API 服务
"""

import logging
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import PROJECT_ROOT
from .system import CampusRAGSystem

logger = logging.getLogger(__name__)

# 定义请求头字段
REQUEST_ID_HEADER = "X-Request-ID"
# 定义响应头字段
PROCESS_TIME_HEADER = "X-Process-Time-MS"
# React 前端构建产物目录
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST_DIR / "index.html"


class AskRequest(BaseModel):
    """问答请求体"""
    # 用户问题 
    question: str = Field(..., min_length=1)  # 必填 长度至少1个字符
    # 是否返回来源文档
    return_sources: bool = True  # 默认返回
    # 是否返回调试链路
    return_trace: bool = False

    @field_validator("question")  # 给 question 字段添加验证器
    @classmethod  # 验证器方法必须是类方法
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class RAGService:
    """懒加载 RAG 系统，避免导入 API 时初始化模型或构建索引"""

    def __init__(self, system_factory: Callable[[], CampusRAGSystem]):
        self._system_factory = system_factory  # 创建 RAG 系统的工厂函数
        self._system: Optional[CampusRAGSystem] = None  # 初始化时系统为空
        self._init_lock = Lock() # 初始化锁，确保线程安全
        self._last_error: Optional[str] = None # 最后一次初始化错误信息

    def get_system(self) -> CampusRAGSystem:
        """获取 RAG 系统实例，首次调用时初始化"""
        if self._system is None:
            with self._init_lock:
                if self._system is None:
                    try:
                        # 创建系统实例
                        system = self._system_factory()
                        # 初始化系统
                        system.initialize_system()
                        # 构建知识库
                        system.build_knowledge_base()
                    except Exception as exc:
                        self._last_error = f"{exc.__class__.__name__}: {exc}"
                        raise
                    self._system = system
                    self._last_error = None
        return self._system

    def ready(self) -> dict:
        """返回 RAG 服务是否已完成初始化"""
        if self._system is None:
            return {
                "ready": False,
                "status": "error" if self._last_error else "not_ready",
                "total_documents": 0,
                "total_chunks": 0,
                "last_error": self._last_error,
            }

        stats = self._get_statistics()
        return {
            "ready": True,
            "status": "ready",
            "total_documents": int(stats.get("total_documents", 0) or 0),
            "total_chunks": int(stats.get("total_chunks", 0) or 0),
            "last_error": None,
        }

    def warmup(self) -> dict:
        """主动初始化 RAG 系统，供部署或演示前预热"""
        self.get_system()
        return self.ready()

    def _get_statistics(self) -> dict[str, Any]:
        """读取当前知识库统计信息"""
        if self._system is None or self._system.data_module is None:
            return {}
        return self._system.data_module.get_statistics()

    def ask(self, question: str, return_sources: bool, return_trace: bool) -> dict:
        """问答接口"""
        response = self.get_system().ask_question(
            question,
            stream=False,
            return_sources=True,
            return_trace=return_trace,
        )
        payload = response.to_dict()
        if not return_sources:
            payload["sources"] = []
        return payload

    def stats(self) -> dict:
        """获取 RAG 系统统计信息"""
        system = self.get_system()
        return system.data_module.get_statistics()


def _build_error_response(
    *,
    code: str,
    message: str,
    request_id: str,
    status_code: int,
) -> JSONResponse:
    """构造统一错误响应"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


def _is_upstream_model_error(exc: Exception) -> bool:
    """判断异常是否来自 OpenAI-compatible 上游模型服务。"""
    module_name = exc.__class__.__module__
    return module_name == "openai" or module_name.startswith("openai.")


def _build_upstream_model_error_response(exc: Exception, request_id: str) -> JSONResponse:
    """构造上游模型服务错误响应，避免前端只能看到泛化的 500。"""
    return _build_error_response(
        code="upstream_model_error",
        message=f"上游模型服务请求失败: {exc}",
        request_id=request_id,
        status_code=502,
    )


def create_app(system_factory: Callable[[], CampusRAGSystem] | None = None) -> FastAPI:
    """
    创建 FastAPI 应用，测试时可注入 fake RAG 系统
    Args:
        system_factory: 用于创建 RAG 系统的工厂函数，默认使用 CampusRAGSystem
    Returns:
        FastAPI 应用实例
    """
    service = RAGService(system_factory or CampusRAGSystem)
    app = FastAPI(title="Campus Knowledge RAG API")
    frontend_assets_dir = FRONTEND_DIST_DIR / "assets"
    if frontend_assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=frontend_assets_dir), name="frontend-assets")

    @app.middleware("http") # http 中间件
    async def request_context_middleware(request: Request, call_next):
        """为每个请求补充 request id、耗时统计和兜底错误响应"""
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        start_time = time.perf_counter() # 请求开始时间
        status_code = 500 # 默认状态码

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            ready_state = service.ready()
            # 如果 RAG 系统初始化失败，返回 503 错误
            if ready_state.get("status") == "error":
                response = _build_error_response(
                    code="rag_initialization_failed",
                    message="RAG service failed to initialize",
                    request_id=request_id,
                    status_code=503,
                )
            elif _is_upstream_model_error(exc):
                response = _build_upstream_model_error_response(exc, request_id)
            else:
                # 其他异常，返回 500 错误
                response = _build_error_response(
                    code="internal_error",
                    message="服务内部错误",
                    request_id=request_id,
                    status_code=500,
                )
            status_code = response.status_code
            logger.exception(
                "request failed method=%s path=%s request_id=%s",
                request.method,
                request.url.path,
                request_id,
            )

        duration_ms = (time.perf_counter() - start_time) * 1000 # 请求耗时，单位毫秒
        response.headers[REQUEST_ID_HEADER] = request_id # 请求 id
        response.headers[PROCESS_TIME_HEADER] = f"{duration_ms:.2f}" # 处理耗时，单位毫秒
        logger.info(
            "request method=%s path=%s status_code=%s duration_ms=%.2f request_id=%s",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
            request_id,
        )
        return response

    @app.get("/", include_in_schema=False)
    def frontend(request: Request):
        """返回 React 问答前端页面"""
        if not FRONTEND_INDEX.exists():
            request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
            return _build_error_response(
                code="frontend_not_built",
                message="React frontend has not been built. Run `npm install` and `npm run build` in frontend/.",
                request_id=request_id,
                status_code=503,
            )
        return FileResponse(FRONTEND_INDEX)

    @app.get("/health")
    def health() -> dict:
        """检查系统健康状态"""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        """检查 RAG 系统是否已完成初始化"""
        return service.ready()

    @app.post("/warmup")
    def warmup() -> dict:
        """主动预热 RAG 系统"""
        return service.warmup()

    @app.get("/stats")
    def stats() -> dict:
        """获取 RAG 系统统计信息"""
        return service.stats()

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        """问答接口"""
        return service.ask(request.question, request.return_sources, request.return_trace)

    return app


app = create_app()
