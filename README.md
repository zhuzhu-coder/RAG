# CampusMind RAG

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=222)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-Frontend-3178C6?logo=typescript&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![Chroma](https://img.shields.io/badge/Chroma-VectorDB-5B5BD6)

面向校园知识库的本地 RAG 问答系统。项目将 PDF、Markdown、TXT 等校园文档接入统一知识库，通过 Chroma 向量检索、BM25 关键词检索、RRF 融合排序和 Grounded Answer 生成，提供 CLI、FastAPI API、React Web 工作台和检索评估闭环。

| 能力 | 说明 |
| --- | --- |
| 知识库接入 | 支持 PDF、Markdown、TXT，自动抽取标题、分类、部门、页码、章节和来源路径 |
| 检索链路 | 支持 Vector、BM25、Hybrid + RRF，默认使用 Hybrid 检索 |
| 向量数据库 | 使用 Chroma 本地持久化，默认写入 `storage/chroma` |
| 生成模型 | 使用 DashScope OpenAI-compatible API，默认模型 `qwen3.6-plus` |
| 使用入口 | CLI、FastAPI、React Web、Swagger API 文档 |
| 可观测性 | 返回结构化 `sources[]`、可选 `trace`、阶段耗时、召回块和证据窗口 |
| 评估闭环 | 38 条校园问答评估集，Hybrid `hit@1=0.9474`、`hit@3=1.0000`、`MRR=0.9737` |

## 功能亮点

- **多格式文档处理**：统一加载校园规章、教务通知、生活服务、图书馆公告和网络维护文档。
- **专业目录结构**：后端代码、React 前端、知识库数据、评估集、测试和运行产物分层组织。
- **Chroma 索引缓存**：通过 manifest 记录数据指纹、embedding 模型和分块策略，避免重复构建。
- **Hybrid 检索增强**：语义检索和关键词检索并行召回，通过 RRF 融合提升中文校园问答稳定性。
- **证据窗口构建**：命中 chunk 后回填相邻片段，兼顾答案完整性、上下文噪声和 token 成本。
- **Grounded Answer 输出**：答案严格基于检索上下文，来源编号与 `sources[]` 对齐。
- **React 问答工作台**：支持状态检查、知识库预热、问答、来源展示和 Trace 调试信息。
- **检索评估工具**：可对 Vector、BM25、Hybrid 策略进行 hit@k、MRR 和 keyword coverage 对比。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 后端工程 | Python 3.11+、setuptools、pytest |
| RAG 编排 | LangChain 1.x、LangChain OpenAI、LangChain Chroma |
| 文档处理 | pypdf、LangChain document loaders、MarkdownHeaderTextSplitter、RecursiveCharacterTextSplitter |
| 检索索引 | Chroma、BM25、jieba、RRF |
| 模型服务 | DashScope OpenAI-compatible API |
| API 服务 | FastAPI、Uvicorn |
| 前端工程 | React 18、TypeScript、Vite、lucide-react |
| 质量验证 | pytest、Vitest、Testing Library、自建检索评估集 |

## 快速开始

### 1. 安装后端依赖

```powershell
cd E:\RAG
python -m pip install -e ".[dev]"
```

安装后会注册命令行入口 `campus-rag`。开发环境依赖包含 `pytest` 和 API 测试所需的 `httpx`。

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填入 DashScope API Key：

```powershell
Copy-Item .env.example .env
```

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

RAG_DATA_PATH=data/knowledge_base/campus
RAG_INDEX_SAVE_PATH=storage/chroma
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_LLM_MODEL=qwen3.6-plus
RAG_TOP_K=3
RAG_RETRIEVAL_CANDIDATE_K=10
RAG_RRF_K=60
RAG_CONTEXT_WINDOW_SIZE=1
RAG_TEMPERATURE=0.1
RAG_MAX_TOKENS=2048
```

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 无 | DashScope API Key，调用 embedding 和 LLM 时必填 |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible API 地址 |
| `RAG_DATA_PATH` | `data/knowledge_base/campus` | 校园知识库文档目录 |
| `RAG_INDEX_SAVE_PATH` | `storage/chroma` | Chroma 本地持久化目录 |
| `RAG_EMBEDDING_MODEL` | `text-embedding-v4` | 向量化模型 |
| `RAG_LLM_MODEL` | `qwen3.6-plus` | 问答生成模型 |
| `RAG_TOP_K` | `3` | 最终返回的来源数量 |
| `RAG_RETRIEVAL_CANDIDATE_K` | `10` | 检索阶段候选数量 |
| `RAG_RRF_K` | `60` | RRF 融合排序平滑参数 |
| `RAG_CONTEXT_WINDOW_SIZE` | `1` | 命中 chunk 前后回填的相邻片段数量 |

所有相对路径都会按项目根目录解析。从根目录、`src/` 子目录或安装后的命令入口运行时，路径行为保持一致。

### 3. 准备知识库文档

默认知识库目录是 `data/knowledge_base/campus/`。推荐按业务域组织文件：

```text
data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md
data/knowledge_base/campus/teaching/exams/期末考试安排.txt
data/knowledge_base/campus/life/dormitory/宿舍晚归登记说明.md
data/knowledge_base/campus/notices/library/图书馆临时闭馆.pdf
```

首次问答或主动预热时，系统会读取该目录并生成 Chroma 索引到 `storage/chroma/`。

### 4. 启动 CLI 问答

```powershell
python -m campus_rag.cli
```

或使用安装后的命令入口：

```powershell
campus-rag
```

### 5. 构建 React 前端

FastAPI 会托管 `frontend/dist`。首次访问 Web 页面前需要先构建前端：

```powershell
cd frontend
npm install
npm run build
cd ..
```

开发前端时可以使用 Vite dev server，接口会代理到本地 FastAPI：

```powershell
cd frontend
npm run dev
```

### 6. 启动 FastAPI 服务

```powershell
python -X utf8 -m uvicorn campus_rag.api:app --host 127.0.0.1 --port 8000
```

Windows 本地运行建议保留 `-X utf8`，避免中文日志和终端编码问题。

启动后访问：

| 地址 | 说明 |
| --- | --- |
| `http://127.0.0.1:8000/` | React Web 问答工作台 |
| `http://127.0.0.1:8000/docs` | Swagger API 文档 |
| `http://127.0.0.1:8000/health` | 进程健康检查 |
| `http://127.0.0.1:8000/ready` | RAG 初始化状态 |

## Web 与 API

React 页面提供状态检查、知识库预热、问题提交、答案展示、来源列表和 Trace 调试信息。生产模式下，FastAPI 从 `frontend/dist` 托管页面；如果前端尚未构建，`/` 会返回明确的 `frontend_not_built` 错误，API 路由仍可正常访问。

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/health` | 进程健康检查，不触发 RAG 初始化 |
| GET | `/ready` | RAG 初始化状态、文档数和 chunk 数 |
| POST | `/warmup` | 主动加载模型、文档和索引 |
| GET | `/stats` | 文档、chunk、分类、部门和文件类型统计 |
| POST | `/ask` | 问答接口，返回答案、来源和可选 Trace |

### `/ask` 请求

```json
{
  "question": "我晚归了会怎么样",
  "return_sources": true,
  "return_trace": true
}
```

### `/ask` 响应

```json
{
  "question": "我晚归了会怎么样",
  "route_type": "detail",
  "rewritten_query": "学生晚归后果及处理规定",
  "answer": "...",
  "sources": [
    {
      "source_id": 1,
      "doc_title": "宿舍晚归登记说明",
      "doc_category": "校园生活",
      "department": "学生处",
      "file_type": "md",
      "section": "晚归登记",
      "source": "data/knowledge_base/campus/life/dormitory/宿舍晚归登记说明.md",
      "page": null,
      "chunk_index": 0,
      "rrf_score": 0.03,
      "snippet": "..."
    }
  ],
  "trace": {
    "retrieval_strategy": "hybrid",
    "retrieval_params": {
      "top_k": 3,
      "candidate_k": 10,
      "rrf_k": 60,
      "context_window_size": 1
    },
    "timings_ms": {
      "retrieval": 8.1,
      "generation": 1200.0,
      "total": 1221.0
    },
    "source_count": 1
  }
}
```

关键字段：

| 字段 | 说明 |
| --- | --- |
| `route_type` | 查询类型，例如列表类问题或细节类问题 |
| `rewritten_query` | 用于检索的查询文本 |
| `sources[]` | 与答案引用编号对齐的结构化来源 |
| `trace.retrieval_strategy` | 实际使用的检索策略 |
| `trace.retrieval_params` | 本次检索使用的 top-k、候选数、RRF 参数和证据窗口大小 |
| `trace.timings_ms` | 查询分析、检索、上下文构建、生成和总耗时 |

`return_sources=false` 时，响应中的 `sources` 返回空数组。`return_trace=true` 时，`trace` 会包含检索参数、召回摘要、上下文文档摘要和阶段耗时，适合调试检索与生成链路。

## 系统架构

```text
PDF / Markdown / TXT 校园文档
        |
        v
文档加载与元数据抽取
        |
        v
结构化分块
parent_id / chunk_id / chunk_index / section
        |
        v
Chroma 向量数据库 + BM25 关键词索引
        |
        v
Vector / BM25 / Hybrid 检索
        |
        v
RRF 融合排序与去重
        |
        v
命中 chunk 邻域证据窗口
        |
        v
Grounded Answer 生成
        |
        v
answer + sources[] + optional trace
```

证据窗口默认取命中 chunk 前后各 1 个相邻片段：

```text
chunk_index - window_size
chunk_index
chunk_index + window_size
```

通过 `RAG_CONTEXT_WINDOW_SIZE` 可以控制窗口大小。调大窗口可以提供更完整上下文，但也会增加 token 成本和噪声。

## 检索评估

评估集位于 `evaluations/datasets/campus_smoke_eval_set.jsonl`，覆盖规章制度、教务教学、生活服务、图书馆公告和信息中心通知等场景。

BM25-only 评估不依赖 DashScope API Key：

```powershell
python evaluations\run_retrieval_eval.py --strategies bm25 --json
```

Vector / Hybrid 评估需要可用的 DashScope API Key：

```powershell
python evaluations\run_retrieval_eval.py --strategies vector bm25 hybrid --json
```

当前三策略评估结果：

| Strategy | Cases | hit@1 | hit@3 | MRR | keyword_coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| vector | 38 | 0.8684 | 0.9737 | 0.9167 | 0.9342 |
| bm25 | 38 | 0.8684 | 1.0000 | 0.9342 | 0.9342 |
| hybrid | 38 | 0.9474 | 1.0000 | 0.9737 | 0.9605 |

已保存的评估结果：

```text
evaluations/results/2026-05-12-bm25-baseline.json
evaluations/results/2026-05-12-vector-bm25-hybrid-summary.json
```

## 测试

后端测试：

```powershell
python -m pytest
```

前端测试和构建：

```powershell
cd frontend
npm test
npm run build
```

测试覆盖范围包括：

- 配置读取、项目路径解析和包入口。
- PDF、Markdown、TXT 文档加载与元数据抽取。
- 文档分块、父子文档映射和证据窗口构建。
- Chroma manifest、本地向量数据库缓存和稳定 ID 写入。
- Vector、BM25、Hybrid 检索和 RRF 排序。
- Grounded Answer prompt、引用来源、响应 schema 和 Trace。
- API 健康检查、预热、问答、错误响应和 React 静态资源托管。
- 前端问答提交流程、答案渲染、来源渲染和 Trace 渲染。

## 项目结构

```text
src/campus_rag/
  api.py                   # FastAPI 应用与路由
  cli.py                   # 命令行问答入口
  config.py                # 环境变量与项目路径配置
  system.py                # RAG 系统编排
  pipeline/                # 文档接入、分块、索引、检索、生成和响应 schema

frontend/                  # React + TypeScript + Vite Web 问答工作台
  src/app/                 # 页面入口与应用组合
  src/api/                 # 前端 API client
  src/components/          # 可复用 UI 组件
  src/styles/              # 前端样式
  src/types/               # API 与业务类型

data/knowledge_base/campus/ # 校园知识库文档
evaluations/               # 检索评估脚本、评估集和结果
tests/                     # 按 api、pipeline、system、evaluation 分组的测试
storage/chroma/            # 本地 Chroma 运行产物，不提交版本库
pyproject.toml             # 包元数据、依赖、pytest 配置和命令入口
.env.example               # 环境变量模板
```

## 开发说明

- `.env`、`.venv/`、`storage/`、`frontend/dist/`、`frontend/node_modules/`、缓存目录和临时评估结果不进入版本库。
- `storage/chroma/` 是本地运行产物，删除后会在下次预热或问答时基于知识库重新生成。
- 旧 FAISS 索引不会迁移；当前版本会使用 Chroma 重新构建向量库。
- 修改检索、分块、来源或 Trace 链路时，建议同时运行 `python -m pytest` 和检索评估命令。
- 修改 React 前端时，建议运行 `npm test` 和 `npm run build`。

## 常见问题

### 访问 `/` 返回 `frontend_not_built`

说明 React 生产构建产物不存在。运行：

```powershell
cd frontend
npm install
npm run build
cd ..
```

然后重新访问 `http://127.0.0.1:8000/`。

### `/ask` 返回上游模型错误

通常是 `DASHSCOPE_API_KEY` 未配置、额度不足、模型不可用或网络访问异常。先检查 `.env`，再访问 `/ready` 确认服务初始化状态。

### Chroma 索引需要重新构建

删除 `storage/chroma/` 后，系统会在下次 `/warmup`、CLI 问答或 `/ask` 时重新读取知识库并构建索引。

### Windows 终端中文显示异常

启动服务时使用：

```powershell
python -X utf8 -m uvicorn campus_rag.api:app --host 127.0.0.1 --port 8000
```

### 前端开发接口请求失败

确认 FastAPI 已运行在 `http://127.0.0.1:8000`。Vite dev server 会将 `/ask`、`/health`、`/ready`、`/stats` 和 `/warmup` 代理到该地址。

## Roadmap

- OCR：支持扫描版 PDF 和图片型公告文本抽取。
- 权限：接入不同角色、部门或数据域的访问控制。
- 多租户：支持多个学院、部门或知识库实例。
- 增量索引：针对新增、修改和删除文档进行局部索引刷新。
- 部署观测：补充结构化日志、请求指标、链路追踪和服务看板。
