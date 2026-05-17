# CampusMind RAG

CampusMind RAG 是一个面向校园知识库的 RAG 问答系统，覆盖规章制度、教务通知、生活服务、图书馆公告和网络维护等高频校园文档场景。系统基于 LangChain 1.x、Chroma、BM25、RRF、FastAPI 和 React 构建，支持 PDF、Markdown、TXT 文档接入，提供 CLI、Web、HTTP API、结构化来源、Trace 链路和检索评估闭环。

| 项目维度 | 说明 |
| --- | --- |
| 运行形态 | 命令行问答、FastAPI 服务、轻量 Web 问答页 |
| 文档类型 | PDF、Markdown、TXT |
| 检索策略 | Vector、BM25、Hybrid + RRF |
| 回答约束 | Grounded Answer、来源编号、证据上下文 |
| 可观测性 | `sources[]`、`trace`、阶段耗时、召回摘要 |
| 评估结果 | 38 条校园问答评估集，Hybrid `hit@1=0.9474`、`hit@3=1.0000`、`MRR=0.9737` |

## 核心亮点

- 多格式文档接入：统一加载 PDF、Markdown、TXT，抽取标题、分类、部门、文件类型、来源路径、页码和章节等元数据。
- 索引缓存机制：使用 Chroma 保存本地向量数据库，通过 manifest 记录数据指纹、embedding 模型和分块策略，减少重复构建成本。
- Hybrid 检索链路：同时保留语义检索和关键词检索能力，通过 RRF 融合排序，在中文校园问答场景中提升召回稳定性。
- 证据窗口：命中 chunk 后回填相邻片段作为生成上下文，兼顾答案完整性、上下文噪声和 token 成本。
- Grounded Answer：回答严格基于检索上下文，输出与答案引用编号对齐的结构化来源。
- 服务化接口：提供健康检查、就绪状态、知识库预热、统计信息和问答接口，便于本地演示和服务集成。
- Trace 链路：返回检索策略、过滤条件、阶段耗时、检索参数、召回块和证据窗口摘要，方便定位检索与生成链路表现。
- 评估闭环：内置校园问答 JSONL 评估集，可对 Vector、BM25、Hybrid 三种策略进行对比。

## 技术栈

| 层级 | 选型 |
| --- | --- |
| 语言与工程 | Python 3.11+、setuptools、pytest |
| 文档处理 | LangChain document loaders、pypdf |
| 分块策略 | MarkdownHeaderTextSplitter、RecursiveCharacterTextSplitter |
| 向量索引 | Chroma 本地持久化 |
| 关键词检索 | BM25、jieba 中文分词 |
| 生成模型 | DashScope OpenAI-compatible API |
| API 服务 | FastAPI、Uvicorn |
| 前端页面 | React、TypeScript、Vite |
| 评估体系 | 自建 JSONL 评估集、hit@k、MRR、keyword coverage |

## 快速开始

### 1. 安装依赖

```powershell
cd E:\RAG
python -m pip install -e ".[dev]"
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填入 DashScope API Key：

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

### 3. 准备校园文档

默认知识库目录为 `data/knowledge_base/campus/`，推荐按业务域组织文件：

```text
data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md
data/knowledge_base/campus/teaching/exams/期末考试安排.txt
data/knowledge_base/campus/life/dormitory/宿舍晚归登记说明.md
data/knowledge_base/campus/notices/library/图书馆临时闭馆.pdf
```

### 4. 启动命令行问答

```powershell
python -m campus_rag.cli
```

安装后也可以使用命令入口：

```powershell
campus-rag
```

### 5. 启动 FastAPI 服务

首次访问 Web 页面前先构建 React 前端：

```powershell
cd frontend
npm install
npm run build
cd ..
```

```powershell
python -X utf8 -m uvicorn campus_rag.api:app --host 127.0.0.1 --port 8000
```

启动后访问：

| 地址 | 说明 |
| --- | --- |
| `http://127.0.0.1:8000/` | React Web 问答页 |
| `http://127.0.0.1:8000/docs` | Swagger API 文档 |
| `http://127.0.0.1:8000/health` | 服务健康检查 |

Windows 本地运行建议保留 `-X utf8`，避免中文输出和终端编码问题。

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
Chroma 向量数据库  +  BM25 关键词索引
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

通过 `RAG_CONTEXT_WINDOW_SIZE` 可以控制证据窗口大小。

## API

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/health` | 进程健康检查，不触发 RAG 初始化 |
| GET | `/ready` | RAG 初始化状态与知识库规模 |
| POST | `/warmup` | 主动加载模型、文档和索引 |
| GET | `/stats` | 文档、chunk、分类、部门、文件类型统计 |
| POST | `/ask` | 问答接口，返回答案、来源和可选 Trace |

### `/ask` 请求示例

```json
{
  "question": "我晚归了会怎么样",
  "return_sources": true,
  "return_trace": true
}
```

### `/ask` 响应示例

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
    "timings_ms": {
      "retrieval": 8.1,
      "generation": 1200.0,
      "total": 1221.0
    },
    "source_count": 1
  }
}
```

### 关键响应字段

| 字段 | 说明 |
| --- | --- |
| `route_type` | 查询类型，例如列表类问题或细节类问题 |
| `rewritten_query` | 用于检索的查询文本 |
| `sources[]` | 与答案引用编号对齐的结构化来源 |
| `trace.retrieval_strategy` | 实际使用的检索策略 |
| `trace.timings_ms` | 查询分析、检索、上下文构建、生成和总耗时 |
| `trace.retrieval_params` | `top_k`、`candidate_k`、`rrf_k`、`context_window_size` |
| `trace.retrieved_chunks` | 原始召回 chunk 摘要 |
| `trace.context_documents` | 进入生成阶段的证据窗口摘要 |

`return_sources=false` 时，响应中的 `sources` 返回空数组。`return_trace=true` 时，`trace.source_count` 仍记录内部构建出的来源数量，方便调试链路。

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

评估结果文件：

```text
evaluations/results/2026-05-12-bm25-baseline.json
evaluations/results/2026-05-12-vector-bm25-hybrid-summary.json
```

## 测试与质量

运行测试：

```powershell
python -m pytest
```

测试覆盖范围：

- 配置读取和项目路径解析
- PDF、Markdown、TXT 文档加载
- 文档分块、父子文档映射和证据窗口
- Chroma 索引 manifest 与本地向量数据库缓存
- Vector、BM25、Hybrid 检索和 RRF 排序
- Grounded Answer prompt、引用来源和流式输出
- API 健康检查、预热、问答、错误响应和静态资源托管
- `sources[]`、`trace`、评估指标和包入口

## 项目结构

```text
src/campus_rag/
  api.py                 # FastAPI 应用与路由
  cli.py                 # 命令行问答入口
  config.py              # 环境变量与项目路径配置
  system.py              # RAG 系统编排
  pipeline/              # 文档接入、分块、索引、检索、生成、响应 schema

frontend/                # React + TypeScript + Vite Web 问答台
  src/app/               # 页面入口与应用组合
  src/api/               # 前端 API client
  src/components/        # 可复用 UI 组件
  src/styles/            # 前端样式
  src/types/             # API 与业务类型

tests/                   # 按 api、pipeline、system、evaluation 分组的测试
evaluations/             # 检索评估脚本、评估集和结果
data/knowledge_base/campus/ # 校园知识库文档
storage/chroma/          # 本地 Chroma 运行产物
pyproject.toml           # 包元数据、依赖、pytest 配置、命令入口
.env.example             # 环境变量模板
```

默认读取 `data/knowledge_base/campus`，默认把本地向量数据库写入 `storage/chroma/`。所有相对路径都会按项目根目录解析，因此从根目录、`src/` 子目录或安装后的包入口运行都能得到一致路径。

## 开发说明

- `.env`、`.venv/`、`storage/`、`frontend/dist/`、`frontend/node_modules/`、缓存目录和临时评估结果不进入版本库。
- `storage/chroma/` 是本地运行产物，删除后会在下次构建知识库时重新生成。
- `evaluations/results/2026-05-12-bm25-baseline.json` 是已保存 baseline，便于对比检索效果。
- 涉及检索、分块、来源或 Trace 链路时，建议同时运行 `python -m pytest` 和检索评估命令。

## Roadmap

- OCR：支持扫描版 PDF 和图片型公告文本抽取。
- 权限：接入不同角色、部门或数据域的访问控制。
- 多租户：支持多个学院、部门或知识库实例。
- 增量索引：针对新增和删除文档进行局部索引刷新。
- 部署观测：补充结构化日志、请求指标、链路追踪和服务看板。
