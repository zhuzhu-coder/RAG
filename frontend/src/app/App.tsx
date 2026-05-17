import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Database,
  Loader2,
  MessageSquareText,
  RefreshCw,
  Search,
  Send,
  Timer,
  ToggleLeft,
  ToggleRight
} from "lucide-react";

import {
  ApiError,
  AskResponse,
  ReadyResponse,
  askQuestion,
  getReady,
  warmupKnowledgeBase
} from "../api/client";
import "../styles/app.css";

type StatusVariant = "ready" | "muted" | "error";

type AnswerMeta = {
  requestId: string;
  processTime: string;
};

const emptyReady: ReadyResponse = {
  ready: false,
  status: "not_ready",
  total_documents: 0,
  total_chunks: 0,
  last_error: null
};

function errorMessage(error: unknown, fallback: string) {
  const apiError = error as ApiError;
  if (apiError?.message) {
    return apiError.request_id ? `${apiError.message} Request ID: ${apiError.request_id}` : apiError.message;
  }
  return fallback;
}

function App() {
  const [readyState, setReadyState] = useState<ReadyResponse>(emptyReady);
  const [statusVariant, setStatusVariant] = useState<StatusVariant>("muted");
  const [statusText, setStatusText] = useState("正在检查知识库状态...");
  const [question, setQuestion] = useState("");
  const [returnTrace, setReturnTrace] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [answerMeta, setAnswerMeta] = useState<AnswerMeta | null>(null);

  const statusLabel = useMemo(() => {
    if (statusVariant === "ready") {
      return "Ready";
    }
    if (statusVariant === "error") {
      return "Error";
    }
    return "Not ready";
  }, [statusVariant]);

  async function refreshReady() {
    setError("");
    try {
      const payload = await getReady();
      setReadyState(payload);
      if (payload.ready) {
        setStatusVariant("ready");
        setStatusText(`知识库已就绪：${payload.total_documents} 个文档，${payload.total_chunks} 个片段。`);
      } else if (payload.status === "error") {
        setStatusVariant("error");
        setStatusText(payload.last_error || "知识库初始化失败。");
      } else {
        setStatusVariant("muted");
        setStatusText("知识库未初始化，请先预热。");
      }
    } catch (caught) {
      setReadyState(emptyReady);
      setStatusVariant("error");
      setStatusText(errorMessage(caught, "无法连接后端服务。"));
    }
  }

  async function handleWarmup() {
    setBusy(true);
    setError("");
    setStatusVariant("muted");
    setStatusText("正在预热知识库...");
    try {
      const payload = await warmupKnowledgeBase();
      setReadyState(payload);
      setStatusVariant("ready");
      setStatusText(`知识库已就绪：${payload.total_documents} 个文档，${payload.total_chunks} 个片段。`);
    } catch (caught) {
      setStatusVariant("error");
      setStatusText("知识库预热失败。");
      setError(errorMessage(caught, "预热请求失败，请确认 API 服务仍在运行。"));
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setError("请输入问题。");
      return;
    }

    setBusy(true);
    setError("");
    setResponse(null);
    setAnswerMeta(null);
    try {
      const result = await askQuestion(trimmedQuestion, returnTrace);
      setResponse(result.payload);
      setAnswerMeta({
        requestId: result.requestId,
        processTime: result.processTime
      });
    } catch (caught) {
      setError(errorMessage(caught, "问答请求失败，请确认 API 服务仍在运行。"));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void refreshReady();
  }, []);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CampusMind RAG</p>
          <h1>校园知识库工作台</h1>
        </div>
        <div className={`status-pill status-${statusVariant}`}>{statusLabel}</div>
      </header>

      <section className="workspace">
        <form className="question-panel" onSubmit={handleSubmit}>
          <div className="panel-header">
            <div>
              <h2>问答</h2>
              <p>{statusText}</p>
            </div>
            <button className="secondary-button" type="button" onClick={handleWarmup} disabled={busy}>
              {busy ? <Loader2 className="spin" size={17} /> : <RefreshCw size={17} />}
              预热知识库
            </button>
          </div>

          <label className="question-label" htmlFor="question-input">
            问题
          </label>
          <textarea
            id="question-input"
            rows={5}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="例如：学生请假超过三天需要谁审批？"
          />

          <div className="action-row">
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={returnTrace}
                onChange={(event) => setReturnTrace(event.target.checked)}
              />
              {returnTrace ? <ToggleRight size={18} /> : <ToggleLeft size={18} />}
              <span>显示调试信息</span>
            </label>
            <button className="primary-button" type="submit" disabled={busy}>
              {busy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              提交问题
            </button>
          </div>
        </form>

        <aside className="stats-panel" aria-label="知识库状态">
          <div className="metric">
            <Database size={18} />
            <div>
              <strong>{readyState.total_documents}</strong>
              <span>文档</span>
            </div>
          </div>
          <div className="metric">
            <Search size={18} />
            <div>
              <strong>{readyState.total_chunks}</strong>
              <span>片段</span>
            </div>
          </div>
          <div className="metric">
            <Timer size={18} />
            <div>
              <strong>{answerMeta?.processTime ?? "-"}</strong>
              <span>ms</span>
            </div>
          </div>
        </aside>
      </section>

      {error ? (
        <section className="error-panel" aria-live="polite">
          <AlertCircle size={18} />
          <span>{error}</span>
        </section>
      ) : null}

      <section className="answer-panel" aria-label="回答">
        <div className="panel-header">
          <h2>回答</h2>
          <span className="meta-info">
            {answerMeta ? `request_id=${answerMeta.requestId} · ${answerMeta.processTime} ms` : "等待提问"}
          </span>
        </div>
        <div className={response?.answer ? "answer-text" : "answer-text empty-state"}>
          {response?.answer || "答案会显示在这里。"}
        </div>
      </section>

      <section className="sources-panel" aria-label="来源">
        <div className="panel-header">
          <h2>来源</h2>
          <span className="meta-info">{response?.sources.length ?? 0} 条</span>
        </div>
        {response?.sources.length ? (
          <div className="sources-list">
            {response.sources.map((source) => (
              <article className="source-card" key={`${source.source_id}-${source.chunk_index}-${source.section}`}>
                <div className="source-title">
                  <MessageSquareText size={17} />
                  <span>{source.doc_title}</span>
                </div>
                <div className="source-meta">
                  {source.doc_category} · {source.department || "未注明"} · {source.section} · page=
                  {source.page ?? "-"} · chunk={source.chunk_index} · score=
                  {source.rrf_score === null ? "-" : source.rrf_score.toFixed(4)}
                </div>
                <p>{source.snippet}</p>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-state">来源会显示在这里。</div>
        )}
      </section>

      {response?.trace ? (
        <section className="trace-panel" aria-label="调试信息">
          <div className="panel-header">
            <h2>Trace</h2>
            <span className="meta-info">{response.trace.retrieval_strategy}</span>
          </div>
          <div className="trace-grid">
            {Object.entries(response.trace.timings_ms).map(([name, value]) => (
              <div className="trace-item" key={name}>
                <span>{name}</span>
                <strong>{value.toFixed(2)} ms</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </main>
  );
}

export default App;
