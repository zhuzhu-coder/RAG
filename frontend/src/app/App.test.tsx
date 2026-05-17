import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import App from "./App";

const readyPayload = {
  ready: true,
  status: "ready",
  total_documents: 23,
  total_chunks: 39,
  last_error: null
};

const answerPayload = {
  question: "学生请假超过三天需要谁审批？",
  route_type: "detail",
  rewritten_query: "学生请假超过三天需要谁审批？",
  answer: "超过三天需要学院审批。[1]",
  sources: [
    {
      source_id: 1,
      doc_title: "学生请假管理办法",
      doc_category: "规章制度",
      department: "学生处",
      file_type: "md",
      section: "请假审批",
      source: "data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md",
      page: null,
      chunk_index: 0,
      rrf_score: 0.03,
      snippet: "学生请假超过三天需由辅导员和学院审批。"
    }
  ],
  trace: {
    retrieval_strategy: "hybrid",
    filters: {},
    timings_ms: {
      analysis: 1,
      retrieval: 3,
      context_build: 4,
      generation: 5,
      total: 13
    },
    retrieval_params: {
      top_k: 3,
      candidate_k: 10,
      rrf_k: 60,
      context_window_size: 1
    },
    retrieved_chunks: [],
    context_documents: [],
    source_count: 1
  }
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("CampusMind React app", () => {
  test("renders the ready status from the FastAPI backend", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify(readyPayload), {
        headers: { "Content-Type": "application/json" }
      })
    );

    render(<App />);

    expect(await screen.findByText("Ready")).toBeInTheDocument();
    expect(screen.getByText(/23 个文档/)).toBeInTheDocument();
    expect(screen.getByText(/39 个片段/)).toBeInTheDocument();
  });

  test("submits a question and renders answer, source, and trace", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" }
        })
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(answerPayload), {
          headers: {
            "Content-Type": "application/json",
            "X-Request-ID": "req-test",
            "X-Process-Time-MS": "18.5"
          }
        })
      );

    render(<App />);

    await screen.findByText("Ready");
    await userEvent.type(screen.getByLabelText("问题"), "学生请假超过三天需要谁审批？");
    await userEvent.click(screen.getByLabelText("显示调试信息"));
    await userEvent.click(screen.getByRole("button", { name: /提交问题/ }));

    expect(await screen.findByText("超过三天需要学院审批。[1]")).toBeInTheDocument();
    expect(screen.getByText("学生请假管理办法")).toBeInTheDocument();
    expect(screen.getByText(/request_id=req-test/)).toBeInTheDocument();
    expect(screen.getByText("retrieval")).toBeInTheDocument();
    expect(screen.getByText("3.00 ms")).toBeInTheDocument();
  });
});
