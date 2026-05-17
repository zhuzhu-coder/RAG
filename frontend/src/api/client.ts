import type { ApiError, AskResponse, ReadyResponse } from "../types/api";

export type { ApiError, AskResponse, ReadyResponse } from "../types/api";

async function readError(response: Response): Promise<ApiError> {
  try {
    const payload = (await response.json()) as { error?: ApiError };
    return payload.error ?? { message: `请求失败，HTTP ${response.status}` };
  } catch {
    return { message: `请求失败，HTTP ${response.status}` };
  }
}

async function fetchJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    throw await readError(response);
  }
  return (await response.json()) as T;
}

export async function getReady(): Promise<ReadyResponse> {
  return fetchJson<ReadyResponse>("/ready");
}

export async function warmupKnowledgeBase(): Promise<ReadyResponse> {
  return fetchJson<ReadyResponse>("/warmup", { method: "POST" });
}

export async function askQuestion(question: string, returnTrace: boolean) {
  const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      return_sources: true,
      return_trace: returnTrace
    })
  });

  if (!response.ok) {
    throw await readError(response);
  }

  return {
    payload: (await response.json()) as AskResponse,
    requestId: response.headers.get("X-Request-ID") ?? "-",
    processTime: response.headers.get("X-Process-Time-MS") ?? "-"
  };
}
