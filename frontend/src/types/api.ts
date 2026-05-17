export type ReadyResponse = {
  ready: boolean;
  status: "ready" | "not_ready" | "error";
  total_documents: number;
  total_chunks: number;
  last_error: string | null;
};

export type RetrievedSource = {
  source_id: number;
  doc_title: string;
  doc_category: string;
  department: string;
  file_type: string;
  section: string;
  source: string;
  page: number | null;
  chunk_index: number;
  rrf_score: number | null;
  snippet: string;
};

export type RagTrace = {
  retrieval_strategy: string;
  filters: Record<string, unknown>;
  timings_ms: Record<string, number>;
  retrieval_params: {
    top_k: number;
    candidate_k: number;
    rrf_k: number;
    context_window_size: number;
  };
  retrieved_chunks: Array<Record<string, unknown>>;
  context_documents: Array<Record<string, unknown>>;
  source_count: number;
};

export type AskResponse = {
  question: string;
  route_type: string;
  rewritten_query: string;
  answer: string;
  sources: RetrievedSource[];
  trace?: RagTrace;
};

export type ApiError = {
  message: string;
  request_id?: string;
};
