from campus_rag.pipeline import RAGResponse, RAGTrace, RetrievedSource


def test_rag_response_serializes_sources_to_plain_dicts():
    source = RetrievedSource(
        source_id=1,
        doc_title="学生请假管理办法",
        doc_category="规章制度",
        department="学生处",
        file_type="md",
        section="请假审批",
        source="data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md",
        page=None,
        chunk_index=2,
        rrf_score=0.0325,
        snippet="学生请假超过三天需由辅导员和学院审批。",
    )
    response = RAGResponse(
        question="学生请假超过三天需要谁审批？",
        route_type="detail",
        rewritten_query="学生请假超过三天需要谁审批？",
        answer="需要辅导员和学院审批。[1]",
        sources=[source],
    )

    assert response.to_dict() == {
        "question": "学生请假超过三天需要谁审批？",
        "route_type": "detail",
        "rewritten_query": "学生请假超过三天需要谁审批？",
        "answer": "需要辅导员和学院审批。[1]",
        "sources": [
            {
                "source_id": 1,
                "doc_title": "学生请假管理办法",
                "doc_category": "规章制度",
                "department": "学生处",
                "file_type": "md",
                "section": "请假审批",
                "source": "data/knowledge_base/campus/regulations/student_affairs/学生请假管理办法.md",
                "page": None,
                "chunk_index": 2,
                "rrf_score": 0.0325,
                "snippet": "学生请假超过三天需由辅导员和学院审批。",
            }
        ],
    }
    assert "trace" not in response.to_dict()


def test_rag_response_serializes_trace_when_present():
    trace = RAGTrace(
        retrieval_strategy="hybrid",
        filters={},
        timings_ms={
            "analysis": 3.3,
            "retrieval": 3.3,
            "context_build": 4.4,
            "generation": 5.5,
            "total": 16.5,
        },
        retrieval_params={
            "top_k": 3,
            "candidate_k": 10,
            "rrf_k": 60,
            "context_window_size": 1,
        },
        retrieved_chunks=[
            {
                "rank": 1,
                "doc_title": "宿舍晚归登记说明",
                "section": "备注",
                "chunk_index": 1,
                "rrf_score": 0.0325,
            }
        ],
        context_documents=[
            {
                "source_id": 1,
                "doc_title": "宿舍晚归登记说明",
                "context_window_size": 1,
                "context_chunk_indices": [0, 1, 2],
            }
        ],
        source_count=1,
    )
    response = RAGResponse(
        question="我晚归了会怎么样",
        route_type="detail",
        rewritten_query="宿舍晚归登记说明",
        answer="需要登记并说明情况。[1]",
        sources=[],
        trace=trace,
    )

    payload = response.to_dict()

    assert payload["trace"] == {
        "retrieval_strategy": "hybrid",
        "filters": {},
        "timings_ms": {
            "analysis": 3.3,
            "retrieval": 3.3,
            "context_build": 4.4,
            "generation": 5.5,
            "total": 16.5,
        },
        "retrieval_params": {
            "top_k": 3,
            "candidate_k": 10,
            "rrf_k": 60,
            "context_window_size": 1,
        },
        "retrieved_chunks": [
            {
                "rank": 1,
                "doc_title": "宿舍晚归登记说明",
                "section": "备注",
                "chunk_index": 1,
                "rrf_score": 0.0325,
            }
        ],
        "context_documents": [
            {
                "source_id": 1,
                "doc_title": "宿舍晚归登记说明",
                "context_window_size": 1,
                "context_chunk_indices": [0, 1, 2],
            }
        ],
        "source_count": 1,
    }

