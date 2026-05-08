from rag_modules import RAGResponse, RetrievedSource


def test_rag_response_serializes_sources_to_plain_dicts():
    source = RetrievedSource(
        source_id=1,
        dish_name="番茄炒蛋",
        category="素菜",
        difficulty="非常简单",
        section="操作步骤",
        source="data/cook/vegetable_dish/番茄炒蛋.md",
        chunk_index=2,
        rrf_score=0.0325,
        snippet="鸡蛋打散，番茄切块。",
    )
    response = RAGResponse(
        question="番茄炒蛋怎么做？",
        route_type="detail",
        rewritten_query="番茄炒蛋怎么做？",
        answer="先炒鸡蛋，再炒番茄。[1]",
        sources=[source],
    )

    assert response.to_dict() == {
        "question": "番茄炒蛋怎么做？",
        "route_type": "detail",
        "rewritten_query": "番茄炒蛋怎么做？",
        "answer": "先炒鸡蛋，再炒番茄。[1]",
        "sources": [
            {
                "source_id": 1,
                "dish_name": "番茄炒蛋",
                "category": "素菜",
                "difficulty": "非常简单",
                "section": "操作步骤",
                "source": "data/cook/vegetable_dish/番茄炒蛋.md",
                "chunk_index": 2,
                "rrf_score": 0.0325,
                "snippet": "鸡蛋打散，番茄切块。",
            }
        ],
    }
