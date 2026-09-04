from app.agent.rag.reranker import derive_rerank_url


def test_rerank_url_is_derived_from_bailian_compatible_base():
    assert derive_rerank_url("https://dashscope.aliyuncs.com/compatible-mode/v1") == (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
