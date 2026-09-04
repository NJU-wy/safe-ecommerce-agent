import json

from app.agent.rag.answerability import classify_answerability
from app.agent.tool_result_compactor import compact_tool_result


def test_future_price_query_is_rejected_without_similarity_threshold():
    decision = classify_answerability("牛仔裤重新补货时价格会不会降到五百元以内")

    assert decision.answerable is False
    assert decision.confidence >= 0.9


def test_policy_scope_question_remains_answerable():
    decision = classify_answerability("知识库能预测下个月的促销价格吗")

    assert decision.answerable is True


def test_compactor_preserves_five_selected_rag_contexts():
    result = {
        "success": True,
        "results": [
            {"doc": f"doc-{i}", "section": "s", "score": 1 - i / 10, "text": "证据"}
            for i in range(6)
        ],
    }

    compacted = json.loads(compact_tool_result("search_knowledge", json.dumps(result)))

    assert len(compacted["results"]) == 5
