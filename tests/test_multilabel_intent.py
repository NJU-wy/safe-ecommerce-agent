from app.agent.response_policy import build_customer_response, classify_intents
from app.evaluation.metrics import multilabel_counts
from app.evaluation.evaluator import EvalResult, Evaluator
from app.schemas.response import IntentType


def test_primary_and_secondary_intents():
    primary, secondary = classify_intents(
        "订单 ORD-20240115-001 到哪了，物流太慢了我要投诉"
    )
    assert primary is IntentType.COMPLAINT
    assert IntentType.ORDER_QUERY in secondary


def test_response_keeps_legacy_intent_alias():
    response = build_customer_response(
        "AirPods Pro 2 续航怎么样，现在有优惠券吗？", "这里是查询结果"
    )
    assert response.intent is IntentType.PROMOTION
    assert response.primary_intent is response.intent
    assert response.secondary_intents == [IntentType.PRODUCT_CONSULT]


def test_multilabel_counts_for_micro_metrics():
    assert multilabel_counts(
        ["complaint", "order_query"],
        ["complaint", "account"],
    ) == (1, 1, 1)


def test_micro_f1_aggregates_labels_globally():
    evaluator = object.__new__(Evaluator)
    report = evaluator._aggregate([
        EvalResult("a", "a", multilabel_tp=2, multilabel_fp=1, multilabel_fn=0),
        EvalResult("b", "b", multilabel_tp=1, multilabel_fp=0, multilabel_fn=1),
    ])
    summary = report["summary"]
    assert summary["intent_micro_precision"] == 3 / 4
    assert summary["intent_micro_recall"] == 3 / 4
    assert summary["intent_micro_f1"] == 3 / 4
