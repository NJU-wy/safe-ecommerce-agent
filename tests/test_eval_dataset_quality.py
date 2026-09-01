from collections import Counter

from app.evaluation.dataset import load_dataset
from app.schemas.response import IntentType


def test_expanded_dataset_has_coverage_and_valid_schema():
    cases = load_dataset("app/evaluation/cases.json")
    ids = [case.id for case in cases]
    valid_intents = {intent.value for intent in IntentType}

    assert len(cases) == 100
    assert len(ids) == len(set(ids))
    assert all(case.turns and all(turn.strip() for turn in case.turns) for case in cases)
    assert all(case.expected_intent in valid_intents for case in cases)
    assert sum(len(case.turns) > 1 for case in cases) >= 10
    assert sum("apply_refund" in case.forbidden_tools for case in cases) >= 20


def test_expanded_dataset_category_distribution():
    cases = load_dataset("app/evaluation/cases.json")
    groups = Counter(case.id.split("_", 1)[0] for case in cases)

    assert groups["order"] >= 12
    assert groups["logistics"] >= 10
    assert groups["product"] >= 12
    assert groups["policy"] >= 10
    assert groups["refund"] >= 15
    assert groups["complaint"] >= 8
