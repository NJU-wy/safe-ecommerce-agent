import json
from pathlib import Path


def test_rag_gold_dataset_is_valid():
    root = Path(__file__).resolve().parent.parent
    cases = json.loads((root / "app/evaluation/rag_cases.json").read_text("utf-8"))["cases"]
    assert len(cases) >= 40
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["query"].strip() and case.get("group") for case in cases)
    assert sum(not case["relevant_chunk_ids"] for case in cases) >= 5
    assert all("#" in cid for case in cases for cid in case["relevant_chunk_ids"])
