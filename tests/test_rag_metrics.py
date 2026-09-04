import json
from pathlib import Path

from app.agent.rag.bm25 import BM25Retriever, tokenize
from app.agent.rag.chunker import Chunk
from app.agent.rag.query_decomposer import decompose_query, needs_extended_context
from app.agent.rag.answerability import classify_answerability


def test_rag_gold_dataset_is_valid():
    root = Path(__file__).resolve().parent.parent
    cases = json.loads((root / "app/evaluation/rag_cases.json").read_text("utf-8"))["cases"]
    assert len(cases) >= 100
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["query"].strip() and case.get("group") for case in cases)
    assert sum(not case["relevant_chunk_ids"] for case in cases) >= 10
    assert all("#" in cid for case in cases for cid in case["relevant_chunk_ids"])


def test_chinese_bm25_prefers_matching_policy():
    chunks = [
        Chunk("return#1", "退换货", "期限", "七天无理由退货从签收次日开始计算"),
        Chunk("shipping#1", "配送", "时效", "普通地区预计三到五天送达"),
    ]
    hits = BM25Retriever(chunks).search("七天无理由从哪天算", top_k=2)
    assert hits[0].chunk.chunk_id == "return#1"
    assert "七天" in tokenize("七天无理由")


def test_frozen_rag_holdout_has_hard_balanced_coverage():
    root = Path(__file__).resolve().parent.parent
    payload = json.loads(
        (root / "app/evaluation/rag_holdout_cases.json").read_text("utf-8")
    )
    cases = payload["cases"]
    assert payload["frozen"] is True
    assert len(cases) == 40
    assert len({case["id"] for case in cases}) == 40
    assert sum(len(case["relevant_chunk_ids"]) >= 2 for case in cases) >= 10
    assert sum(not case["relevant_chunk_ids"] for case in cases) == 10
    assert {case["group"] for case in cases} == {
        "multi_hop", "hard_confusable", "negation", "long_noise", "hard_no_answer"
    }


def test_query_decomposition_only_splits_composite_questions():
    assert decompose_query("手机激活后还能退吗") == []
    parts = decompose_query("耳机戴过但壳没拆，我只想退壳，同时想知道运费谁出")
    assert len(parts) >= 2
    assert any("运费" in part for part in parts)


def test_dynamic_context_and_independent_answerability():
    assert needs_extended_context("手机激活后不断重启，需要检测还是可以退货")
    assert not needs_extended_context("七天无理由从哪天算")
    assert not classify_answerability("明年会发布什么新型号").answerable
    assert not classify_answerability("快递员身份证号码是多少").answerable
    assert classify_answerability("手机激活后发生故障怎么售后").answerable
