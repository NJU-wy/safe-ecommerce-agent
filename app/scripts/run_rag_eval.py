"""评估 RAG 召回质量：输出 Recall@1、Recall@3 和 MRR。"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.agent.rag.llm_decomposer import LLMQueryDecomposer  # noqa: E402
from app.agent.rag.reranker import BailianReranker  # noqa: E402
from app.config.settings import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RAG 检索金标评估")
    parser.add_argument("--dataset", default="app/evaluation/rag_cases.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="app/sessions/rag_eval_report.json")
    parser.add_argument(
        "--no-answer-threshold", type=float, default=None,
        help="可选的模式专属拒识阈值；不同检索模式不可共用",
    )
    parser.add_argument(
        "--mode", choices=["semantic", "bm25", "hybrid"],
        default=settings.rag_retrieval_mode,
        help="检索方式；用于在同一金标集上做公平对比",
    )
    parser.add_argument(
        "--decomposer", choices=["none", "rule", "llm"], default="none",
        help="查询拆解方式",
    )
    parser.add_argument("--candidate-k", type=int, default=None, help="每个原/子查询召回的候选数")
    parser.add_argument("--rerank", action="store_true", help="用百炼排序模型将候选重排到top-k")
    args = parser.parse_args()

    cases = json.loads((ROOT / args.dataset).read_text("utf-8"))["cases"]
    if settings.rag_backend == "numpy":
        backend = create_backend("numpy", index_path=ROOT / settings.kb_index_path)
    else:
        backend = create_backend("chroma", persist_dir=ROOT / settings.chroma_persist_dir,
                                 collection_name=settings.chroma_collection)
    retriever = KnowledgeRetriever(
        Embedder(
            settings.openai_api_key, settings.openai_base_url,
            settings.embedding_model, settings.embedding_batch_size,
        ),
        backend,
        mode=args.mode,
    )
    retriever.load()
    llm_decomposer = (
        LLMQueryDecomposer(
            settings.openai_api_key, settings.openai_base_url,
            settings.rag_decomposition_model,
        ) if args.decomposer == "llm" else None
    )
    reranker = (
        BailianReranker(
            settings.openai_api_key, settings.openai_base_url, settings.rag_rerank_model
        ) if args.rerank else None
    )

    rows, reciprocal_ranks = [], []
    doc_stats = defaultdict(lambda: {"count": 0, "hit1": 0, "hit3": 0})
    for case in cases:
        relevant = set(case["relevant_chunk_ids"])
        subqueries = llm_decomposer.decompose(case["query"]) if llm_decomposer else None
        output_k = max(3, args.top_k)
        retrieve_k = max(output_k, args.candidate_k or output_k)
        hits = retriever.search(
            case["query"], top_k=retrieve_k,
            decompose=args.decomposer == "rule", candidate_k=args.candidate_k,
            subqueries=subqueries,
        )
        if reranker:
            hits = reranker.rerank(case["query"], hits, top_n=output_k)
        ids = [hit.chunk.chunk_id for hit in hits]
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in relevant), None)
        rr = 1 / rank if rank else 0.0
        recall_1 = len(set(ids[:1]) & relevant) / len(relevant) if relevant else 0.0
        recall_3 = len(set(ids[:3]) & relevant) / len(relevant) if relevant else 0.0
        recall_5 = len(set(ids[:5]) & relevant) / len(relevant) if relevant else 0.0
        dcg_3 = sum(
            1.0 / math.log2(i + 2) for i, cid in enumerate(ids[:3]) if cid in relevant
        )
        ideal_dcg_3 = sum(
            1.0 / math.log2(i + 2) for i in range(min(3, len(relevant)))
        ) if relevant else 0.0
        ndcg_3 = dcg_3 / ideal_dcg_3 if ideal_dcg_3 else 0.0
        top_score = hits[0].score if hits else 0.0
        answerable = bool(relevant)
        predicted_answerable = (
            top_score >= args.no_answer_threshold
            if args.no_answer_threshold is not None else None
        )
        if answerable:
            reciprocal_ranks.append(rr)
            expected_doc = case["relevant_chunk_ids"][0].split("#", 1)[0]
            stats = doc_stats[expected_doc]
            stats["count"] += 1
            stats["hit1"] += int(rank == 1)
            stats["hit3"] += int(rank is not None and rank <= 3)
        rows.append({**case, "retrieved_chunk_ids": ids, "first_relevant_rank": rank,
                     "recall_at_1": recall_1, "recall_at_3": recall_3,
                     "recall_at_5": recall_5,
                     "ndcg_at_3": ndcg_3, "reciprocal_rank": rr, "top_score": top_score,
                     "predicted_answerable": predicted_answerable})

    total = len(rows)
    answerable_rows = [r for r in rows if r["relevant_chunk_ids"]]
    no_answer_rows = [r for r in rows if not r["relevant_chunk_ids"]]
    group_stats = {}
    for group in sorted({r.get("group", "unknown") for r in answerable_rows}):
        subset = [r for r in answerable_rows if r.get("group", "unknown") == group]
        group_stats[group] = {
            "count": len(subset),
            "hit_at_1": sum(r["first_relevant_rank"] == 1 for r in subset) / len(subset),
            "hit_at_3": sum(r["first_relevant_rank"] is not None and r["first_relevant_rank"] <= 3 for r in subset) / len(subset),
            "hit_at_5": sum(r["first_relevant_rank"] is not None and r["first_relevant_rank"] <= 5 for r in subset) / len(subset),
            "recall_at_1": sum(r["recall_at_1"] for r in subset) / len(subset),
            "recall_at_3": sum(r["recall_at_3"] for r in subset) / len(subset),
            "recall_at_5": sum(r["recall_at_5"] for r in subset) / len(subset),
            "ndcg_at_3": sum(r["ndcg_at_3"] for r in subset) / len(subset),
        }
    no_answer_tp = sum(
        r["predicted_answerable"] is False for r in no_answer_rows
    ) if args.no_answer_threshold is not None else 0
    no_answer_fn = len(no_answer_rows) - no_answer_tp if args.no_answer_threshold is not None else 0
    no_answer_fp = sum(
        r["predicted_answerable"] is False for r in answerable_rows
    ) if args.no_answer_threshold is not None else 0
    no_answer_precision = (
        no_answer_tp / (no_answer_tp + no_answer_fp)
        if no_answer_tp + no_answer_fp else 0.0
    ) if args.no_answer_threshold is not None else None
    no_answer_recall = (
        no_answer_tp / (no_answer_tp + no_answer_fn)
        if no_answer_tp + no_answer_fn else 0.0
    ) if args.no_answer_threshold is not None else None
    no_answer_f1 = (
        2 * no_answer_precision * no_answer_recall / (no_answer_precision + no_answer_recall)
        if no_answer_precision + no_answer_recall else 0.0
    ) if args.no_answer_threshold is not None else None
    summary = {
        "total": total,
        "answerable": len(answerable_rows),
        "no_answer": len(no_answer_rows),
        "backend": settings.rag_backend,
        "embedding_model": settings.embedding_model,
        "retrieval_mode": args.mode,
        "query_decomposer": args.decomposer,
        "candidate_k": args.candidate_k or max(3, args.top_k),
        "reranker": settings.rag_rerank_model if reranker else None,
        "hit_at_1": sum(r["first_relevant_rank"] == 1 for r in answerable_rows) / len(answerable_rows),
        "hit_at_3": sum(r["first_relevant_rank"] is not None and r["first_relevant_rank"] <= 3 for r in answerable_rows) / len(answerable_rows),
        "hit_at_5": sum(r["first_relevant_rank"] is not None and r["first_relevant_rank"] <= 5 for r in answerable_rows) / len(answerable_rows),
        "recall_at_1": sum(r["recall_at_1"] for r in answerable_rows) / len(answerable_rows),
        "recall_at_3": sum(r["recall_at_3"] for r in answerable_rows) / len(answerable_rows),
        "recall_at_5": sum(r["recall_at_5"] for r in answerable_rows) / len(answerable_rows),
        "ndcg_at_3": sum(r["ndcg_at_3"] for r in answerable_rows) / len(answerable_rows),
        "mrr": sum(reciprocal_ranks) / len(answerable_rows),
        "no_answer_threshold": args.no_answer_threshold,
        "no_answer_accuracy": (
            sum(not r["predicted_answerable"] for r in no_answer_rows) / len(no_answer_rows)
            if args.no_answer_threshold is not None else None
        ),
        "answerable_coverage": (
            sum(r["predicted_answerable"] for r in answerable_rows) / len(answerable_rows)
            if args.no_answer_threshold is not None else None
        ),
        "no_answer_precision": no_answer_precision,
        "no_answer_recall": no_answer_recall,
        "no_answer_f1": no_answer_f1,
        "by_group": group_stats,
        "by_document": dict(doc_stats),
    }
    if reranker:
        summary["rerank_calls"] = reranker.calls
        summary["rerank_tokens"] = reranker.total_tokens
    report = {"summary": summary, "cases": rows}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告已写入: {output}")


if __name__ == "__main__":
    main()
