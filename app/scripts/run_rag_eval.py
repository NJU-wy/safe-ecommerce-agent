"""评估 RAG 召回质量：输出 Recall@1、Recall@3 和 MRR。"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.config.settings import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RAG 检索金标评估")
    parser.add_argument("--dataset", default="app/evaluation/rag_cases.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="app/sessions/rag_eval_report.json")
    parser.add_argument("--no-answer-threshold", type=float, default=0.18)
    args = parser.parse_args()

    cases = json.loads((ROOT / args.dataset).read_text("utf-8"))["cases"]
    if settings.rag_backend == "numpy":
        backend = create_backend("numpy", index_path=ROOT / settings.kb_index_path)
    else:
        backend = create_backend("chroma", persist_dir=ROOT / settings.chroma_persist_dir,
                                 collection_name=settings.chroma_collection)
    retriever = KnowledgeRetriever(
        Embedder(settings.openai_api_key, settings.openai_base_url, settings.embedding_model),
        backend,
    )
    retriever.load()

    rows, reciprocal_ranks = [], []
    doc_stats = defaultdict(lambda: {"count": 0, "hit1": 0, "hit3": 0})
    for case in cases:
        relevant = set(case["relevant_chunk_ids"])
        hits = retriever.search(case["query"], top_k=max(3, args.top_k))
        ids = [hit.chunk.chunk_id for hit in hits]
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in relevant), None)
        rr = 1 / rank if rank else 0.0
        top_score = hits[0].score if hits else 0.0
        answerable = bool(relevant)
        predicted_answerable = top_score >= args.no_answer_threshold
        if answerable:
            reciprocal_ranks.append(rr)
            expected_doc = case["relevant_chunk_ids"][0].split("#", 1)[0]
            stats = doc_stats[expected_doc]
            stats["count"] += 1
            stats["hit1"] += int(rank == 1)
            stats["hit3"] += int(rank is not None and rank <= 3)
        rows.append({**case, "retrieved_chunk_ids": ids, "first_relevant_rank": rank,
                     "reciprocal_rank": rr, "top_score": top_score,
                     "predicted_answerable": predicted_answerable})

    total = len(rows)
    answerable_rows = [r for r in rows if r["relevant_chunk_ids"]]
    no_answer_rows = [r for r in rows if not r["relevant_chunk_ids"]]
    group_stats = {}
    for group in sorted({r.get("group", "unknown") for r in answerable_rows}):
        subset = [r for r in answerable_rows if r.get("group", "unknown") == group]
        group_stats[group] = {
            "count": len(subset),
            "recall_at_1": sum(r["first_relevant_rank"] == 1 for r in subset) / len(subset),
            "recall_at_3": sum(r["first_relevant_rank"] is not None and r["first_relevant_rank"] <= 3 for r in subset) / len(subset),
        }
    summary = {
        "total": total,
        "answerable": len(answerable_rows),
        "no_answer": len(no_answer_rows),
        "backend": settings.rag_backend,
        "embedding_model": settings.embedding_model,
        "recall_at_1": sum(r["first_relevant_rank"] == 1 for r in answerable_rows) / len(answerable_rows),
        "recall_at_3": sum(r["first_relevant_rank"] is not None and r["first_relevant_rank"] <= 3 for r in answerable_rows) / len(answerable_rows),
        "mrr": sum(reciprocal_ranks) / len(answerable_rows),
        "no_answer_threshold": args.no_answer_threshold,
        "no_answer_accuracy": sum(not r["predicted_answerable"] for r in no_answer_rows) / len(no_answer_rows),
        "answerable_coverage": sum(r["predicted_answerable"] for r in answerable_rows) / len(answerable_rows),
        "by_group": group_stats,
        "by_document": dict(doc_stats),
    }
    report = {"summary": summary, "cases": rows}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告已写入: {output}")


if __name__ == "__main__":
    main()
