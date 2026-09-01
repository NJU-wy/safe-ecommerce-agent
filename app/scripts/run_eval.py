"""离线运行 Agent 评估体系。

用法：
  # 先跑确定性路径（只用代码规则，快、零额外 LLM 开销），验证沙箱与采集
  python app/scripts/run_eval.py --no-judge

  # 全量评估（含 LLM-as-judge：回答质量 / 幻觉 / 过程合理性）
  python app/scripts/run_eval.py

  # 评估 Multi-Agent 编排器，并把报告写到文件
  python app/scripts/run_eval.py --mode multi --output app/sessions/eval_report.json

  # 对已有规则报告的全部失败项 + 固定数量成功样本运行 Judge
  python app/scripts/run_eval.py --judge --select-from-report app/sessions/eval_report.json

流程：
  1. 加载数据集（cases.json）。
  2. 构建 Sandbox（隔离 session + 关记忆读 + 本地 mock 工具 + 给共享 client 插桩）。
  3. Evaluator 逐用例：沙箱跑用例 → 过程层 + 结果层双层评分。
  4. 打印双层报告表格 + token 汇总，可选写入 JSON。
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.evaluator import Evaluator  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402


def _fmt(value) -> str:
    """格式化评分单元格：None→"-"，bool→✓/✗，float→百分比。"""
    if value is None:
        return "  - "
    if isinstance(value, bool):
        return "  ✓ " if value else "  ✗ "
    return f"{value * 100:4.0f}%"


def _print_report(report: dict) -> None:
    cases = report["cases"]

    print("\n" + "=" * 78)
    print("  过程指标（工具准确率 / 调用效率 / token达标 / 过程合理性）")
    print("=" * 78)
    print(f"  {'用例':<22}{'召回率':>8}{'精确率':>8}{'调用效率':>8}{'token':>8}{'安全':>8}")
    for c in cases:
        p = c["process"]
        print(
            f"  {c['case_id']:<22}"
            f"{_fmt(p['tool_accuracy']):>8}{_fmt(p['tool_precision']):>8}"
            f"{_fmt(p['tool_efficiency']):>8}{p['token_cost']:>8}"
            f"{_fmt(p['sensitive_action_safety']):>8}"
        )

    print("\n" + "=" * 78)
    print("  结果指标（意图 / 关键信息完整性 / 转人工 / 回答质量 / 忠实度无幻觉）")
    print("=" * 78)
    print(f"  {'用例':<22}{'意图':>8}{'完整性':>8}{'转人工':>8}{'回答质量':>8}{'无幻觉':>8}")
    for c in cases:
        r = c["result"]
        flag = "" if c["passed"] else "  ❌"
        if c["error"]:
            print(f"  {c['case_id']:<22}  运行/评分异常: {c['error']}")
            continue
        print(
            f"  {c['case_id']:<22}"
            f"{_fmt(r['intent_match']):>8}{_fmt(r['keyword_coverage']):>8}"
            f"{_fmt(r['requires_human_match']):>8}{_fmt(r['answer_quality']):>8}"
            f"{_fmt(r['faithfulness']):>8}{flag}"
        )

    s = report["summary"]
    print("\n" + "=" * 78)
    print("  汇总")
    print("=" * 78)
    print(f"  通过率        : {s['passed']}/{s['total']}  ({s['pass_rate'] * 100:.0f}%)")
    print(f"  平均过程得分  : {_fmt(s['avg_process_score']).strip()}")
    print(f"  平均结果得分  : {_fmt(s['avg_result_score']).strip()}")
    print(f"  总 token 消耗 : {s['total_tokens']}（平均每用例 {s['avg_tokens_per_case']:.0f}）")
    print(f"  工具精确率    : {_fmt(s['avg_tool_precision']).strip()}")
    print(f"  敏感操作安全率: {_fmt(s['sensitive_action_safety_rate']).strip()}")
    print(
        "  意图 Micro P/R/F1: "
        f"{s['intent_micro_precision']:.1%} / {s['intent_micro_recall']:.1%} / "
        f"{s['intent_micro_f1']:.1%}"
    )


def main():
    parser = argparse.ArgumentParser(description="运行 Agent 评估体系")
    parser.add_argument(
        "--dataset", default=settings.eval_dataset_path,
        help=f"测试集 JSON 路径（默认: {settings.eval_dataset_path}）",
    )
    parser.add_argument(
        "--mode", choices=["single", "multi"],
        default="multi" if settings.multi_agent_enabled else "single",
        help="被测 Agent 模式（默认据 multi_agent_enabled）",
    )
    parser.add_argument(
        "--judge", dest="judge", action="store_true", default=settings.eval_use_judge,
        help="启用 LLM-as-judge（默认开）",
    )
    parser.add_argument(
        "--no-judge", dest="judge", action="store_false",
        help="只跑代码规则指标，不调 LLM judge（快、便宜）",
    )
    parser.add_argument("--output", default=None, help="将完整报告写入 JSON 文件")
    parser.add_argument(
        "--case-ids", default="",
        help="仅运行指定用例 ID，多个 ID 用英文逗号分隔",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="仅运行数据集前 N 条，用于低成本冒烟评估",
    )
    parser.add_argument(
        "--select-from-report", default="",
        help="从已有规则报告选择全部失败用例，并抽样成功用例",
    )
    parser.add_argument(
        "--success-sample", type=int, default=15,
        help="与 --select-from-report 配合使用的成功用例抽样数（默认 15）",
    )
    parser.add_argument("--seed", type=int, default=42, help="成功集抽样随机种子")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)

    print("=" * 78)
    print("  EcomGuard Agent · 评估体系")
    print(f"  模式      : {args.mode}")
    print(f"  数据集    : {dataset_path}")
    print(f"  LLM judge : {'开启' if args.judge else '关闭（仅规则）'}")
    print(f"  裁判模型  : {settings.model_name}")
    print("=" * 78)

    print("\n[1/3] 加载测试集...")
    if not dataset_path.exists():
        print(f"❌ 测试集不存在: {dataset_path}")
        sys.exit(1)
    cases = load_dataset(dataset_path)
    selection_info = None
    if args.select_from_report:
        source_path = (
            ROOT / args.select_from_report
            if not Path(args.select_from_report).is_absolute()
            else Path(args.select_from_report)
        )
        source = json.loads(source_path.read_text(encoding="utf-8"))
        failed_ids = [item["case_id"] for item in source["cases"] if not item["passed"]]
        passed_ids = [item["case_id"] for item in source["cases"] if item["passed"]]
        rng = random.Random(args.seed)
        sampled_ids = rng.sample(passed_ids, min(args.success_sample, len(passed_ids)))
        selected = set(failed_ids + sampled_ids)
        cases = [case for case in cases if case.id in selected]
        selection_info = {
            "source_report": str(source_path),
            "failed_count": len(failed_ids),
            "success_sample_count": len(sampled_ids),
            "seed": args.seed,
            "case_ids": [case.id for case in cases],
        }
        print(
            f"   选择规则失败 {len(failed_ids)} 条 + "
            f"成功抽样 {len(sampled_ids)} 条（seed={args.seed}）"
        )
    if args.case_ids:
        requested = {item.strip() for item in args.case_ids.split(",") if item.strip()}
        cases = [case for case in cases if case.id in requested]
        missing = requested - {case.id for case in cases}
        if missing:
            print(f"❌ 未找到用例: {', '.join(sorted(missing))}")
            sys.exit(1)
    if args.limit is not None:
        if args.limit <= 0:
            print("❌ --limit 必须大于 0")
            sys.exit(1)
        cases = cases[:args.limit]
    if not cases:
        print("❌ 测试集为空")
        sys.exit(1)
    print(f"   共 {len(cases)} 条用例")

    print(f"\n[2/3] 在沙箱中重跑测试集（{args.mode} 模式）...")
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    sandbox = Sandbox(mode=args.mode)
    evaluator = Evaluator(
        sandbox=sandbox,
        client=client,
        model=settings.model_name,
        use_judge=args.judge,
        pass_threshold=settings.eval_pass_threshold,
    )
    report = evaluator.run_all(cases)
    if selection_info:
        report["selection"] = selection_info

    print("\n[3/3] 生成评估报告...")
    _print_report(report)

    if args.output:
        out_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n   报告已写入: {out_path}")

    print("\n🎉 评估完成。")


if __name__ == "__main__":
    main()
