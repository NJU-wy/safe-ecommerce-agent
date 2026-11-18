"""从失败报告提出固定规则，在独立留出集配对验证后再允许人工采用。"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from app.agent.strategies.policy import (  # noqa: E402
    RULES, current_policy, passes_gate, policy_scope, save_validated, switch_policy,
)
from app.config.settings import settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.evaluator import Evaluator, run_comparison  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402

DEFAULT_HOLDOUT = ROOT / "app" / "evaluation" / "evolution_holdout.json"


def read_report(path: Path, mode: str, baseline: tuple[int, tuple[str, ...]]) -> list[dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = report.get("protocol", {})
    expected = {
        "producer": "app.scripts.run_eval", "mode": mode,
        "model": settings.model_name, "policy_scope": policy_scope(mode),
        "policy_version": baseline[0], "policy_rules": list(baseline[1]),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"报告字段 {key} 与当前运行配置不一致，请重新运行评估")
    rows = report.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("开发报告缺少逐题结果")
    return rows


def propose(rows: list[dict], active: tuple[str, ...], threshold: float) -> tuple[str | None, dict]:
    evidence = {key: [] for key in RULES if key not in active}
    for row in rows:
        if row.get("passed") is not False or row.get("error"):
            continue
        process, result, trace = row["process"], row["result"], row["trace"]
        reasons = {
            "lookup_first": process.get("tool_accuracy") is not None
                            and process["tool_accuracy"] < threshold,
            "human_handoff": result.get("requires_human_match") is not None
                             and result["requires_human_match"] < threshold,
            "ground_reply": bool(trace.get("tool_calls")) and (
                (result.get("faithfulness") is not None and result["faithfulness"] < threshold)
                or (result.get("keyword_coverage") is not None
                    and result["keyword_coverage"] < threshold)
            ),
        }
        for key in evidence:
            if reasons[key]:
                evidence[key].append(row["case_id"])
    evidence = {key: ids for key, ids in evidence.items() if ids}
    selected = max(evidence, key=lambda key: len(evidence[key]), default=None)
    return selected, evidence


def validate(mode: str, baseline: tuple[int, tuple[str, ...]], selected: str,
             development_cases: list[str], holdout: Path) -> dict:
    cases = load_dataset(holdout)
    candidate_rules = (*baseline[1], selected)
    with tempfile.TemporaryDirectory(prefix="policy_eval_") as root:
        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        base_eval = Evaluator(
            Sandbox(mode, str(Path(root) / "base"), policy_override=baseline[1]),
            client, settings.model_name, use_judge=False,
            pass_threshold=settings.eval_pass_threshold,
        )
        candidate_eval = Evaluator(
            Sandbox(mode, str(Path(root) / "candidate"), policy_override=candidate_rules),
            client, settings.model_name, use_judge=False,
            pass_threshold=settings.eval_pass_threshold,
        )
        report = run_comparison(cases, base_eval, candidate_eval)
        client.close()
    errors = sum(report[side]["summary"]["errors"] for side in ("baseline", "candidate"))
    evidence = {
        "development_cases": development_cases,
        "paired": report["paired"],
        "errors": errors,
        "holdout_cases": [case.id for case in cases],
    }
    eligible = passes_gate(report["paired"], errors)
    version = save_validated(mode, baseline[0], candidate_rules, evidence) if eligible else None
    report.update(rule=selected, candidate_rules=list(candidate_rules), eligible=eligible, version=version)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="评测驱动的客服策略版本管理")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("propose", "validate", "adopt", "rollback", "status"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--mode", choices=["single", "multi"], default="single")
        if action in {"propose", "validate"}:
            cmd.add_argument("--report", type=Path, required=True)
        if action == "validate":
            cmd.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
            cmd.add_argument("--output", type=Path)
        if action == "adopt":
            cmd.add_argument("version", type=int)
        if action == "rollback":
            cmd.add_argument("version", type=int, nargs="?")
    args = parser.parse_args()
    try:
        if args.action == "adopt":
            switch_policy(args.mode, args.version)
        elif args.action == "rollback":
            active = current_policy(args.mode)[0]
            switch_policy(args.mode, None, rollback_from=args.version or active)
        baseline = current_policy(args.mode)
        result: dict = {"mode": args.mode, "active_version": baseline[0],
                        "active_rules": list(baseline[1])}
        if args.action in {"propose", "validate"}:
            rows = read_report(args.report, args.mode, baseline)
            selected, evidence = propose(rows, baseline[1], settings.eval_pass_threshold)
            result.update(rule=selected, text=RULES.get(selected), evidence=evidence)
            if args.action == "validate" and selected:
                result = validate(args.mode, baseline, selected, evidence[selected], args.holdout)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
