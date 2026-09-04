"""评估执行器：编排沙箱用例、双层评分并聚合报告。

Evaluator 自己不碰 Agent，只负责：让 Sandbox 跑出 RunTrace，再用 metrics 对轨迹
打分（过程层 + 结果层，代码规则恒算、LLM judge 受 use_judge 控制），最后聚合。
多意图指标先逐用例累计标签级 TP/FP/FN，再在数据集级计算 Micro P/R/F1。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from app.evaluation import metrics
from app.evaluation.dataset import EvalCase
from app.evaluation.sandbox import Sandbox


@dataclass
class EvalResult:
    """单条用例的评估结果。所有评分维度 None 表示该用例未指定（不计入聚合）。"""

    case_id: str
    description: str

    # ---------- 过程指标 ----------
    tool_accuracy: float | None = None
    tool_precision: float | None = None
    tool_efficiency: float | None = None
    tool_call_budget_pass: bool | None = None
    sensitive_action_safety: float | None = None
    token_cost: int = 0
    token_pass: bool | None = None
    process_soundness: float | None = None  # judge 1-5 归一化到 0-1
    route_match: float | None = None

    # ---------- 结果指标 ----------
    intent_match: float | None = None
    multilabel_tp: int = 0
    multilabel_fp: int = 0
    multilabel_fn: int = 0
    keyword_coverage: float | None = None
    requires_human_match: float | None = None
    answer_quality: float | None = None  # judge 1-5 归一化到 0-1
    faithfulness: float | None = None  # 1.0 忠实 / 0.0 幻觉

    # ---------- 汇总 ----------
    process_score: float | None = None
    result_score: float | None = None
    passed: bool = False

    trace: dict = field(default_factory=dict)  # RunTrace 精简快照
    judge_reasons: dict = field(default_factory=dict)  # judge 文字理由
    error: str | None = None


def _avg(values: list[float | None]) -> float | None:
    """对非 None 项求均值；全为 None 返回 None。"""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


class Evaluator:
    """评估器：对一批用例执行沙箱运行 + 双层评分 + 聚合。"""

    def __init__(
        self,
        sandbox: Sandbox,
        client: OpenAI,
        model: str,
        use_judge: bool = True,
        pass_threshold: float = 0.6,
    ):
        self.sandbox = sandbox
        self.client = client
        self.model = model
        self.use_judge = use_judge
        self.pass_threshold = pass_threshold

    def run_case(self, case: EvalCase) -> EvalResult:
        trace = self.sandbox.run(case)
        res = EvalResult(case_id=case.id, description=case.description, trace=trace.to_dict())

        if trace.error:
            res.error = trace.error
            return res

        try:
            resp = trace.final_response
            reply = resp.reply if resp else ""
            actual_intent = resp.intent.value if resp else ""
            # intent 是兼容字段；多标签评估使用主意图与全部次要意图的并集。
            actual_intents = (
                [actual_intent] + [intent.value for intent in resp.secondary_intents]
                if resp else []
            )
            actual_human = resp.requires_human if resp else False
            last_input = case.turns[-1]
            called = trace.tool_call_names

            res.token_cost = trace.total_tokens

            # ---------- 过程指标（代码规则）----------
            res.tool_accuracy = metrics.tool_accuracy(case.expected_tools, called)
            res.tool_precision = metrics.tool_precision(case.expected_tools, called)
            res.tool_efficiency = metrics.tool_efficiency(case.min_tool_calls, trace.num_tool_calls)
            res.tool_call_budget_pass = metrics.tool_call_budget_pass(
                case.max_tool_calls, trace.num_tool_calls
            )
            res.sensitive_action_safety = metrics.forbidden_tool_safety(
                case.forbidden_tools, called
            )
            res.token_pass = metrics.token_cost_pass(trace.total_tokens, case.max_tokens)
            # expected_route 仅适用于多Agent；单Agent没有Router，不应被统一记为失败。
            res.route_match = (
                metrics.route_match(case.expected_route, trace.route)
                if self.sandbox.mode == "multi" else None
            )

            # ---------- 结果指标（代码规则）----------
            res.intent_match = metrics.intent_match(case.expected_intent, actual_intent)
            if case.expected_intents:
                res.multilabel_tp, res.multilabel_fp, res.multilabel_fn = (
                    metrics.multilabel_counts(case.expected_intents, actual_intents)
                )
            res.keyword_coverage = metrics.keyword_coverage(case.expected_keywords, reply)
            res.requires_human_match = metrics.requires_human_match(
                case.expected_requires_human, actual_human
            )

            # ---------- LLM judge ----------
            if self.use_judge:
                q_score, q_reason = metrics.judge_answer_quality(
                    self.client, self.model, last_input, reply, case.expected_keywords
                )
                res.answer_quality = q_score / 5.0 if q_score else None
                res.judge_reasons["answer_quality"] = q_reason

                f_score, f_reason = metrics.judge_faithfulness(
                    self.client, self.model, reply, trace.tool_observations
                )
                res.faithfulness = f_score
                res.judge_reasons["faithfulness"] = f_reason

                p_score, p_reason = metrics.judge_process_soundness(
                    self.client, self.model, last_input, called
                )
                res.process_soundness = p_score / 5.0 if p_score else None
                res.judge_reasons["process_soundness"] = p_reason

            # ---------- 汇总 ----------
            res.process_score = _avg([
                res.tool_accuracy, res.tool_precision, res.tool_efficiency,
                res.sensitive_action_safety,
                res.process_soundness, res.route_match,
            ])
            res.result_score = _avg([
                res.intent_match, res.keyword_coverage,
                res.requires_human_match, res.answer_quality, res.faithfulness,
            ])
            res.passed = self._decide_pass(res)

        except Exception as e:  # noqa: BLE001 —— 评分异常隔离到单条用例
            res.error = f"评分异常 {type(e).__name__}: {e}"

        return res

    def run_all(self, cases: list[EvalCase]) -> dict:
        results = [self.run_case(c) for c in cases]
        return self._aggregate(results)

    def _decide_pass(self, res: EvalResult) -> bool:
        """所有被指定的评分维度都 ≥ 阈值，且 token 不超预算，才算通过。"""
        if res.error:
            return False
        if res.token_pass is False:
            return False
        if res.tool_call_budget_pass is False:
            return False
        dims = [
            res.tool_accuracy, res.tool_precision, res.tool_efficiency,
            res.sensitive_action_safety, res.process_soundness, res.route_match,
            res.intent_match, res.keyword_coverage, res.requires_human_match,
            res.answer_quality, res.faithfulness,
        ]
        present = [d for d in dims if d is not None]
        return all(d >= self.pass_threshold for d in present) if present else True

    def _aggregate(self, results: list[EvalResult]) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        total_tokens = sum(r.token_cost for r in results)
        tp = sum(r.multilabel_tp for r in results)
        fp = sum(r.multilabel_fp for r in results)
        fn = sum(r.multilabel_fn for r in results)
        micro_precision = tp / (tp + fp) if tp + fp else 0.0
        micro_recall = tp / (tp + fn) if tp + fn else 0.0
        summary = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_process_score": _avg([r.process_score for r in results]),
            "avg_result_score": _avg([r.result_score for r in results]),
            "total_tokens": total_tokens,
            "avg_tokens_per_case": total_tokens / total if total else 0,
            "sensitive_action_safety_rate": _avg([
                r.sensitive_action_safety for r in results
            ]),
            "avg_tool_precision": _avg([r.tool_precision for r in results]),
            "intent_micro_precision": micro_precision,
            "intent_micro_recall": micro_recall,
            "intent_micro_f1": (
                2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall else 0.0
            ),
        }
        return {
            "summary": summary,
            "cases": [self._result_to_dict(r) for r in results],
        }

    @staticmethod
    def _result_to_dict(r: EvalResult) -> dict:
        return {
            "case_id": r.case_id,
            "description": r.description,
            "passed": r.passed,
            "process": {
                "tool_accuracy": r.tool_accuracy,
                "tool_precision": r.tool_precision,
                "tool_efficiency": r.tool_efficiency,
                "tool_call_budget_pass": r.tool_call_budget_pass,
                "sensitive_action_safety": r.sensitive_action_safety,
                "token_cost": r.token_cost,
                "token_pass": r.token_pass,
                "process_soundness": r.process_soundness,
                "route_match": r.route_match,
                "process_score": r.process_score,
            },
            "result": {
                "intent_match": r.intent_match,
                "multilabel_tp": r.multilabel_tp,
                "multilabel_fp": r.multilabel_fp,
                "multilabel_fn": r.multilabel_fn,
                "keyword_coverage": r.keyword_coverage,
                "requires_human_match": r.requires_human_match,
                "answer_quality": r.answer_quality,
                "faithfulness": r.faithfulness,
                "result_score": r.result_score,
            },
            "judge_reasons": r.judge_reasons,
            "trace": r.trace,
            "error": r.error,
        }
