"""子 Agent 定义：每个子 Agent 有专属的 system prompt 和工具子集。

SubAgent 封装了一个轻量级 ReAct 循环，由 Orchestrator 调度执行。
"""

import json

from openai import OpenAI

from app.prompts.agents import COMPLAINT_PROMPT, POSTSALE_PROMPT, PRESALE_PROMPT
from app.agent.tools.manager import ToolManager
from app.config.model_options import completion_kwargs
from app.agent.refund_safety import (
    latest_user_text,
    refund_confirmation_required_result,
    secure_refund_arguments,
)
from app.agent.refund_workflow import RefundWorkflow
from app.config.settings import settings
from app.agent.response_policy import recent_user_context
from app.agent.tool_policy import required_tool_names, select_tool_definitions
from app.agent.tool_result_compactor import compact_tool_result


AGENT_CONFIGS = {
    "presale": {
        "name": "小夕-售前",
        "prompt": PRESALE_PROMPT,
        "tools": {"query_product", "search_knowledge", "list_user_orders", "load_skill", "escalate_to_human"},
    },
    "postsale": {
        "name": "小夕-售后",
        "prompt": POSTSALE_PROMPT,
        "tools": {
            "query_order", "query_logistics", "apply_refund",
            "list_user_orders", "search_knowledge", "load_skill", "escalate_to_human",
        },
    },
    "complaint": {
        "name": "小夕-投诉",
        "prompt": COMPLAINT_PROMPT,
        "tools": {"query_order", "query_logistics", "search_knowledge", "load_skill", "escalate_to_human"},
    },
}


class SubAgent:
    """专业子 Agent：拥有独立的 prompt 和工具子集，执行 ReAct 循环。"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tool_manager: ToolManager,
        client: OpenAI,
        model: str,
        temperature: float,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_manager = tool_manager
        self.client = client
        self.model = model
        self.temperature = temperature

    def handle(
        self, messages: list[dict], max_steps: int = 5,
        refund_workflow: RefundWorkflow | None = None,
    ) -> tuple[str, list[dict]]:
        """执行 ReAct 循环，返回 (最终文本, 新增消息列表)。"""
        new_messages: list[dict] = []
        working = list(messages)
        called_this_turn: set[str] = set()

        for step in range(max_steps):
            context = recent_user_context(working)
            visible_tools = select_tool_definitions(
                context, self.tool_manager.tool_definitions
            )
            missing_required = required_tool_names(context) - called_this_turn
            if missing_required:
                visible_tools = [
                    tool for tool in visible_tools
                    if tool["function"]["name"] in missing_required
                ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=working,
                temperature=self.temperature,
                max_tokens=settings.max_response_tokens,
                **completion_kwargs(),
                **({"tools": visible_tools} if visible_tools else {}),
                **({"tool_choice": "required"} if visible_tools and (step == 0 or missing_required) else {}),
            )
            assistant_msg = response.choices[0].message

            if assistant_msg.content:
                self._print_thought(assistant_msg.content)

            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                msg = {"role": "assistant", "content": content}
                new_messages.append(msg)
                return content, new_messages

            msg_dict: dict = {"role": "assistant", "content": assistant_msg.content}
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
            new_messages.append(msg_dict)
            working.append(msg_dict)

            for tc in assistant_msg.tool_calls:
                func_name = tc.function.name
                called_this_turn.add(func_name)
                func_args = json.loads(tc.function.arguments)
                if func_name == "apply_refund":
                    # 子 Agent 不可信：没有最新用户明确确认，就只返回安全观察，
                    # 不调用 ToolManager，因此不会产生退款副作用。
                    func_args = secure_refund_arguments(
                        func_args, working, settings.memory_user_id
                    )
                    if not refund_workflow or not refund_workflow.authorize(
                        str(func_args.get("order_id") or ""), latest_user_text(working),
                        str(func_args.get("idempotency_key") or ""),
                        str(func_args.get("reason") or ""),
                    ):
                        result_str = refund_confirmation_required_result()
                        self._print_observation(result_str)
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str,
                        }
                        new_messages.append(tool_msg)
                        working.append(tool_msg)
                        continue

                self._print_action(func_name, func_args)
                result_str = self.tool_manager.execute_tool(func_name, func_args)
                if func_name == "apply_refund" and refund_workflow:
                    try:
                        refund_workflow.record_result(json.loads(result_str))
                    except json.JSONDecodeError:
                        refund_workflow.audit("invalid_tool_result")
                self._print_observation(result_str)
                context_result = compact_tool_result(func_name, result_str)

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": context_result,
                }
                new_messages.append(tool_msg)
                working.append(tool_msg)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=working,
            temperature=self.temperature,
            max_tokens=settings.max_response_tokens,
            **completion_kwargs(),
        )
        content = response.choices[0].message.content or ""
        new_messages.append({"role": "assistant", "content": content})
        return content, new_messages

    def _print_thought(self, text: str) -> None:
        print(f"\n  💭 [{self.name}·思考] {text}")

    def _print_action(self, func_name: str, func_args: dict) -> None:
        args_str = ", ".join(f"{k}={v!r}" for k, v in func_args.items())
        print(f"  🔧 [{self.name}·调用工具] {func_name}({args_str})")

    def _print_observation(self, result: str) -> None:
        display = result if len(result) <= 300 else result[:300] + "..."
        print(f"  📋 [{self.name}·工具结果] {display}")
