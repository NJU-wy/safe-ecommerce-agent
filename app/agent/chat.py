import json
from typing import Optional

from openai import OpenAI

from app.agent.storage import delete_session, load_session, save_session
from app.agent.summarizer import summarize
from app.config.settings import settings
from app.config.model_options import completion_kwargs
from app.prompts.customer_service import SYSTEM_PROMPT
from app.schemas.response import CustomerServiceResponse, IntentType
from app.agent.tools.manager import ToolManager
from app.agent.refund_safety import (
    latest_user_text,
    refund_confirmation_required_result,
    secure_refund_arguments,
)
from app.agent.refund_workflow import RefundWorkflow
from app.agent.response_policy import build_customer_response, recent_user_context
from app.agent.tool_policy import select_tool_definitions, should_include_skill_catalog
from app.agent.tool_result_compactor import compact_tool_result


class EcomAgent:
    """集成 ReAct、工具调用、记忆与 Skill 的电商客服 Agent。"""

    def __init__(self, session_path: Optional[str] = None):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.session_path = session_path or settings.session_path
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_react_steps = settings.max_react_steps

        self.tool_manager = ToolManager(
            use_mcp=settings.mcp_enabled,
            mcp_server_url=settings.mcp_server_url,
        )

        from app.agent.memory import MemoryManager
        self.memory_manager = MemoryManager(
            client=self.client,
            model=self.model,
            user_id=settings.memory_user_id,
            memory_dir=settings.memory_dir,
            memory_enabled=settings.memory_enabled,
            max_ltm_facts=settings.max_ltm_facts,
        )

        if settings.memory_enabled:
            from app.agent.tools.memory_tool import set_memory_manager
            set_memory_manager(self.memory_manager)

        from app.agent.skills import SkillManager
        self.skill_manager = SkillManager(
            skills_dir=settings.skills_dir,
            enabled=settings.skills_enabled,
        )
        if settings.skills_enabled:
            from app.agent.tools.skill_tool import set_skill_manager
            set_skill_manager(self.skill_manager)

        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None
        self.refund_workflow = RefundWorkflow(
            settings.memory_user_id, settings.refund_audit_path
        )

        loaded = load_session(self.session_path)
        if loaded:
            self.summary = loaded["summary"]
            self.raw_messages = loaded["messages"]
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])
            if loaded.get("refund_workflow"):
                self.refund_workflow = RefundWorkflow.from_dict(
                    loaded["refund_workflow"], settings.memory_user_id,
                    settings.refund_audit_path,
                )

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    def chat(self, user_input: str) -> CustomerServiceResponse:
        """处理用户输入：ReAct 生成正文 → 确定性主/次意图元数据 → 返回结果。"""
        self.refund_workflow.observe_user(user_input)
        self.raw_messages.append({"role": "user", "content": user_input})

        final_text = self._react_loop()

        # 结构化元数据由可审计的业务规则生成，省去每轮第二次 LLM 调用。
        result = build_customer_response(recent_user_context(self.raw_messages), final_text)

        self.memory_manager.update_short_term(self.raw_messages[-6:])

        # `_react_loop` 已保存最终正文；不再重复保存包含完整 reply 的结构化 JSON。
        # 这能显著降低下一轮对话的历史 Token，同时响应对象仍完整返回给调用方。

        if len(self.raw_messages) > self.history_threshold:
            self._compress_history()

        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
            refund_workflow=self.refund_workflow.to_dict(),
        )
        return result

    def reset(self):
        self.raw_messages = []
        self.summary = None
        self.memory_manager.reset_short_term()
        self.refund_workflow = RefundWorkflow(
            settings.memory_user_id, settings.refund_audit_path
        )
        delete_session(self.session_path)

    def save(self) -> None:
        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
            refund_workflow=self.refund_workflow.to_dict(),
        )

    def close(self):
        self.memory_manager.consolidate_to_long_term(self.raw_messages, self.summary)
        self.tool_manager.close()

    def _react_loop(self) -> str:
        """ReAct 循环：调用 LLM → 执行工具 → 观察结果 → 重复，直到模型给出最终回答。"""
        for step in range(self.max_react_steps):
            messages = self._build_messages()
            visible_tools = select_tool_definitions(
                recent_user_context(self.raw_messages), self.tool_manager.tool_definitions
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=settings.max_response_tokens,
                **completion_kwargs(),
                # 空工具集时省略 tools 参数，兼容不接受 tools=[] 的供应商。
                **({"tools": visible_tools} if visible_tools else {}),
                # 数据类请求首步必须落到真实工具；后续步骤恢复 auto，避免死循环。
                **({"tool_choice": "required"} if visible_tools and step == 0 else {}),
            )
            choice = response.choices[0]
            assistant_msg = choice.message

            if assistant_msg.content:
                self._print_thought(assistant_msg.content)

            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                self.raw_messages.append({"role": "assistant", "content": content})
                return content

            msg_dict = {"role": "assistant", "content": assistant_msg.content}
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
            self.raw_messages.append(msg_dict)

            for tc in assistant_msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                if func_name == "apply_refund":
                    # 执行层安全边界：即使模型受提示词注入影响主动调用退款，
                    # 最新用户消息没有明确确认时也绝不进入真实工具。
                    func_args = secure_refund_arguments(
                        func_args, self.raw_messages, settings.memory_user_id
                    )
                    if not self.refund_workflow.authorize(
                        str(func_args.get("order_id") or ""),
                        latest_user_text(self.raw_messages),
                        str(func_args.get("idempotency_key") or ""),
                        str(func_args.get("reason") or ""),
                    ):
                        result_str = refund_confirmation_required_result()
                        self._print_observation(result_str)
                        self.raw_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str,
                        })
                        continue

                self._print_action(func_name, func_args)
                result_str = self.tool_manager.execute_tool(func_name, func_args)
                if func_name == "apply_refund":
                    try:
                        self.refund_workflow.record_result(json.loads(result_str))
                    except json.JSONDecodeError:
                        self.refund_workflow.audit("invalid_tool_result")
                self._print_observation(result_str)
                context_result = compact_tool_result(func_name, result_str)

                self.raw_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": context_result,
                })

        messages = self._build_messages()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=settings.max_response_tokens,
            **completion_kwargs(),
        )
        content = response.choices[0].message.content or ""
        self.raw_messages.append({"role": "assistant", "content": content})
        return content

    def _extract_structured_response(self, text: str) -> CustomerServiceResponse:
        """从最终文本中提取结构化元数据（意图、置信度等）。"""
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "基于以下客服回复内容，提取结构化信息。"
                            "reply 字段直接使用原文，不要修改或缩减。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                response_format=CustomerServiceResponse,
                **completion_kwargs(),
            )
            return response.choices[0].message.parsed
        except Exception:
            return self._extract_structured_fallback(text)

    def _extract_structured_fallback(self, text: str) -> CustomerServiceResponse:
        """当 response_format 不被 API 支持时，用 prompt 引导 JSON 输出。"""
        intent_values = ", ".join(f'"{e.value}"' for e in IntentType)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "基于以下客服回复内容，提取结构化信息并输出 JSON。\n"
                        "reply 字段直接使用原文，不要修改或缩减。\n\n"
                        "必须严格按照以下 JSON 格式输出（不要加 markdown 代码块）：\n"
                        "{\n"
                        f'  "intent": <从以下选择: {intent_values}>,\n'
                        '  "confidence": <0.0到1.0的浮点数>,\n'
                        '  "reply": <原文回复内容>,\n'
                        '  "requires_human": <true或false>,\n'
                        '  "follow_up_question": <追问问题或null>\n'
                        "}"
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            **completion_kwargs(),
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return CustomerServiceResponse.model_validate_json(raw)

    def _build_messages(self) -> list[dict]:
        system_content = SYSTEM_PROMPT
        context = recent_user_context(self.raw_messages)
        if (
            self.skill_manager
            and self.skill_manager.enabled
            and should_include_skill_catalog(context)
        ):
            system_content += self.skill_manager.build_catalog_prompt()

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]
        messages.extend(self.memory_manager.build_memory_prompt_sections())
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
                }
            )
        messages.extend(self.raw_messages)
        return messages

    def _compress_history(self) -> None:
        keep = self.history_keep_recent
        split = len(self.raw_messages) - keep
        while split > 0 and self.raw_messages[split].get("role") in ("tool",):
            split -= 1
        if split <= 0:
            return
        old_messages = self.raw_messages[:split]
        recent = self.raw_messages[split:]

        new_summary = summarize(
            client=self.client,
            model=self.model,
            old_messages=old_messages,
            prev_summary=self.summary,
        )
        self.summary = new_summary
        self.raw_messages = recent
        print(
            f"\n💾 [已压缩 {len(old_messages)} 条老消息 → summary "
            f"({len(new_summary)} 字)]\n"
        )

    def _print_thought(self, text: str) -> None:
        print(f"\n💭 [思考] {text}")

    def _print_action(self, func_name: str, func_args: dict) -> None:
        args_str = ", ".join(f"{k}={v!r}" for k, v in func_args.items())
        print(f"🔧 [调用工具] {func_name}({args_str})")

    def _print_observation(self, result: str) -> None:
        display = result if len(result) <= 300 else result[:300] + "..."
        print(f"📋 [工具结果] {display}")
