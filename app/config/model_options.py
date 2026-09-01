"""Provider-specific options shared by chat completion calls."""

from app.config.settings import settings


def completion_kwargs() -> dict:
    """Return provider-specific options for short deterministic Agent calls."""
    base_url = settings.openai_base_url.lower()
    model = settings.model_name.lower()
    if "api.deepseek.com" in base_url:
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if (
        "dashscope.aliyuncs.com" in base_url
        or ".maas.aliyuncs.com" in base_url
    ) and model.startswith("qwen"):
        # 百炼 OpenAI 兼容接口使用 enable_thinking 控制千问思考模式。
        # 客服工具调用强调低延迟和稳定 JSON，因此默认关闭思考模式。
        return {"extra_body": {"enable_thinking": False}}
    return {}
