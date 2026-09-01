from app.config.model_options import completion_kwargs
from app.config.settings import settings


def test_qwen_bailian_disables_thinking(monkeypatch):
    monkeypatch.setattr(
        settings,
        "openai_base_url",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setattr(settings, "model_name", "qwen3.7-plus")
    assert completion_kwargs() == {"extra_body": {"enable_thinking": False}}


def test_qwen_workspace_domain_is_supported(monkeypatch):
    monkeypatch.setattr(
        settings,
        "openai_base_url",
        "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setattr(settings, "model_name", "qwen3.7-plus")
    assert completion_kwargs() == {"extra_body": {"enable_thinking": False}}
