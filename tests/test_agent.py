"""
测试客服 Agent 的结构化输出能力

测试策略：真实调用 API，验证：
1. 返回类型是否正确（CustomerServiceResponse）
2. 意图识别是否准确
3. 多轮对话上下文是否保持
4. reset 功能是否正常
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.chat import EcomAgent
from app.config.settings import settings
from app.schemas.response import CustomerServiceResponse, IntentType

# 测试用例：(用户输入, 期望的意图类型列表)
TEST_CASES = [
    ("你好呀", [IntentType.GREETING]),
    ("我的订单 2024010112345 什么时候发货？", [IntentType.ORDER_QUERY]),
    ("这件衣服买了三天了想退货", [IntentType.RETURN_REQUEST]),
    # 仅表达质量不满但未明确提出投诉，归为投诉或售后都符合业务语义。
    ("你们这个商品质量也太差了吧！", [IntentType.COMPLAINT, IntentType.AFTER_SALE]),
    ("有没有什么优惠活动？", [IntentType.PROMOTION]),
    ("帮我推荐一款适合送女朋友的包包", [IntentType.PRODUCT_CONSULT]),
]


def test_structured_output(tmp_path, monkeypatch):
    """测试单轮对话的结构化输出"""
    monkeypatch.setattr(settings, "memory_enabled", False)
    agent = EcomAgent(session_path=str(tmp_path / "structured.json"))

    print("=" * 60)
    print("  测试 1：结构化输出 & 意图识别")
    print("=" * 60)

    passed = 0
    failed = 0

    for user_input, expected_intents in TEST_CASES:
        agent.reset()

        response = agent.chat(user_input)

        # 验证返回类型
        assert isinstance(response, CustomerServiceResponse), (
            f"返回类型错误: {type(response)}"
        )

        # 验证字段完整性
        assert response.reply, "reply 不能为空"
        assert 0.0 <= response.confidence <= 1.0, (
            f"confidence 越界: {response.confidence}"
        )
        assert isinstance(response.requires_human, bool)

        # 验证意图
        intent_ok = response.intent in expected_intents
        status = "PASS" if intent_ok else "FAIL"

        if intent_ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] 输入: {user_input}")
        print(f"  意图: {response.intent.value} (期望: {[i.value for i in expected_intents]})")
        print(f"  置信度: {response.confidence:.0%}")
        print(f"  回复: {response.reply[:80]}...")

    print(f"\n结果: {passed} 通过, {failed} 失败 / 共 {len(TEST_CASES)} 条")
    assert failed == 0


def test_multi_turn(tmp_path, monkeypatch):
    """测试多轮对话上下文保持"""
    print("\n" + "=" * 60)
    print("  测试 2：多轮对话上下文")
    print("=" * 60)

    monkeypatch.setattr(settings, "memory_enabled", False)
    agent = EcomAgent(session_path=str(tmp_path / "multi_turn.json"))

    # 第一轮：提出问题
    r1 = agent.chat("我想退掉上周买的那双运动鞋")
    print(f"\n[轮次1] 输入: 我想退掉上周买的那双运动鞋")
    print(f"  回复: {r1.reply[:80]}...")

    # 第二轮：基于上下文追问（不再提"运动鞋"，看模型是否记住）
    r2 = agent.chat("穿了一次，鞋底就开胶了")
    print(f"\n[轮次2] 输入: 穿了一次，鞋底就开胶了")
    print(f"  回复: {r2.reply[:80]}...")

    # 工具轨迹可能触发历史压缩，因此第一轮可能已进入 summary。
    assert r1.reply and r2.reply
    assert any(keyword in r2.reply for keyword in ("质量", "开胶", "退货", "退款"))
    assert agent.summary is not None or sum(
        m.get("role") == "user" for m in agent.raw_messages
    ) == 2
    print(
        f"\n[PASS] 两轮上下文已保留（raw={len(agent.raw_messages)}, "
        f"compressed={agent.summary is not None}）"
    )


def test_reset(tmp_path, monkeypatch):
    """测试 reset 功能"""
    print("\n" + "=" * 60)
    print("  测试 3：对话重置")
    print("=" * 60)

    monkeypatch.setattr(settings, "memory_enabled", False)
    agent = EcomAgent(session_path=str(tmp_path / "reset.json"))
    agent.chat("你好")

    assert any(m.get("role") == "user" for m in agent.raw_messages)
    assert any(m.get("role") == "assistant" for m in agent.raw_messages)

    agent.reset()
    assert agent.raw_messages == []

    print("\n[PASS] reset 后对话历史已清空，仅保留 system prompt")


if __name__ == "__main__":
    results = []
    results.append(("结构化输出 & 意图识别", test_structured_output()))
    results.append(("多轮对话上下文", test_multi_turn()))
    results.append(("对话重置", test_reset()))

    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("全部测试通过！")
    else:
        print("存在失败的测试，请检查。")
        sys.exit(1)
