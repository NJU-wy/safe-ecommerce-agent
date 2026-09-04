from __future__ import annotations


def escalate_to_human(reason: str, priority: str = "normal") -> dict:
    """记录可审计的人工升级事件；演示项目不连接真实工单系统。"""
    safe_priority = priority if priority in {"normal", "high", "urgent"} else "normal"
    return {
        "success": True,
        "status": "queued",
        "priority": safe_priority,
        "reason": reason,
        "message": "已创建人工客服转接请求，请保持当前会话在线。",
    }
