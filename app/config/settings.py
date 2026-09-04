from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目配置，从 .env 文件读取"""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "qwen3.7-plus-2026-05-26"
    temperature: float = 0.7

    # ReAct 循环
    max_react_steps: int = 5
    # 客服正文上限；约束冗长表格、重复复述和无效结束语。
    max_response_tokens: int = 600

    # MCP 配置
    mcp_enabled: bool = False
    mcp_server_url: str = "http://127.0.0.1:9123/mcp"

    # RAG 配置
    embedding_model: str = "text-embedding-3-small"
    # qwen3.7-text-embedding 的 OpenAI 兼容接口单次最多接收 20 条文本。
    embedding_batch_size: int = 20
    kb_dir: str = "app/agent/rag/knowledge"
    # 向量后端：numpy（手写余弦，教学透明，零依赖，默认）/ chroma（向量数据库，生产代表，需 pip install chromadb）
    rag_backend: str = "numpy"
    # 百炼中文语义向量在金标集上实测最佳；BM25 与 hybrid 保留用于对照。
    rag_retrieval_mode: str = "semantic"
    rag_query_decomposition: bool = False
    rag_query_decomposer: str = "llm"
    rag_candidate_k: int = 5
    rag_context_k: int = 3
    rag_decomposition_model: str = "qwen3.7-plus-2026-05-26"
    rag_rerank_enabled: bool = False
    rag_rerank_model: str = "qwen3.7-text-rerank"
    # NumpyBackend 的 JSON 索引路径
    kb_index_path: str = "app/sessions/kb_index.json"
    # ChromaBackend 的持久化目录与 collection 名
    chroma_persist_dir: str = "app/sessions/chroma"
    chroma_collection: str = "ecom_kb"

    # Multi-Agent 配置
    multi_agent_enabled: bool = False

    # Memory 配置
    memory_enabled: bool = True
    memory_dir: str = "app/sessions/memory"
    memory_user_id: str = "default"
    max_ltm_facts: int = 50

    # Skill 配置
    skills_enabled: bool = True
    skills_dir: str = "app/agent/skills/definitions"

    # Evaluation 配置（离线评估工具，无聊天开关）
    eval_dataset_path: str = "app/evaluation/cases.json"
    eval_use_judge: bool = True  # 是否启用 LLM-as-judge（质量/幻觉/过程合理性）
    eval_pass_threshold: float = 0.6  # 单维度通过阈值（judge 归一化到 0-1 后比较）

    # 敏感操作审计与幂等账本
    refund_ledger_path: str = "app/sessions/refund_ledger.json"
    refund_audit_path: str = "app/sessions/refund_audit.jsonl"

    # 多轮对话管理
    session_path: str = "app/sessions/session.json"
    # 2026-09 优化：工具消息会快速增加消息条数，阈值 10 会在两轮内触发一次额外
    # 摘要模型调用。提高阈值后，普通 1-3 轮客服会话不会过早压缩。
    history_threshold: int = 30
    history_keep_recent: int = 8

    model_config = {"env_file": ".env"}


settings = Settings()
