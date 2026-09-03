# EcomGuard Agent：求职项目说明

## 一句话介绍

EcomGuard Agent 是一个面向电商客服场景的安全型 LLM Agent，支持 Function Calling、RAG、Memory、MCP、单/多 Agent，并通过规则评估、多意图 Micro-F1 和 LLM-as-Judge 验证工具调用、回答质量、Token 成本与退款安全。

## 项目为什么值得展示

这个项目解决的不是“让模型能聊天”，而是三个更接近真实 LLM 应用的问题：

1. 模型是否会根据问题调用正确的订单、物流、商品或知识库工具。
2. 模型受到提示词注入时，是否仍然无法绕过退款确认和幂等控制。
3. 项目效果是否有数据、指标和可复现命令支持，而不是只展示几段成功对话。

## 技术方案

- 使用 ReAct 和 Function Calling 组织客服对话与业务工具。
- 通过动态最小工具集减少误调用和工具 Schema Token。
- 使用 NumPy/Chroma RAG 检索退换货、配送和会员政策。
- 提供售前、售后、投诉三个专业子 Agent，按主意图选择一个主责 Agent。
- 使用确定性策略输出主意图与次要意图，不增加额外 LLM 调用。
- 在 ToolManager 前增加敏感操作执行边界，阻止模型伪造退款确认。
- 使用稳定幂等键和持久化账本防止同一订单重复退款。
- 将退款建模为显式跨轮状态机，绑定订单级确认并记录可审计状态迁移。
- 建立沙箱轨迹采集、规则指标、Micro-F1 和 LLM-as-Judge 评估链路。
- 建立包含困难改写和无答案问题的 RAG 金标集，用 Recall@K、MRR 和拒识覆盖率判断优化方向。

## 可写入简历的版本

### 项目名称

**EcomGuard Agent｜安全、可评估的电商智能客服**

### 项目描述

基于 Python、Qwen、Function Calling 和 RAG 构建电商客服 Agent，覆盖订单、物流、商品咨询、退换货及投诉场景，并实现多 Agent 路由、会话记忆、MCP 工具接入和敏感退款安全控制。

### 简历要点

- 设计 ReAct＋Function Calling 客服工作流，根据用户上下文动态暴露最小工具集，并压缩工具结果和会话历史，将单 Agent 100 条评估的平均 Token 控制在 2,077，Token 达标率达到 100%。
- 实现退款明确确认、订单状态校验、执行层提示词注入拦截和持久化幂等账本；退款对抗用例定向复测安全率达到 100%。
- 将退款升级为显式跨轮状态机并输出审计轨迹，4 条换单、历史确认复用和 JSON 注入用例通过率 100%。
- 建立包含 100 条正常、边界、多轮及对抗用例的评估体系，覆盖工具召回率、精确率、调用效率、意图、转人工、Token 和安全指标；单/多 Agent 严格通过率分别为 63%/64%。
- 支持主意图与次要意图输出，在 6 条人工标注复合意图集上取得 Micro-Precision 92.3%、Micro-Recall 100%、Micro-F1 96.0%。
- 对 37 条规则失败用例和 15 条成功抽样运行 LLM-as-Judge，区分规则误判与真实质量问题，并定位工具过程合理性为主要优化方向。
- 将知识库扩展至 31 个 Chunk，构建 44 条困难检索金标集；测得 Recall@1 63.2%、Recall@3 76.3%、MRR 68.9%，定位口语召回为主要瓶颈，并据此优先选择语义 Embedding/查询改写而非盲目增加 Reranker。

## 面试讲解顺序

### 1. 业务问题

普通聊天机器人可能直接编造订单信息，或者把模型输出的 `confirmed=true` 当成真实授权。项目因此把查询、敏感操作和回答评估分成独立层。

### 2. 架构选择

用户输入先进入单 Agent 或 Router；Agent 只看到当前场景需要的工具。工具返回经过压缩后重新进入上下文，最终正文由模型生成，意图和转人工元数据由确定性策略生成。

### 3. 安全设计

退款确认不信任模型参数，只检查最新真实用户消息。未确认请求在 ToolManager 之前拦截；确认后再生成稳定幂等键并写入退款账本。

### 4. 评估设计

规则层检查工具、结果、Token 和安全；复合意图集计算 Micro-F1；Judge 检查回答质量、忠实度和过程合理性。三类评估使用不同数据口径，避免用一个通过率概括所有能力。

### 5. 结果与不足

多 Agent 结果得分略高，但平均 Token 和 LLM 调用次数也更高，因此项目没有默认追求更多 Agent。当前 Judge 与被测模型相同，下一步更有价值的工作是引入异构 Judge 或人工复核，而不是继续增加界面功能。

## 需要诚实说明的边界

- 订单、商品、物流和退款均为 Mock 数据。
- 6 条复合意图集适合验证实现链路，但规模仍小，不能代表生产分布。
- 100 条完整报告保留了安全修复前快照，修复结果由定向回归报告证明。
- 当前 Multi-Agent 根据主意图选择一个主责 Agent，不会为次要意图同时调用多个 Agent。
- 项目未接入真实身份认证、支付系统、数据库和生产监控平台。

## 推荐仓库信息

- Repository：`safe-ecommerce-agent`
- Description：`A safety-first, evaluation-driven ecommerce customer-service agent with Function Calling, RAG, multi-intent classification and idempotent refund workflows.`
- Topics：`llm-agent`、`function-calling`、`rag`、`multi-agent`、`llm-evaluation`、`prompt-injection`、`python`
