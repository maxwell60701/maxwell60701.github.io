---
title: "AI 智能客服系统架构设计与实现"
date: 2026-04-17
---

## 1. 项目概述

本文档详细介绍一款基于检索增强生成（RAG）技术的智能客服系统的架构设计与实现方案。该系统通过自然语言处理技术，为企业提供智能化客户服务能力。

### 1.1 核心功能

系统提供三大核心功能：

- **智能问答**：基于知识库的智能问答服务，支持多轮对话
- **问题推荐**：根据用户问题智能推荐相关问题，提升咨询效率
- **知识库管理**：支持批量导入问答数据，自动向量化存储

### 1.2 技术选型

| 组件 | 技术栈 | 说明 |
|------|--------|------|
| 后端框架 | FastAPI | 高性能异步 API 框架 |
| 向量数据库 | Milvus | 开源向量数据库，支持亿级向量检索 |
| 大语言模型 | Ollama / DeepSeek | 本地或云端 LLM 部署 |
| Embedding | Ollama Embeddings | 本地 embedding 模型 |
| RAG 框架 | LangChain + LangGraph | 检索增强生成编排 |
| 对话历史 | MemorySaver | 会话状态持久化 |

---

## 2. 系统架构

### 2.1 系统架构

系统采用分层架构设计，主要包含以下四层：

**1. 客户端层**

用户通过 Web 或 App 客户端发起 HTTP 请求，与 API 服务层进行交互。

**2. API 服务层（FastAPI）**

核心服务层，处理三类主要请求：

- `POST /chat` - 智能问答接口，支持多轮对话
- `POST /suggest_questions` - 根据用户问题推荐相关问题
- `POST /ingest_qa_csv` - 批量导入问答数据到知识库

**3. 机器学习层（Ollama）**

提供本地 Embedding 模型服务，将文本转换为向量表示，用于向量检索。

**4. 数据存储层**

- **Milvus**：向量数据库，负责存储知识库的向量表示，支持高效的相似度检索
- **Memory Saver**：会话历史管理，支持多用户多轮对话的上下文记忆

### 2.2 组件说明

#### API 服务层（server.py）

FastAPI 应用，负责处理三类请求：

- `POST /chat` - 核心问答接口
- `POST /suggest_questions` - 问题推荐接口
- `POST /ingest_qa_csv` - 知识库导入接口

#### 向量检索层

使用 Milvus 向量数据库存储知识库的向量表示，支持高效的相似度检索。

#### LLM 层

支持多种大语言模型切换：

- `LLM_TYPE=0`：Ollama 本地 Qwen3 模型
- `LLM_TYPE=1`：Ollama Cloud GPT-OSS 模型
- `LLM_TYPE=2`：DeepSeek 云端 API
- `LLM_TYPE=3`：Ollama GPT-OSS 20B 模型
- `LLM_TYPE=4`：Ollama Gemma 31B 模型

---

## 3. 核心模块详解

### 3.1 智能问答流程

```
用户问题 ──► 检索向量库 ──► 获取相关上下文
                │
                ▼
         构建 Prompt ──► 调用 LLM ──► 返回答案
```

**关键代码实现**：

```python
# 构建检索链
chain = RunnableParallel({
    "context": RunnableLambda(lambda x:x["question"]) | app.state.retriever,
    "question": RunnableLambda(lambda x:x["question"]),
    "history": RunnableLambda(lambda x: get_by_session_id(x["session_id"]))
}) | prompt | app.state.chat_llm

# 添加对话历史支持
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
) | StrOutputParser()
```

### 3.2 对话历史管理

系统使用内存存储对话历史，支持跨请求的会话追踪：

```python
class InMemoryHistory(BaseChatMessageHistory):
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
        # 保留最近 MAX_HISTORY_MESSAGES 条消息
        self.messages = self.messages[-MAX_HISTORY_MESSAGES:]
```

通过 `openid` 区分不同用户会话，实现多用户并发支持。

### 3.3 知识库导入

支持从 CSV 文件批量导入问答数据：

```python
def build_docs_from_qa_csv(csv_path, question_col, answer_col, ...):
    # CSV 格式：问题列、答案列、来源列
    # 转换为 LangChain Document 对象
    content = "问题:" + q + "\n" + "答案:" + a
    metadata = {"source": src}
    return Document(page_content=content, metadata=metadata)
```

导入时自动去重，避免重复数据进入向量库。

---

## 4. 部署方案

### 4.1 Docker Compose 部署

项目提供完整的 Docker Compose 配置，包含所有依赖服务：

```yaml
services:
  app:
    # FastAPI 应用服务
    ports:
      - "8081:8000"
    depends_on:
      - ollama
      - milvus-standalone

  ollama:
    # 本地 LLM 服务
    image: ollama/ollama:0.13.0
    ports:
      - "11434:11434"

  milvus-standalone:
    # 向量数据库（嵌入式 ETCD）
    image: milvusdb/milvus:v2.6.4
    environment:
      - ETCD_USE_EMBED=true
      - DEPLOY_MODE=STANDALONE
    ports:
      - "19530:19530"

  attu:
    # Milvus Web 管理界面
    image: zilliz/attu:v2.6
    ports:
      - "8000:3000"
```

### 4.2 环境变量配置

```bash
# Ollama 服务地址
CUSTOMER_SERVICE_OLLAMA_HOST=http://localhost:11434

# Milvus 数据库地址
MILVUS_HOST=localhost
MILVUS_PORT=19530

# LLM 类型选择
LLM_TYPE=1

# DeepSeek API（如使用云端模型）
DEEPSEEK_API_KEY=your_api_key
```

### 4.3 启动命令

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f app
```

---

## 5. API 接口文档

### 5.1 智能问答

**接口地址**：`POST /chat`

**请求参数**：

```json
{
  "question": "如何重置密码？",
  "openid": "user123"
}
```

**返回结果**：

```json
{
  "answer": "您可以通过以下步骤重置密码..."
}
```

### 5.2 问题推荐

**接口地址**：`POST /suggest_questions`

**请求参数**：

```json
{
  "question": "账号登录失败"
}
```

**返回结果**：

```json
{
  "questions": [
    "如何修改绑定的手机号？",
    "账号被锁定怎么办？",
    "如何联系人工客服？"
  ]
}
```

### 5.3 知识库导入

**接口地址**：`POST /ingest_qa_csv`

**请求参数**：

```json
{
  "csv_path": "questionLibrary.csv",
  "collection_name": "customer_service_rag",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "source_prefix": "客服问答库"
}
```

---

## 6. 总结

本系统采用 RAG 架构，结合向量检索与大语言模型，实现了高效的智能客服能力。系统具备以下特点：

- **可扩展**：支持多种 LLM 模型灵活切换
- **易部署**：Docker Compose 一键部署
- **可维护**：模块化设计，代码清晰
- **高性能**：Milvus 向量检索，亚秒级响应

如需进一步了解项目细节，请参考源码注释或联系开发团队。
