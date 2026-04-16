+++
title = '文明6 RAG 知识库：架构设计与实现'
date = 2025-12-29T17:43:40+08:00
draft = false
tags = ['RAG', 'LangChain', 'Milvus', 'FastAPI', 'Ollama', 'Civ6']
categories = ['AI 应用']
summary = '基于文明6百科构建 RAG：路由判定、多路检索与生成，详解系统架构与核心设计。'
+++

## 1. 概述

本项目基于 [Civilopedia](https://www.civilopedia.net/zh-CN/) 构建《文明6》RAG 问答系统，支持用户询问游戏相关的各类问题。

**核心挑战**：游戏知识涉及多个领域（伟人、建筑、单位、奇观等），用户问题往往跨领域。例如问"秦始皇的特色单位是什么"，需要同时检索"领袖"和"单位"两个知识库。

**解决方案**：路由判定 + 多路检索。用户提问后，系统先判断涉及哪些知识库，再并行召回、聚合生成。

---

## 2. 整体架构

系统分为三大阶段：

| 阶段 | 职责 | 关键技术 |
|------|------|----------|
| 数据采集 | 抓取、解析、入库 | BeautifulSoup, LangChain |
| 检索生成 | 路由判定、多路召回、生成回答 | Ollama LLM, Milvus, LangChain |
| 服务部署 | API 服务化 | FastAPI, Docker Compose |

---

## 3. 数据采集层

### 3.1 采集策略

数据来源为 Civilopedia 网站，共 15 个采集器：

| 采集器 | 主题 | 示例 URL |
|--------|------|----------|
| fetch_great_person.ipynb | 伟人 | great_person_individual_albert_einstein |
| fetch_building.ipynb | 建筑 | building_factory |
| fetch_unit.ipynb | 单位 | unit_destroyer |
| fetch_wonder.ipynb | 奇观 | building_big_ben |
| fetch_religion.ipynb | 宗教 | religion_catholicism |
| fetch_leader.ipynb | 领袖 | leader_qin_shi_huang |
| ... | ... | ... |

### 3.2 解析与文档化

每个采集器的工作流程相同：

1. **构造 URL 列表**：根据主题构造待抓取页面 ID 列表
2. **发送请求**：requests.get 抓取页面
3. **解析 HTML**：BeautifulSoup 提取关键字段
4. **封装 Document**：LangChain Document 对象

**以伟人为例**，解析字段包括：

```
伟人姓名: 阿尔伯特·爱因斯坦
特色能力: 核裂变 · 理论科学家
身份: 伟人点数 +8 · 消耗 920 点伟人之力
```

### 3.3 文本分词

使用 `RecursiveCharacterTextSplitter` 进行分词：

- **chunk_size**: 1000 字符
- **chunk_overlap**: 200 字符

保证上下文窗口有足够重叠，便于检索时保持连贯性。

---

## 4. 向量存储层

### 4.1 Milvus Collection 设计

| Collection | 存储内容 | 示例字段 |
|------------|----------|----------|
| `belief` | 万神殿信仰 | 信仰名、效果 |
| `religion` | 宗教 | 宗教名、信徒领袖 |
| `country` | 文明国家 | 文明特性 |
| `leader` | 领袖 | 名称、特色能力 |
| `building` | 建筑 | 名称、成本、产出 |
| `wonder` | 奇观 | 名称、建造条件 |
| `resource` | 资源 | 类型、分布 |
| `citystate` | 城邦 | 名称、增益类型 |
| `district` | 城区 | 名称、建造条件 |
| `feature` | 地形 | 地形效果 |
| `great_person` | 伟人 | 名称、时代、特长 |
| `improvement` | 改良设施 | 名称、建造条件 |
| `moment` | 历史时刻 | 名称、触发条件 |
| `unit` | 单位 | 名称、强度、升级 |
| `unit_promotion` | 晋升 | 名称、效果 |

### 4.2 向量化

使用 `embeddinggemma` 模型进行向量化，存入 Milvus。

---

## 5. 检索生成层

这是系统的核心，分为三个步骤：

### 5.1 路由判定

**问题**：用户问题可能涉及多个领域，如何确定检索范围？

**方案**：让 LLM 自主判断。

```
用户问题 → gpt-oss 分析 → llama3.1 结构化输出 → 确定涉及的表
```

**RouteQuery 定义**：

```python
class RouteQuery(BaseModel):
    tables: List[Literal[
        "belief", "religion", "country", "leader", "building", "wonder",
        "resource", "citystate", "district", "feature", "great_person",
        "improvement", "moment", "unit", "unit_promotion"
    ]]
```

**判定示例**：

| 用户问题 | 判定结果 |
|----------|----------|
| "秦始皇的特色单位是什么" | `["leader", "unit"]` |
| "哪些领袖信仰天主教" | `["religion", "leader"]` |
| "科学类伟人有哪些" | `["great_person"]` |

**为什么二次调用**？
1. 第一次用 gpt-oss 输出自然语言判断
2. 第二次用 llama3.1 + `with_structured_output` 确保 JSON 稳定可解析

### 5.2 多路检索

**问题**：跨领域查询需要同时从多个 Collection 召回。

**方案**：为每个涉及的 Collection 创建独立检索器，并行执行。

```python
retriever_dict = {
    "leader": vectorstore.as_retriever(search_kwargs={"k": 100}),
    "unit": vectorstore.as_retriever(search_kwargs={"k": 100}),
}
RunnableParallel(retriever_dict)
```

**效果**：各检索器同时执行，结果自动聚合为上下文。

### 5.3 回答生成

**原则**：严格基于检索结果，禁止编造和省略。

**提示词约束**：

1. 只能用上下文中的内容
2. 按 source 逐条列出，不合并
3. 不省略、不推断

**RAG Chain**：

```python
RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough(),
}) | prompt | llm | StrOutputParser()
```

---

## 6. 服务接口

FastAPI 提供单一端点：

**POST /ask**

```json
// 请求
{ "question": "哪些领袖信仰天主教" }

// 响应
{
  "answer": "信仰天主教的领袖包括：\n\n**来源：** ...\n- 佩德罗二世\n- 凯瑟琳·德·美第奇\n..."
}
```

---

## 7. 部署架构

Docker Compose 编排四个服务：

| 服务 | 镜像 | 端口 | 职责 |
|------|------|------|------|
| app | 自定义 | 8081 | RAG API |
| ollama | ollama/ollama:0.13.0 | 11434 | LLM 推理 + Embedding |
| milvus | milvusdb/milvus:v2.6.4 | 19530 | 向量数据库 |
| attu | zilliz/attu:v2.6 | 8000 | Milvus 管理界面 |

**核心环境变量**：

| 变量 | 说明 |
|------|------|
| `CIVI6_OLLAMA_HOST` | Ollama 服务地址 |
| `MILVUS_HOST` | Milvus 服务地址 |
| `MILVUS_PORT` | Milvus 端口 |

---

## 8. 设计亮点

| 设计 | 意图 |
|------|------|
| **路由判定** | 按需检索，避免全量扫描，节省资源 |
| **结构化输出** | 二次调用确保 JSON 稳定可解析 |
| **多路并行** | RunnableParallel 实现真正并行 |
| **严格回答** | 提示词约束保证答案准确性 |
| **按需扩展** | 新增领域只需新建 Collection |
