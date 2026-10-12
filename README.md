# Memory-Agent V1.1

**基于 LangGraph、PostgreSQL 和 RAG 的可持久化记忆智能体**

Memory-Agent 是在开源项目 [langchain-ai/memory-agent](https://github.com/langchain-ai/memory-agent) 基础上扩展的智能体工程。项目将原始的简单记忆工具扩展为结构化记忆写入流水线，并结合 PostgreSQL/pgvector 知识库检索、查询路由、FastAPI 服务和 Docker Compose 部署。

项目面向两个实际需求：**跨会话保留用户偏好**，以及**在技术问答中按需联合检索用户记忆与外部知识库**。

> 当前版本：**V1.1**（本地功能验收版本）。已经对核心路由、记忆状态和部分异常路径进行实际测试。尚未将其定义为经过大规模性能评估的生产级服务。

## 功能概览

| 模块 | 已实现的能力 |
| --- | --- |
| LangGraph Agent | 状态图编排、运行时上下文、对话 Checkpoint |
| 结构化 Memory | Extractor → Gate → Resolver → Storage |
| 记忆冲突处理 | `ADD`、`UPDATE`、`NOOP`，过滤临时信息 |
| 长期记忆 | PostgreSQL Store 持久化；按 `user_id` 隔离；可跨 Thread 检索 |
| RAG | 文档加载、切分、Embedding、pgvector 检索、来源引用 |
| 查询路由 | `chat`、`memory`、`rag`、`memory_rag` |
| API | FastAPI `/agent/chat` 和 `/agent/health` |
| 保存确认 | 返回真实 `memory_write_results`；支持确定性 `memory_confirmation` 与 `memory_command` |
| 部署 | Docker Compose 编排 Agent、API、PostgreSQL 和 Redis |

## 系统架构

```text
客户端（HTTP / JSON）
          |
          v
FastAPI /agent/chat
          |
          v
LangGraph Agent Server
          |
          +--> Memory Write Pipeline（每轮评估）
          |      |
          |      +--> Structured Extractor
          |      +--> Memory Gate
          |      +--> Memory Resolver
          |      +--> PostgreSQL Store
          |                 |
          |                 +--> memory_write_results
          |
          +--> Query Router
                 |
                 +--> chat        : 普通对话
                 +--> memory      : 长期用户记忆检索
                 +--> rag         : pgvector 知识库检索
                 +--> memory_rag  : 记忆 + 知识库联合检索
                          |
                          v
                     LLM 生成回答
                          |
                          v
FastAPI 校验写入状态并生成 memory_confirmation
          |
          +--> memory_command=false : 保留自然语言回答
          +--> memory_command=true  : 按执行状态生成确定性确认
```

### 两类持久化数据

- **Checkpoint**：保存当前 Thread 的消息和运行状态，主要服务于同一会话的连续交互。
- **Long-term Store**：保存跨 Thread 使用的结构化用户事实和偏好，以用户身份进行隔离。

两者都可以存储在 PostgreSQL 中，但用途不同。**Memory Gate 拒绝写入长期记忆，并不意味着该信息完全不会进入对话 Checkpoint。** 不应把此项目当作专门的敏感凭据保管系统。

## 技术栈

- Python 3.11、LangGraph、LangChain
- DeepSeek 兼容接口（对话模型）
- PostgreSQL 16、pgvector（长期记忆和 RAG 存储）
- BAAI/bge-m3（RAG Embedding）、Qwen3-Embedding-8B（Memory Embedding）；两路向量空间独立使用
- FastAPI、LangGraph SDK
- Docker、Docker Compose、Redis

模型接口、密钥和数据库参数由实际环境配置提供；不应写死在源码或 README 中。

## 项目结构

```text
memory-agent/
├── compose.yaml                  # 四服务部署配置
├── Dockerfile.api                # FastAPI 镜像
├── Dockerfile.agent              # Agent 相关 Dockerfile
├── langgraph.json                # LangGraph 图和 Store 配置
├── pyproject.toml
├── fastapi_demo/
│   ├── main.py                   # FastAPI 应用
│   ├── agent_api.py              # /agent/* 接口
│   └── memory_confirmation.py    # 确定性记忆确认
├── src/memory_agent/
│   ├── graph.py                  # LangGraph 主流程
│   ├── state.py                  # State 定义
│   ├── memory_schema.py          # 结构化记忆数据模型
│   ├── memory_gate.py            # 写入决策
│   ├── memory_resolver.py        # 冲突解析
│   ├── store.py                  # 长期记忆 Store
│   ├── router.py                 # 查询路由
│   ├── embeddings.py             # Embedding 配置
│   └── rag/                      # RAG 加载、切分、索引、检索、生成
├── scripts/
│   └── ingest_knowledge_base.py  # 知识库导入入口
├── tests/                       # 当前测试代码
└── tests-yuan/                  # 保留的原项目测试
```

`knowledge_base/` 原始文档、`backups/` 数据库备份、`.env` 本地密钥文件及大型评测数据默认不纳入代码版本库。

## 快速开始

### 1. 环境准备

推荐 Windows + Docker Desktop，或具有 Docker Compose 的 Linux 环境。需要 Python 3.11 及已安装的 LangGraph CLI 来从源码构建 Agent 镜像。

获取代码并进入项目目录后，按实际部署配置准备 `.env`：

```powershell
Copy-Item .env.example .env
```

请检查 `.env.example` 是否覆盖当前 `compose.yaml`、模型接入与 Embedding 所需的全部配置。至少应检查数据库密码、对话模型服务、Embedding 服务所需的环境变量。**不要把真实 `.env` 提交到 GitHub。**

> **首次部署提示：** 当前已验证的是本地配置完整、数据库及镜像已准备好的环境。全新机器上的环境变量清单、PostgreSQL 初始化步骤及 LangGraph 服务授权要求，仍应在公开发布前与实际源码逐项核对。

### 2. 构建镜像

首先构建 Agent：

```powershell
langgraph build --api-version 0.16.0 -t memory-agent-langgraph-api:0.1.1
```

构建 FastAPI：

```powershell
docker build -f Dockerfile.api -t memory-agent-api:0.1.1 .
```

项目的 Compose 配置使用以上两个正式镜像名称；如果修改了镜像标签，应同步修改 `compose.yaml`。

本项目使用独立的 PostgreSQL 数据卷。**首次启动前**，如 `compose.yaml` 仍将该卷声明为 `external: true`，请先创建：

```powershell
docker volume create memory-agent-pgdata
```

### 3. 启动服务

```powershell
docker compose config -q
docker compose up -d --no-build
docker compose ps
```

服务健康后，本地入口为：

| 服务 | 地址 |
| --- | --- |
| FastAPI | `http://127.0.0.1:8004` |
| Agent Server | `http://127.0.0.1:2025` |
| PostgreSQL | 本机 `5433` 端口（Compose 内为 `db:5432`） |

健康检查：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8004/agent/health
```

> **数据初始化：** 第一次构建空数据库时，还需要确保长期记忆 Store 及 `rag_documents`、`rag_chunks` 等相关结构已经初始化。知识库文档和索引不会因启动容器而自动出现。请先检查项目内的初始化与导入脚本后再进行首次入库；此处不提供未经核实的脚本参数。

## API 使用示例

### 正常对话

不传 `thread_id` 会创建新对话。不传 `memory_command` 时默认是普通对话模式。

```powershell
$body = @{
    user_id = "demo_user"
    message = "请解释 Python 装饰器的工作原理。"
} | ConvertTo-Json

$result = Invoke-RestMethod `
    -Uri http://127.0.0.1:8004/agent/chat `
    -Method POST `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

$result | Format-List *
```

### 保存一条长期偏好

```powershell
$body = @{
    user_id = "demo_user"
    message = "请把我常用 VS Code 作为开发 IDE 保存为长期偏好。"
    memory_command = $true
} | ConvertTo-Json

$result = Invoke-RestMethod `
    -Uri http://127.0.0.1:8004/agent/chat `
    -Method POST `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

$result.answer
$result.memory_write_results
$result.memory_confirmation
```

同一用户后续重新发送请求时，可以不传 `thread_id` 来创建新 Thread，并要求 Agent 从长期记忆中查询偏好。

### 响应字段

| 字段 | 含义 |
| --- | --- |
| `thread_id` | 当前会话标识，继续原会话时原样传回 |
| `answer` | 最终返回给客户端的回答 |
| `route` | `chat` / `memory` / `rag` / `memory_rag` |
| `sources` | RAG 检索来源（普通聊天一般为空） |
| `memory_write_results` | 本轮记忆写入流程的程序执行状态 |
| `memory_confirmation` | 根据上述状态确定性生成的确认列表 |

支持的 Memory 状态：

| `status` | 含义 |
| --- | --- |
| `stored` | 新增成功 |
| `updated` | 已有记忆更新成功 |
| `unchanged` | 相同事实已存在，未重复写入 |
| `ignored` | Memory Gate 拒绝保存 |
| `none` | 本轮没有发生记忆写入 |

当 `memory_command=true` 时，API 将以程序执行结果生成保存确认，覆盖 LLM 自由生成的保存声明；**这并不会强制写入长期记忆**。当该参数为 `false` 时，系统仍保留正常 LLM 回答，不能保证模型绝不产生不准确的保存表述。

如果 LangGraph 未返回合法的 `memory_write_results`，FastAPI 会拒绝把该调用当作成功响应，并返回 HTTP 502。

## RAG 知识库

RAG 使用 PostgreSQL 中的两个主要数据表：

- `rag_documents`：文档级元数据。
- `rag_chunks`：文本 Chunk、元数据以及用于相似度检索的向量。

现有本地验证环境已导入 **15 份 LangChain / LangGraph 教学文档，合计 1,585 个 Chunk**。这是特定测试环境中的数据规模，不代表代码仓库自带这些文档，也不是系统容量上限。

RAG 检索结果会通过 `sources` 返回来源标题、路径、章节或页码以及检索分数，回答中可使用 `[S1]` 等来源标记。

原始文档、数据库备份和已经生成的向量索引不随源码分发。请仅导入有权使用的文档，并通过 `scripts/ingest_knowledge_base.py` 对应的实际参数完成初始化。

## 验证记录

以下是 V1.1 候选版本上已经执行并核对过的代表性测试，不等于所有测试文件都已通过，也不等于完成了长周期生产稳定性评估。

| 场景 | 已观察到的结果 |
| --- | --- |
| ADD | API 返回 `stored`；PostgreSQL 可查到新记录 |
| UPDATE | 返回 `updated`；同一 Memory Key 对应的值被更新 |
| NOOP | 返回 `unchanged`；未产生重复关系记录 |
| IGNORE | 临时验证码返回 `ignored`；长期 Store 中记录数为 0 |
| NONE | 普通对话返回 `none` |
| 跨 Thread 记忆 | 新 Thread 可读取已保存的 IDE 偏好 |
| RAG | 命中目标 LangGraph 记忆管理章节，返回 5 条来源 |
| Memory + RAG | 同时使用已更新的 PyCharm 测试偏好和知识库来源 |
| 错误处理 | 模拟数据库写入失败时 Storage 异常向上传递；FastAPI 模拟错误返回 HTTP 502 |
| 保存确认 | `memory_command=true` 可根据执行状态纠正错误的 LLM 保存声明 |

在现有本地运行环境中，`agent`、`api`、`db`、`redis` 四个容器均曾通过健康检查。

## 限制与后续工作

1. 当前的强一致性保存确认依赖客户端显式使用 `memory_command=true`。一般聊天模式仍允许 LLM 自由回答。
2. LangGraph Checkpoint 中保留的原始 AI 消息与经过 FastAPI 处理的 `answer` 不保证逐字一致；如需保证历史回放一致，应在 Agent 内部进一步约束或持久化最终对外回答。
3. 拒绝写入长期 Store 的临时信息仍可能存在于当前 Thread 的 Checkpoint。正式处理敏感信息前需要设计脱敏及保留策略。
4. 尚未完成可公开复现的全新环境安装验证、完整测试覆盖率统计及并发负载评估。

## 项目来源与致谢

本项目以 LangChain 团队的开源仓库 [langchain-ai/memory-agent](https://github.com/langchain-ai/memory-agent) 为学习与二次开发基础，保留了 LangGraph 智能体开发的核心理念，并在此基础上加入结构化记忆、冲突解析、RAG、FastAPI 与 Docker 部署等工程能力。

公开发布时请保留并遵守原项目实际适用的许可证与版权声明。README 中不应使用原仓库的 CI 状态徽章冒充当前项目的 CI 结果。
