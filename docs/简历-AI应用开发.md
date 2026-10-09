# 【姓名】

**求职意向：AI 应用开发工程师 / 大模型应用开发（LLM Application Engineer）**

📧 【邮箱】 ｜ 📱 【手机】 ｜ 💻 【GitHub / 博客】 ｜ 📍 【城市 · 可到岗时间】

---

## 个人简介

- 独立完成两个可运行的 LLM 应用：一个 **7 节点 LLM 工作流 + 双路混合检索（RAG）** 的内容洞察工作台，一个 **古典文学知识库 MCP 服务**（已作为工具接入前者）。
- 技术面覆盖「模型接入 → 检索 → 编排 → 事件推送 → 前端呈现 → 基础设施」全链路：Python / FastAPI / LangGraph / Qdrant / Neo4j / Redis Streams / React + TypeScript。
- 习惯把不变量写成测试：后端 535 条用例、另一个仓 91 条（含真起 MCP 子进程的协议级测试）、前端 62 条，集成测试在依赖缺失时显式跳过而不是假绿。

---

## 专业技能

| 方向 | 内容 |
|---|---|
| **大模型应用** | Prompt 工程与结构化输出校验（越界引用由 Python 丢弃）、多 provider 工厂 + 三级降级（真实 / mock / 硬失败）、embedding 结果缓存与哈希差分（重复导入零嵌入调用）、工作流式编排（LangGraph StateGraph，非自主 Agent）、逐字流式输出与重生成策略 |
| **检索增强（RAG）** | 段落优先切分 + 长段句装箱、bge-m3（1024 维）向量索引、Qdrant 相似检索、Neo4j 图谱检索（情绪 → 意象 → 作品）、**RRF 双路融合**、按库配额分配、引用覆盖率校验与自动重生成 |
| **MCP（Model Context Protocol）** | 自研 MCP server（6 个只读工具，stdio 传输）与 MCP client（stdio / streamable-http），工具 schema 与中文 description 设计、失败载荷识别、错误分档（503/504/502） |
| **后端** | FastAPI、SQLAlchemy 2.0 async、Alembic 迁移、pydantic v2 / pydantic-settings、SSE（手写帧 + 断线续传）、Redis Streams 事件总线、Typer CLI、structlog 结构化日志、pytest（含真连 PG / Qdrant 的集成测试） |
| **数据与基础设施** | PostgreSQL、Qdrant、Neo4j、Redis、Docker Compose；三层存储的一致性处理（向量库存正文、图库只存片段，避免双写漂移） |
| **前端** | React 19、TypeScript、Zustand、Tailwind CSS 4、Vite；SSE 手写解析 + 静默看门狗、事件 → 纯函数 reducer → 状态树的单向数据流 |
| **工程化** | ruff / tsc / vitest / Playwright 端到端、Docker 编排、跨仓 MCP 集成的联调与回归 |

---

## 项目经历

### 一、观心 · 抖音内容心理洞察工作台 ｜ 独立开发（后端 / 前端 / 数据 / 部署）

【起止时间】｜ 技术栈：Python 3.13 · FastAPI · LangGraph · PostgreSQL · Qdrant · Neo4j · Redis Streams · DeepSeek · bge-m3 · React 19 · TypeScript · Tailwind 4 · Docker Compose

**项目定位**：输入一条抖音链接或一段文本，自动完成「评论抓取 → 语义聚类 → 心理语义抽取 → 知识库检索 → 推理链 → 文学化洞察」，每条结论都可追溯到证据原文。控制流由人写死、模型只在固定点位被当函数调用的**工作流系统**。

**关键实现**

1. **编排与可恢复执行**：LangGraph `StateGraph` 定义 7 节点线性拓扑——节点只返回状态补丁，由 LangGraph 按 reducer 合并（`errors`/`warnings` 累加、`stale` 合并），`AsyncPostgresSaver` 逐节点把状态写进 PG（`thread_id = run_id`）；前端刷新 / 重启后恢复现场走的是**事件回放**（Streams + PG 事件表 + `?since=`），不依赖检查点。踩坑：Windows 默认 `ProactorEventLoop` 与 psycopg3 异步不兼容，检查点会**静默降级为内存**（只表现为「刷新后进度恢复不了」，不报错）——启动时强制 `SelectorEventLoop` 修复。

2. **事件流与断线续传**：进度事件双写 Redis Streams（`MAXLEN 2000`）+ PG `run_event`；SSE 端点语义是「补齐 `seq` 之后的一切再继续跟推」，Streams 被裁剪时回退 PG。前端不用 `EventSource`（重连不带 `since`），改为手写帧解析 + 40 秒静默看门狗兜底——起因是 vite dev 代理不转发上游断开，页面会永久卡在「分析中」。

3. **双路混合检索**：向量路（Qdrant：主题像什么）+ 图谱路（Neo4j：情绪借什么意象说）经 **RRF 融合**并按库配额分配；越界引用由 Python 校验层丢弃，引用覆盖率不达标自动重生成。

4. **知识库导入流水线**：epub（zipfile + html.parser 手写解析，零第三方依赖）/ txt（UTF-8 → GB18030 兜底）；段落优先切分、长段按句装箱、三道体量闸门；LLM 批量打标 + 规则兜底；bge-m3 按 `chunk_id` 哈希**差分嵌入**，重导入几乎零嵌入开销；三写（Qdrant / Neo4j / PG）后支持幂等回滚。

5. **两个 MCP 数据源**：第三方抖音源（字段映射表适配工具名与返回结构，识别「失败载荷伪装成成功」）+ 自研古典文学 MCP；统一 provider 抽象 + 三级降级（真实 / mock / 硬失败）。

**规模与质量**：后端 24 个 REST 端点 + SSE、11 张表，Python 1.7 万行 / 93 文件，前端 TS 6.6 千行 / 38 文件；**535 条 pytest**（含真连 PG / Qdrant / Neo4j 的集成测试，依赖缺失显式 skip）+ **62 条 vitest** + Playwright 端到端；知识库 2,025 个向量点、Neo4j 六类节点 3,000+；工程文档 `docs/运行流程与技术栈.md`。

---

### 二、古典文学知识库 MCP 服务 ｜ 独立开发（已作为工具接入上方的观心工作台）

【起止时间】｜ 技术栈：Python · SQLAlchemy 2.0 · SQLite · Alembic · MCP SDK（stdio）· pytest

**项目定位**：把古籍原文本加工成结构化知识库（书目 / 章 / 段 / 对话 / 人物），并以 **MCP Server** 的形式把「查阅原文」能力开放给任意 LLM 客户端——让模型不只能「检索相似片段」，还能读到逐字原文。

**关键实现**

1. **文本加工组件**：回目切分、段落切分、对话抽取（含说话人识别与人物别名归一）、人物解析，全部为可单测的纯函数组件。
2. **六个只读 MCP 工具**：书目 → 人物 → 对话原文 → 回目 → 整章原文 → 段落上下文，参数扁平、带中文 description，形成一条「从检索到原文」的完整链路；每段原文都带全书段号，使「对话 → 段落」可双向定位。渲染层输出 Markdown（给人 / 给 LLM 读），**刻意不返回 JSON**，避免下游把它当 API 解析。
3. **数据规模**：红楼梦 119 回、**107.9 万字**、7,702 个段落、15,523 条对话、82 个人物。
4. **踩坑与防护**：`passage.idx`（章内序号）与 `passage.global_idx`（全书序号）同名不同义，历史上出现过「96.2% 段落挂错章且不报错」的静默故障——用不变量测试与坐标隔离（整章按章内序、段落按全书序）钉死。

**规模与质量**：**91 条 pytest**，其中包含真起 stdio 子进程的 **MCP 协议级测试**（`tools/list` 数量断言、逐字原文断言、未知章号返回说明文字而非报错）。

---

## 教育经历

【学校 · 专业 · 学历 · 起止时间】
【主修课程 / GPA / 奖项——与技术相关的写，其余可省】

---

## 工作经历

【公司 · 职位 · 起止时间】
- 【职责与产出，尽量量化：负责的模块、性能 / 稳定性指标、协作规模】

---

## 可展示物

- **运行流程与技术栈文档**：`docs/运行流程与技术栈.md`（含时序图、逐节点技术选型、文件级定位、实测坑位清单）
- **界面截图**：`docs/screenshots/`
- **测试**：后端 535 条 / MCP 服务 91 条 / 前端 62 条，均可一条命令复现
- **可现场演示**：起 Docker 基础设施 → 起前后端 → 输入链接或粘贴文本 → 观察 7 节点进度与洞察输出；「📖 原文」入口可实时查阅两个知识库的原文
