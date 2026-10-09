"""全局枚举与常量。

这里定义的字符串值会跨 API / 数据库 / 前端三处出现，因此集中一处定义，
避免出现 "n3_cluster" 与 "n3_clustering" 这类拼写漂移。
"""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED)


class AgentStatus(StrEnum):
    """故事工坊的会话状态。

    **与 `RunStatus` 分开**，因为这里多了一个「没事发生」：一次运行要么没开始
    要么在跑，而 agent 会话在两次改稿之间就是闲着的。少这一个状态，前端只能
    用「事件流最后一条是不是终态」去反推，那正是最容易推错的地方。

    `failed` / `cancelled` 是**会话的**终态，不是某一轮的：进程被杀留下的
    半截会话不该继续被喂消息——它下一轮会带着一串谁也说不清的上下文去找模型。
    界面上给的是「重新开始」，不是「接着聊」。
    """

    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (AgentStatus.FAILED, AgentStatus.CANCELLED)

    @property
    def label(self) -> str:
        return {
            AgentStatus.IDLE: "待命",
            AgentStatus.RUNNING: "写作中",
            AgentStatus.FAILED: "已中断",
            AgentStatus.CANCELLED: "已停止",
        }[self]


class NodeKey(StrEnum):
    """流水线的 7 个节点。顺序即执行顺序。"""

    N1_VIDEO = "n1_video"
    N2_COMMENTS = "n2_comments"
    N3_CLUSTER = "n3_cluster"
    N4_PSYCH = "n4_psych"
    N5_RETRIEVAL = "n5_retrieval"
    N6_REASONING = "n6_reasoning"
    N7_LITERARY = "n7_literary"


class SourceKind(StrEnum):
    """这次运行的数据源是「抖音视频」还是「用户粘的一段文本」。

    落库位置是 `AnalysisRun.providers["source_kind"]`，**刻意不新加列**：
    项目没有 alembic（表由 `create_all()` 建，加列要手动 ALTER 现有库），
    而 providers 是 JSON、本来就是 per-run 的信息载体。

    读的时候一律走 `run_service.source_kind_of()`——判别式只有那一处，
    别处不要再自己判断。
    """

    DOUYIN = "douyin"
    TEXT = "text"


class TextMode(StrEnum):
    """纯文本数据源的切分方式（仅 source=text 时有意义）。"""

    LINE = "line"  # 一行一条评论
    PARAGRAPH = "paragraph"  # 按空行分段，一段一条（长文用）


class EventType(StrEnum):
    """SSE 事件类型。前端 store/applyEvent.ts 按此分派。"""

    RUN_STARTED = "run_started"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    PROGRESS = "progress"
    # 结构化产物增量：data.kind 指明是 video/comments/clusters/evidence/reasoning 等
    PARTIAL = "partial"
    # 文本流式增量：data.text 为增量片段，data.section 指明目标段落
    DELTA = "delta"
    SECTION_UPDATED = "section_updated"
    INTERVENTION = "intervention"
    WARNING = "warning"
    ERROR = "error"
    RUN_COMPLETED = "run_completed"
    RUN_CANCELLED = "run_cancelled"
    HEARTBEAT = "heartbeat"

    # ---- 故事工坊（`app/agent/`）------------------------------------
    # 与上面的流水线事件共用同一个信封（`RunEvent`），但**命名空间不同**
    # （Redis stream key 前缀 `run` vs `agent`），两端点也各有一套表。
    # 复用协议、不复用通道——两边的 seq 各自独立计数，混在一条流里
    # 会让刷新时的 `?since=` 游标失去意义。
    AGENT_STARTED = "agent_started"
    #: 一轮模型调用开始（data.round = 第几轮）。工具调用之间的静默期里，
    #: 前端靠它维持「还在动」的观感。
    AGENT_THINKING = "agent_thinking"
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_TOOL_RESULT = "agent_tool_result"
    AGENT_COMPLETED = "agent_completed"
    AGENT_CANCELLED = "agent_cancelled"

    # ---- 对话工坊（`app/scene/`）--------------------------------------
    # 第三条链路。同一套信封与 SSE 帧机，命名空间 `scene`，表也是新的。
    # 复用协议、不复用通道——理由与上面一行完全相同。
    SCENE_STARTED = "scene_started"
    #: 导演的安排（处境/各人目标/发言顺序）。开演时一次，给面板顶部的折叠区。
    SCENE_SETUP = "scene_setup"
    #: 轮到谁了（data.turn_index/speaker）。连续几个回合之间的静默期里，
    #: 前端靠它维持「还在动」的观感，也是重写就地进行时的起点。
    SCENE_THINKING = "scene_thinking"
    #: 一句台词定稿（落库之后才发）。改稿时是**替换**同一个 turn_index。
    SCENE_TURN = "scene_turn"
    #: 创作者的意见被受理。POST 一回来就发，不依赖导演调用成功——
    #: 否则「意见发出去了但界面什么都没发生」会让人以为没提交上。
    SCENE_NOTE = "scene_note"
    #: 导演的修改安排：改哪几回合、每回合怎么改。
    SCENE_REWRITES = "scene_rewrites"
    SCENE_COMPLETED = "scene_completed"
    SCENE_CANCELLED = "scene_cancelled"


class SectionKey(StrEnum):
    """右栏洞察输出的四个段落。"""

    PROFILE = "profile"  # 心理侧写
    MECHANISM = "mechanism"  # 科学机制
    ALLUSION = "allusion"  # 文学类比
    INSIGHT = "insight"  # 最终洞察

    @property
    def title(self) -> str:
        return {
            SectionKey.PROFILE: "心理侧写",
            SectionKey.MECHANISM: "科学机制",
            SectionKey.ALLUSION: "文学类比",
            SectionKey.INSIGHT: "最终洞察",
        }[self]


class AgentStage(StrEnum):
    """面板上那句「正在查询知识库…」。`status` 是机器读的，这个是给人看的。"""

    PREPARING = "preparing"
    THINKING = "thinking"
    TOOL = "tool"
    WRITING = "writing"

    @property
    def label(self) -> str:
        return {
            AgentStage.PREPARING: "正在准备材料…",
            AgentStage.THINKING: "模型正在思考…",
            AgentStage.TOOL: "正在查资料…",
            AgentStage.WRITING: "正在写…",
        }[self]


class SceneStage(StrEnum):
    """对话工坊面板上的那句「导演正在排戏…」。

    `status` 复用 `AgentStatus` 的**值**——idle/running/failed/cancelled 的
    语义完全一样，连同「终态是单向门」那条规矩。但 label 要另给一份：
    `AgentStatus.label` 会说「写作中」，用在戏上是错的。
    """

    DIRECTING = "directing"
    PERFORMING = "performing"
    REVISING = "revising"

    @property
    def label(self) -> str:
        return {
            SceneStage.DIRECTING: "导演正在排戏…",
            SceneStage.PERFORMING: "正在演…",
            SceneStage.REVISING: "导演正在看你的意见…",
        }[self]


class Library(StrEnum):
    """知识库分库。顺序用于检索配额的展示与合并。"""

    PSYCHOLOGY = "psychology"
    LITERATURE = "literature"
    POETRY = "poetry"

    @property
    def label(self) -> str:
        return {
            Library.PSYCHOLOGY: "心理学·神经科学",
            Library.LITERATURE: "古典文学",
            Library.POETRY: "诗词",
        }[self]


class ImportStage(StrEnum):
    """知识库导入的五个阶段。`status` 复用 `RunStatus`，这里只描述「走到哪一步」。

    进度带按阶段划分（reading 0–5 / chunking 5–15 / tagging 15–70 /
    embedding 70–90 / indexing 90–100）。打标占到一半以上，因为它是唯一
    要逐块调模型的一步——进度条的节奏必须反映这一点，否则前 15% 走得飞快、
    然后卡在 20% 一动不动，看起来像挂了。
    """

    READING = "reading"
    CHUNKING = "chunking"
    TAGGING = "tagging"
    EMBEDDING = "embedding"
    INDEXING = "indexing"

    @property
    def label(self) -> str:
        return {
            ImportStage.READING: "读取文件",
            ImportStage.CHUNKING: "切分正文",
            ImportStage.TAGGING: "AI 抽取标签",
            ImportStage.EMBEDDING: "生成向量",
            ImportStage.INDEXING: "写入知识库",
        }[self]


class EvidenceKind(StrEnum):
    PSYCHOLOGY = "psychology"
    LITERATURE = "literature"
    POETRY = "poetry"


class RetrievalPath(StrEnum):
    """证据来自哪条检索路径——用于证据卡展示与 trace 追溯。"""

    VECTOR = "vector"
    GRAPH = "graph"


class ClusterMethod(StrEnum):
    HDBSCAN = "hdbscan"
    KMEANS = "kmeans"
    NONE = "none"  # 样本过少，未做聚类


# 噪声簇的固定 cluster_key。HDBSCAN 的 -1 标签会被物化成这个真实簇，
# 否则边缘声音会从画像中凭空消失，让结论建立在有偏子集上。
NOISE_CLUSTER_KEY = "noise"
NOISE_CLUSTER_LABEL = "边缘声音"

# 超出 max_clusters 的尾部簇会被并入这个簇
TAIL_CLUSTER_KEY = "tail"
TAIL_CLUSTER_LABEL = "其他·长尾"

# 评论条数低于此值时不做聚类，直接整体作为一个簇交给 LLM 打标
MIN_COMMENTS_FOR_CLUSTERING = 12

# 噪声占比超过此值才值得物化成独立簇
NOISE_MATERIALIZE_RATIO = 0.15

# 余弦相似度高于此值视为复读并去重
DUPLICATE_COSINE_THRESHOLD = 0.95

# ----------------------------------------------------------------------
# 主题质量闸门
# ----------------------------------------------------------------------

#: 成员少于此数的簇不算一个「主题」，降级为噪声。
#:
#: 这条阈值不在原始设计里，是实测加上的：mock 嵌入没有语义几何
#: （本机三个 fixture 的两两余弦上限只有 0.34–0.40），HDBSCAN 会把评论切成
#: 「一个大块 + 一堆 2 条小簇」。不设闸门时中栏渲染成 1 张真卡片加 4 张
#: 两条评论的卡片，看起来像坏了。设为 3 之后，加闸门仍不足 2 个合格簇
#: 就走 KMeans 兜底。
MIN_THEME_SIZE = 3

#: 至少要有这么多个合格簇，才认为 HDBSCAN 的结果可用。
MIN_THEMES_KEPT = 2

#: 簇卡上展示的 TOP 关键词个数。用确定性 n-gram 统计算出，不调 LLM——
#: 卡片上除了模型给的标签，还该有一条可核查的原始线索，且没有模型时也算得出来。
KEYWORDS_TOP_N = 5

#: 每个簇取几条代表评论送进提示词，以及落库几条供前端展示。
REPRESENTATIVE_COMMENTS = 3
REPRESENTATIVE_STORED = 5

#: 代表评论按点赞数降序取，但太长的先截断——一条 500 字的评论会挤掉
#: 另外两条的信息量。
REPRESENTATIVE_MAX_CHARS = 120

#: HDBSCAN 的 min_cluster_size 由 n 推出，并钳在这个区间内。
HDBSCAN_MIN_SIZE_FLOOR = 3
HDBSCAN_MIN_SIZE_CEIL = 8

#: 降级重试时 min_cluster_size 减到这个值。
HDBSCAN_MIN_SIZE_RETRY = 2

#: 「只找到 1 个簇」在 n 达到这个数时视为**没找到结构**，改走 KMeans。
#: 12 条评论挤在一簇里可能是真的同质，50 条还挤在一簇基本只能说明
#: 聚类参数没调对——这时候给一张「全体」的卡片等于什么都没说。
SINGLE_THEME_SPLIT_AT = 20

# ----------------------------------------------------------------------
# 检索（Node 5）
# ----------------------------------------------------------------------

#: RRF 的平滑常数，取原论文的 60。
#:
#: 融合的是**名次**而不是分数，这不是审美选择而是实测结论：本机向量分
#: 落在 0.06–0.29、图谱权重落在 0.7–0.95，量纲完全不可比——任何加权求和
#: 都得先拍一个归一化系数，而那个系数没有任何依据。名次没有量纲。
#: 也因此**不能给向量检索设 `score_threshold`**：0.29 已经是不错的命中了。
RRF_K = 60

#: 每条查询在每个库里取回多少候选。融合前多取一些，配额再往下削——
#: 反过来（按配额取）会让回填无货可用。
QUERY_TOP_K = 8

#: 各深度的查询条数上限。簇多的时候查询会溢出，按权重砍到这里。
MAX_QUERIES: dict[str, int] = {"quick": 4, "standard": 8, "deep": 12}

#: 各库的配额，顺序与 `Library` 一致（psychology / literature / poetry）。
#:
#: 配额是必须的：纯按 RRF 排序时，939 个点里压倒性多数的文学与诗词
#: 会把 6 条心理学证据挤出榜单前 13 名，而心理学才是结论的支柱。
LIBRARY_QUOTA: dict[str, tuple[int, int, int]] = {
    "quick": (3, 2, 1),
    "standard": (6, 4, 3),
    "deep": (9, 6, 5),
}

#: 图谱路径单次取回多少条。
GRAPH_TOP_K = 6

#: 图谱路径最多发几条查询（簇情绪 → 意象 → 语料）。
#: 图上 `Emotion` 只有 24 个节点，而本机 mock 情绪词表里的 11 个词只有 6 个
#: 在图里——发多了只会得到一串空召回，白等网络往返。
MAX_GRAPH_QUERIES = 3

#: 概念桥最多发几条（心理概念 → 意象 → 文学语料）。
MAX_CONCEPT_BRIDGES = 2

#: 由心理学语料的 concept 桥接到意象时，取几个意象。
CONCEPT_BRIDGE_LIMIT = 3

#: 证据正文入库前截断到多少字。原文动辄上千字，全存进去只会让
#: 证据卡变成一堵墙，而卡片要的是「一眼看懂这条能佐证什么」。
EVIDENCE_TEXT_CAP = 400

#: 逐条发证据事件的间隔（毫秒）。仅 demo 模式——真实检索下这一串事件
#: 本来就是几百毫秒内连续到达的，这里只是把节奏放慢给人看。
EVIDENCE_STAGGER_MS = 80

# ----------------------------------------------------------------------
# 推理（Node 6）
# ----------------------------------------------------------------------

#: confidence 的三项权重：证据支撑 / 机制 / 典故。
#:
#: 刻意做成**可由落库字段复算**的公式，而不是让模型自评。模型的自我评分
#: 系统性偏高，而这个数字是给用户判断可信度用的——一个不可核查的数字
#: 放在那里，比没有更糟。
CONFIDENCE_WEIGHTS = (0.55, 0.30, 0.15)

#: 几步证据算「支撑度满分」。一条证据就下结论和一打证据下同一个结论，
#: 不该拿同样的分。
SUPPORT_FULL_MARKS = 2

#: 一步推理最多几条。
MAX_REASONING_STEPS = 5

# ----------------------------------------------------------------------
# 洞察生成（Node 7）
# ----------------------------------------------------------------------

#: 覆盖率低于这个值的段落重生成一次。
#:
#: 0.6 的依据是它**必须能在黄金路径上不被触发**：同一条 mock 数据上按段落
#: 覆盖率算是 0.32、按唯一 id 算是 0.46，两种口径都会让每次演示都重试一遍，
#: 把健康结果报成有问题。本定义（每次出现计一次）在同数据上是 1.0。
REGEN_COVERAGE = 0.6

#: 逐字输出的切块长度（字符）。取 40 是因为中文正文一行约 20–25 字，
#: 40 字约两行——比这更碎会像卡顿，更大就不像在「写」了。
DELTA_CHARS = 40

#: 遇到句末标点时，攒够这么多字就提前断开。太低会让正文被逗号切得稀碎，
#: 高于 `DELTA_CHARS` 则形同虚设。
DELTA_FLOOR = 12

#: 两块之间的间隔（毫秒，仅演示模式）。25ms × 40 字 ≈ 1600 字/分钟，
#: 略慢于人的阅读速度，能看清在写什么又不至于让人等。
DELTA_INTERVAL_MS = 25

# ----------------------------------------------------------------------
# 簇配色
# ----------------------------------------------------------------------

#: 六个簇色，按 order_index 取模分配。刻意放在后端：它是流水线的产物，
#: 跟着 `Cluster.color` 落库，前端只是把它画出来。前端再存一份的话，
#: 两边的取值迟早会漂移，而症状是「同一个簇在卡片和环形图里是两个颜色」。
#:
#: 选色原则（沿用前端 theme.ts 的原始说明）：同一明度带，保证各扇区视觉
#: 权重相等；噪声簇不用这里的颜色，它固定走灰色以表明「这不是一个真实主题」。
CLUSTER_PALETTE = (
    "#5b6abf",  # 靛
    "#c4703a",  # 赭
    "#4a8c7e",  # 青
    "#a8577e",  # 紫
    "#7a8b3c",  # 橄榄
    "#5f7fa8",  # 灰蓝
)
