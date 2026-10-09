# 语料库

四个 JSONL 文件，每行一条。**磁盘上保留各领域的原始字段名**（人按领域习惯写最不容易出错），
加载时由 `app/kb/loader.py` 映射成统一的 `KBChunk`（`app/kb/schema.py`）。

| 文件 | 条数 | 内容 |
| --- | --- | --- |
| `psychology.jsonl` | 60 | 依恋理论、哀伤心理学、情绪科学、社会心理学、神经科学、正念与自我决定论 |
| `literature.jsonl` | 40 | 红楼梦、诗经、楚辞、史记、世说新语、归有光、纳兰性德…… `context` 字段是一段现代处境解读 |
| `poetry.jsonl` | 50 | 唐诗宋词，手工精选，每条带意象与情感标注 |
| `graph_edges.jsonl` | 43 | 手写的聚合边：意象↔情绪、意象↔意象、人物→意象、概念→意象 |

**这不是 PRD 里说的 50–100 本书 / 5–10 万 chunk。** 那是一个需要数月的语料工程，
不是能在一次交付里凭空生成的东西。这里交付的是：**完整的摄取管线 + 一份每个库都能
检索出真实结果的高质量种子语料**，外加 `expand-poetry` 从 chinese-poetry 批量扩充。
规模可以长，管线的正确性（差分、幂等、硬失败）是现在就必须对的。

## 字段

### psychology.jsonl

```json
{"chunk_id":"psy:attachment:bowlby:001","discipline":"依恋理论","concept":"依恋行为系统",
 "source":"《依恋与失落》第一卷·依恋","author":"John Bowlby","year":1969,
 "keywords":["依恋","分离焦虑","安全基地"],"text":"……"}
```

必填：`chunk_id` `discipline` `concept` `source` `author` `text`。
`source` 自动去掉书名号后存进 `work`；`·` 之后的部分（卷名）会被截掉，只留书名。

### literature.jsonl

```json
{"chunk_id":"lit:hongloumeng:zanghua:001","book":"红楼梦","chapter":"第二十七回",
 "character":"林黛玉","imagery":["落花","泪","春"],"emotion":["哀伤","自怜","无常"],
 "type":"诗词","text":"花谢花飞花满天……","context":"黛玉葬花时所吟……"}
```

必填：`chunk_id` `book` `character` `imagery` `emotion` `text`。
`context` 不参与嵌入——它是元评论，不是原文；嵌进去会把作者的解释
和作品的原文混在同一个向量里。

### poetry.jsonl

```json
{"chunk_id":"poem:lishangyin:jinse","poem_title":"锦瑟","author":"李商隐","dynasty":"唐",
 "imagery":["明月","泪"],"emotion":["哀伤","回忆"],"type":"七言律诗","text":"锦瑟无端五十弦……"}
```

必填：`chunk_id` `poem_title` `author` `dynasty` `imagery` `emotion` `text`。

### graph_edges.jsonl

```json
{"source":"落花","source_type":"Imagery","target":"无常","target_type":"Emotion",
 "relation":"ASSOCIATED_WITH","weight":0.9}
```

**这是关于文化惯例的断言，不是「某条语料同时提到两者」的自动推导。**
权重是人评的。允许的节点类型与关系名在 `loader.GRAPH_NODE_TYPES` /
`GRAPH_RELATIONS` / `neo4j_index._ALLOWED_PATTERNS` 里白名单化——
Cypher 的标签与关系名无法参数化，只能拼字符串，白名单是唯一的防线。

## 规则

- **`chunk_id` 必须唯一，且形如 `{lib}:{slug}:{seq}`。** 重复的 id 会让两条语料
  在 Qdrant 里互相覆盖，其中一条永远查不到——这种错误不会报任何警，所以加载层
  直接硬失败。
- **校验失败一律硬失败并报 `文件:行号`。** 跳过坏行的后果是知识库静默地少几条，
  检索偶尔找不到本该找到的典故，没人会发现。参见 `loader.py` 的模块注释。
- **单元素数组可以写成裸字符串**（`"imagery":"落花"`），加载层自动升级为列表。
  语料是手写的，为此丢一条不值得。
- **意象与情感标注是检索质量的主要杠杆。** `schema.text_for_embedding()` 会把它们
  一起嵌进向量——用户查的是「落花 哀伤」这种短标签，原文是「花谢花飞花满天」，
  只嵌原文两者在向量空间里离得很远。改语料时**标注比原文更重要**。

## 改完语料之后

```bash
python -m app.cli ingest-kb            # 增量：只重嵌改动过的（按 content_hash 判断）
python -m app.cli ingest-kb --reset    # 全量重建（换 embedding 模型后必须）
python -m app.cli ingest-kb --library poetry   # 只处理一个库
python -m app.cli kb-search "落花 哀伤"        # 向量路径 + 图谱路径并排验证
```

从语料里**删掉**一条 chunk 后重跑 `ingest-kb`，它会连同 Qdrant 点、Neo4j 节点、
登记行一起清掉（`_prune_orphans`）。不清理的话，那条语料仍能被检索到，
而用户无从解释为什么。

## 扩充

```bash
python -m app.cli expand-poetry --count 300 --dry-run   # 先看会收下什么
python -m app.cli expand-poetry --count 300
python -m app.cli ingest-kb --library poetry
```

数据来自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT）。
上游只有诗句，没有标注，所以脚本按一份简繁并列的词表打标，
**只收下意象与情感都认得出的那些**——检索完全依赖标注，灌进去无标注的原文
等于往库里加噪声。详见 `app/kb/expand.py`。
