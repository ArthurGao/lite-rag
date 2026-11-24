# lite-rag S25：Lineage（溯源账本）设计

> 状态：草案，待 owner 确认（2026-11-24）。取代 `docs/LINEAGE_GUIDE.md` 中与本文冲突的部分：guide 是初始意图，本文是对照现有代码后的定稿方向。
> 实现时同步更新 `docs/lite-rag-design.md` §4（新表）、§9（配置）、§12（S25 阶段与验收）。

## 1. 意图

owner 希望 lite-rag 成为跨项目复用的通用 RAG 模块，并具备成熟 RAG 应有的**答案溯源**能力：

- 任何一个已索引的 chunk 都能查到它来自哪篇文档的**哪个版本**；
- 任何一篇文档都能查到它派生出的全部产物（影响分析）；
- 任何一次问答都能查到用了哪些 chunk（检索到 / 放进 prompt / 被引用），以及**当时这些 chunk 是否已经过期**，并能看到过期版本当时的原文。

最典型的用法：用户投诉"答案说签证费 100 元，实际是 120 元"——查 trace 能说清是"当时文档就写 100、之后才改"，还是检索 / 模型的问题。

记录"谁派生了谁"的那一层是**通用的**（不认识 RAG），将来给 TenderTED、Cairn、AdForge 等项目复用；RAG 专属的部分（chunk 怎么记、查询 trace）在 lite-rag 里。

## 2. 目标 / 非目标 / 反目标

**目标**
1. 通用账本包 `lite_rag.lineage`：记录产物（artifact）、版本、派生边；查上游、下游、某时刻的有效版本；清理过期原文。
2. lite-rag 的写入路径（upsert / delete / delete_kb / reindex）在**同一事务**里记账。
3. 查询 trace：`search` / `ask` 可选地记录四个阶段（retrieved / reranked / in_context / cited）用到的 chunk 版本。
4. 旧版本 chunk 保留原文，默认 90 天后清掉原文、保留 hash 和元数据行。

**非目标**
- 不由账本驱动计算（不做 guide §4 的 `run_step(fn)`、不做基于指纹的跳过）。lite-rag 已有：文档 hash 未变即跳过（`engine.py` `upsert_text`）、`embedding_cache_<dim>` 按 `(model_id, key)` 复用向量、lite-cache L1 复用 contextual / graph 的 LLM 调用、PG 外键级联删除向量。
- 不改 rule 3（换 embedding 模型 = 新 kb + `reindex`）和 rule 4（chunk id `"{doc_id}#{n}"` 稳定、upsert 整篇替换）。检索表只放当前数据，账本是旁路的只追加记录。
- 不做 run / step 表（guide §3 的 `lineage_steps` / `lineage_runs`）。产出方式（chunker 配置、模型）记在 artifact 的 `producer` 和 `metadata` 里。
- 不做可视化 UI、不做 OpenLineage 集成、不做 trace 聚合统计。
- 不拆独立 repo（触发条件见 §8）。

**反目标**（技术上能跑也算失败）
- 账本写入改变了 lite-rag 现有的任何检索或摄取结果（eval 数字变化、upsert 跳过逻辑变化）。
- 账本与索引不一致：索引里有的 chunk 在账本里找不到当前有效的 artifact，或反之。
- `lite_rag.lineage` import 了任何 `lite_rag.*` 的其他模块。
- trace 写入失败被静默吞掉（违背 rule 5 "no silent fallbacks"）。

## 3. 约束（来自现有代码与规则）

- 两个存储后端都要支持：PostgreSQL（默认，`stores/pg.py`，一次 upsert 一个事务、`_lock_kb` 按 kb 串行）和 Chroma + SQLite（`stores/chroma.py`，SQLite 事务 + `pending/ready` 状态，单进程）。账本表放在**同一个数据库**里（PG：同一 schema；Chroma 后端：同一个 `lite_rag.sqlite`），这样能进同一个事务。
- PG 迁移由 `db.migrate(conn, schema)` 执行（`migrations/000N_*.sql`），SQLite 由 `migrate_sqlite`（`migrations_sqlite/`）执行。
- 测试不调真实模型；PG 集成测试用 `LITERAG_TEST_DATABASE_URL`、schema `lite_rag_test`；新测试每个都要先见红。
- 每个 stage 走 close-out（simplify → Codex 只读审查 → 修复 → 全部检查重跑）。

## 4. 方案对比（owner 已选 A）

| | A. 账本（选定） | B. 驱动（guide 原版） | C. 只做 RAG 内置溯源 |
|---|---|---|---|
| 做法 | 通用层只记录和查询；lite-rag 流程不变，在已有事务里记账 | 通用层 `run_step(fn)` 算指纹决定是否计算；ingestion 改造成 step | `documents` 加版本、chunk 记来源版本、两张 trace 表 |
| 复用 | 可（账本不认识 RAG） | 可，且附带增量引擎 | 不可 |
| 代价 | 别的项目要自己做"跳过重算" | 与已有跳过 / 向量缓存 / lite-cache 重叠；ingestion 重写，风险大 | 违背"通用模块"目标 |

选 A 的理由：同时满足"通用"和"RAG 溯源"，不重复已有能力，不动 24 个 stage 积累的行为。artifact 带 `content_hash`，将来有不自带跳过逻辑的项目接入时，可在账本之上加 `run_step`，不用重做账本。

## 5. 通用层 `lite_rag.lineage`

### 5.1 概念

| 概念 | 含义 |
|---|---|
| artifact | 一个**不可变**的产物版本：源文档的某个版本、某个 chunk 的某个版本……都是 artifact。由 `(namespace, kind, key)` 标识逻辑身份，`content_hash` 标识内容 |
| 版本 | 同一 `(namespace, kind, key)` 的多个 artifact，按 `version` 递增；任一时刻最多一个 `active` |
| edge | `parent → child` 的派生关系，只记派生（kb 成员关系、查询消费不放这里） |
| 有效期 | `valid_from` / `ended_at`；`status` 为 `active` / `superseded` / `deleted` |

**决定：源文档不单独建表，作为 `kind` 的一种 artifact。** 理由：边只连 artifact，上下游递归查询只有一种节点；版本、有效期、删除对文档和 chunk 是同一套规则。guide §3 的 `lineage_documents` 字段（`source_uri`、`metadata`）放进 artifact 的 `key` / `metadata`。

**决定：所有派生关系都写边，不在 artifact 上存 `parent_id`。** 理由：lite-rag 里 embedding 不单独成为 artifact（见 6.1），剩下的 document → chunk 是一对多，每个 chunk 一条边，没有 guide 担心的一对一膨胀。

### 5.2 表（PG 写法；SQLite 用 TEXT 存 JSON 和时间，语义相同）

```sql
CREATE TABLE lineage_artifacts (
  id           TEXT PRIMARY KEY,           -- uuid4 字符串
  namespace    TEXT NOT NULL,              -- 调用方给定的隔离键；lite-rag 用 kb 物理键 "<ns>:<kb>"
  kind         TEXT NOT NULL,              -- 通用层不解释："document"、"chunk"……
  key          TEXT NOT NULL,              -- 逻辑身份：doc_id、chunk_id……
  version      INT  NOT NULL,              -- 同 (namespace, kind, key) 内从 1 递增
  content_hash TEXT NOT NULL,
  producer     TEXT,                       -- 怎么做出来的，如 "chunker:markdown@<digest>"
  payload      TEXT,                       -- 原文，可为 NULL；gc 后置 NULL
  metadata     JSONB NOT NULL DEFAULT '{}',
  status       TEXT NOT NULL DEFAULT 'active',   -- active / superseded / deleted
  valid_from   TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at     TIMESTAMPTZ,                -- superseded 或 deleted 的时刻
  UNIQUE (namespace, kind, key, version)
);
CREATE UNIQUE INDEX ON lineage_artifacts (namespace, kind, key) WHERE status = 'active';
CREATE INDEX ON lineage_artifacts (namespace, status);

CREATE TABLE lineage_edges (
  parent_id TEXT NOT NULL REFERENCES lineage_artifacts (id) ON DELETE CASCADE,
  child_id  TEXT NOT NULL REFERENCES lineage_artifacts (id) ON DELETE CASCADE,
  metadata  JSONB NOT NULL DEFAULT '{}',   -- 子节点在这个父节点里的位置等（如 chunk_id、offset）
  PRIMARY KEY (parent_id, child_id)
);
CREATE INDEX ON lineage_edges (child_id);
```

- `ended_at` 取代 guide 的 `superseded_at`，因为删除也用它。
- 边带 `metadata`：同一个产物可能出现在父节点的不同版本、不同位置（见 6.1），位置信息属于"这条派生关系"，不属于不可变的产物本身。
- 迁移文件放在包内 `lite_rag/lineage/migrations/`（PG）和 `migrations_sqlite/`，有自己的版本表 `lineage_schema_migrations`，由通用层的 `migrate(conn)` 执行。**lite-rag 侧（实现者的决定，审阅后定）**：`db.migrate`、`migrate_store` 和 Chroma store 打开时的自动迁移都在 lite-rag 自己的迁移之后调用它；`PgStore` 打开时"有未执行迁移就拒绝"的检查同时覆盖账本迁移。理由：llm-steward 的合约测试直接调用 `lite_rag.db.migrate`、进程内 provider 自己建 store，只挂在 `literag migrate` 上会让它们在第一次 upsert 时才失败。这样 `lineage.enabled: true` 作为默认值对库用户也安全，pin 升级时 llm-steward 不需要改代码。

### 5.3 API

通用层接收调用方的连接，在调用方的事务里工作，自己不开、不提交事务。两个实现：`PgLedger(conn)`（psycopg）和 `SqliteLedger(conn)`（sqlite3），满足同一个 `Ledger` Protocol。

```python
record(namespace, kind, key, content_hash, *, parents=(), producer=None,
       payload=None, metadata=None, edge_metadata=None) -> Artifact
    # 已有同 key 的 active 版本且 content_hash 相同 → 复用它，补上 parents 的边，返回它
    # 否则新建 version+1，旧 active 版本标 superseded（ended_at=now），写边，返回新的

retire(artifact_id, *, status="superseded"|"deleted") -> list[Artifact]
    # 结束这个 artifact，并级联结束它的后代中"已没有任何 active 父节点"的那些；
    # 返回被结束的清单（含自身）

active(namespace, kind, key) -> Artifact | None
as_of(namespace, kind, key, at: datetime) -> Artifact | None   # 某时刻有效的版本
get(artifact_ids) -> dict[str, Artifact]
upstream(artifact_id) -> list[Artifact]      # 递归向上，含边信息足以还原路径
downstream(artifact_id) -> list[Artifact]    # 递归向下（影响分析）
retire_namespace(namespace, status="deleted") -> int
gc(ended_before: datetime) -> int            # 已结束且 ended_at 早于该时刻的 artifact：payload 置 NULL
migrate(conn) -> list[str]
```

**核心规则："孤儿即结束"**：一个 artifact 只在它所有父节点都结束后才结束。这条规则让"文档出新版本、未变的 chunk 继续有效"自然成立（见 6.2），也让删除级联成立。

### 5.4 边界

- `lite_rag/lineage/` 只依赖标准库、psycopg（PG 实现）和 sqlite3；一个测试遍历它的源码 import，出现 `lite_rag.`（自身包除外）即失败。
- 不读 lite-rag 的 settings；表名前缀固定 `lineage_`，PG schema 由调用方连接的 `search_path` 决定。

## 6. RAG 层（lite-rag 里）

### 6.1 记什么

| kind | key | content_hash | producer | payload | 父节点 |
|---|---|---|---|---|---|
| `document` | `doc_id` | lite-rag 现有的 `content_hash(title, text, metadata, pages)` | — | 不存（全文已在 `documents.text`；旧版本原文由 chunk 原文覆盖） | 无 |
| `chunk` | `"{doc_id}/{text_hash}/{k}"` | `text_hash` = sha256(chunk 原文 `Chunk.text`) | kb 的 chunker 身份 digest | chunk 原文 | 包含它的每个文档版本 |

- **chunk 按内容认身份，不按位置（实现者的决定，审阅后定）。** 位置 `doc#n` 和 offset 会因为前面段落的增删而整体移动；contextual 开启时，每个 chunk 的 context 由整篇文档生成，任何修改都会改变它。如果把这些算进身份，改一个字就会让后面所有 chunk 出新版本，"只有那一段过期"就不成立。所以身份只看 chunk 原文：
  - `k` 是同一篇文档里同样原文的第几次出现（从 0 开始），用来区分重复的套话段落；
  - 位置信息（`chunk_id`、`n`、`start`、`end`、`page`、`parent_n`）和这个版本的 contextual context 写在**边的 metadata** 上：同一个 chunk 在 v1 是 `doc#4`、在 v2 是 `doc#5`，就是两条边各记各的；
  - 含义："过期" = 这段原文在当前文档版本里已经不存在。原文没变、只是位置或 context 变了，不算过期。
  - 代价：chunk 的 embedding 输入含 context 和标题，账本不能还原"当时被索引的完整文本"，只能还原原文和当时的 context（边上）。这与溯源的用途一致。
- namespace = kb 物理键（`<namespace>:<kb>`），所以不同 kb 的同名文档互不干扰；`reindex` 到新 kb 是另一套独立的 artifact。
- **embedding 不单独成为 artifact**：一个 kb 只绑定一个模型（rule 3），向量与 chunk 一一对应，chunk 的 `metadata` 记 `embedding`（`model_id`）、`dim`；contextual / graph 的模型也记在 metadata（若启用）。`upstream(chunk)` 即可回答"这个向量来自哪篇文档的哪个版本"。
- parent_child 的父块：本期不单独记（父块只用于展示，不参与索引）；chunk metadata 记 `parent_n`。

### 6.2 写入路径（都在 store 已有的事务里）

- **upsert（文档变了）**：`record(document)` → 新版本 v(n+1)，旧版本被 superseded；对新切出的每个 chunk `record(chunk, parents=[新文档版本], edge_metadata=位置与 context)`——原文相同的 chunk 复用旧 artifact 并补一条边，所以**只改了第 5 段时，只有原文变了的 chunk 出新版本**，前后段落的编号和 offset 移动不影响；最后 `retire(旧文档版本)`，按"孤儿即结束"，只有不再被新版本引用的旧 chunk 被 superseded。
- 注意：一个 chunk 的原文跨越被修改处时（切块边界随修改移动），它也会出新版本。这是真实的内容变化，不是误报。
- **upsert（文档未变）**：lite-rag 已在进事务前跳过，账本不动。
- **delete**：`retire(文档 active 版本, status="deleted")`，级联到它的 chunk。
- **delete_kb**：`retire_namespace(kb 物理键)`。
- **Chroma 后端**：账本写在 SQLite 那个事务里（`pending` 阶段）；写 Chroma 失败时现有逻辑会删掉这篇文档，账本随之记 `deleted`。原则：账本反映 store 的最终状态。
- **Store Protocol 不改签名**：store 实现已能取得 kb 信息（chunker、模型），记账所需的 producer / metadata 由 store 在事务内组装。

### 6.3 查询 trace

```sql
CREATE TABLE rag_query_traces (
  id            TEXT PRIMARY KEY,
  namespace     TEXT NOT NULL,           -- 引擎的 namespace
  kbs           JSONB NOT NULL,          -- 本次查询的 kb 名
  operation     TEXT NOT NULL,           -- 'search' | 'ask'
  query         TEXT,                    -- 按配置：原文 / sha256 / NULL
  retrieval     JSONB NOT NULL,          -- mode、top_k、rerank、where、route 选择
  model_id      TEXT,                    -- 回答模型（ask），来自 lite-model provenance
  usage         JSONB,
  cached        BOOLEAN,
  status        TEXT,                    -- Answer.status
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE rag_query_sources (
  trace_id     TEXT NOT NULL REFERENCES rag_query_traces (id) ON DELETE CASCADE,
  artifact_ns  TEXT NOT NULL,            -- kb 物理键
  artifact_key TEXT NOT NULL,            -- （实现：可为 NULL，见 §12）chunk 的账本 key "{doc_id}/{text_hash}/{k}"
  chunk_id     TEXT NOT NULL,            -- 检索时的位置，便于阅读
  stage        TEXT NOT NULL,            -- retrieved / reranked / in_context / cited
  rank         INT,
  score        DOUBLE PRECISION,
  PRIMARY KEY (trace_id, artifact_ns, chunk_id, stage)   -- chunk_id 只在 kb 内唯一
);
```

- 阶段定义：`retrieved` = 融合后进入候选池的 chunk（`pool` 个）；`reranked` = rerank 后的顺序（仅启用 rerank 时）；`in_context` = 最终返回 / 放进 prompt 的 hits；`cited` = `check_citations` 后保留的引用。parent_child 下记的是**命中的子 chunk**。
- **在检索那一刻钉住版本（实现者的决定，审阅后定）**：trace 不在写入时去查"当前 active 的 artifact"，而是用检索时读到的 chunk 原文（`StoredChunk.text`）直接算出账本 key。这样 `ask` 在检索和写 trace 之间（模型调用期间）文档被改，trace 记的仍是当时真正读到的那段文字。为此 `chunks` 表加一列 `lineage_key`，upsert 时算好写入（与账本开关无关，key 只由文档 id 和原文决定），检索读 chunk 时一并读出，trace 直接用它，不需要再推算 `k`。
- 因此 `rag_query_sources` 不对 `lineage_artifacts` 加外键：账本里查不到这个 key（kb 还没回填、或记账关闭时写入的）也能正常写 trace，`trace show` 把它显示为"未记录"。
- 过期判断（`trace show` 时算）：取该 key 在 `trace.created_at` 时刻有效的 artifact（`as_of`；实现改用 `started_by`，见 §12）。它之后被结束 → "当时有效、之后过期"，并显示它的原文（若未被 gc）；仍 active → "仍然有效"；找不到 → "未记录"。这是标记，不抛错。
- `Answer` 增加 `trace_id`（未开启时为 None）；HTTP `ask` 响应带上它。
- **search 的 trace（owner 决定，2026-11-24）**：现有 `search` / `search_kbs` 签名和行为不变，不记 trace。新增：
  - `search_traced(...)` / `search_kbs_traced(...)`：参数与原方法相同，返回 `Traced(hits: list[Hit], trace_id: str | None)`；记 `retrieved` / `reranked` / `in_context` 三个阶段，`operation='search'`。
  - `record_cited(trace_id, chunk_ids)`（实现：接收 Hit，见 §12）：自己生成答案的宿主（llm-steward 只调用 `search_kbs`）把最终引用的 chunk 补记为 `cited` 阶段；chunk_id 必须是该 trace `in_context` 里出现过的，否则抛 `InvalidRequest`。
  - 理由：llm-steward 消费 `list[Hit]`，不能改返回类型；`ask` 内部改为调用 traced 版本。
  - HTTP 本期不加 traced search 端点，llm-steward 走进程内调用。
- trace 写入失败时抛错（rule 5）。trace 写在检索读取之后的独立小事务里。
- 不记 latency：lite-rag 现有代码不测耗时，加上属于另一件事。

### 6.4 配置

```yaml
lineage:
  enabled: true            # 记账；关掉则写入路径不记，查询 API 报错
  keep_text_days: 90       # gc 清掉已结束 artifact 原文的期限
trace:
  enabled: false           # 默认关：会存问题原文
  query: text              # text / hash / none
  retention_days: 90       # gc 删除更早的 trace
```

`literag lineage gc` 一次处理两件事：清原文（`keep_text_days`）、删旧 trace（`retention_days`）。不自动定时运行，由使用方调度。被 trace 引用的 artifact 行永不删除，只清 `payload`。

### 6.5 已有数据的回填

现有 kb 的文档在账本里没有记录，而文档不变时 upsert 会跳过，永远不会补记（包括 llm-steward 现有的全部 kb）。未回填不会让任何功能报错：trace 照常写，相关来源显示"未记录"；下一次文档变化时自然开始记账。要补全历史，提供 `literag lineage backfill <kb>`：从 `documents` 和 `chunks` 现有行生成 v1 文档和 chunk artifact（chunk 的 context 已在 `chunks.context` 列，可还原被索引的文本）。对已有账本的文档是 no-op。`documents.text` 为 NULL 的老文档照样能回填（chunk 原文在 `chunks` 表里）。

### 6.6 CLI / HTTP

- CLI：`literag lineage show <kb> <doc_id>`（各版本及其 chunk）、`literag lineage upstream <artifact_id>`、`literag lineage downstream <artifact_id>`、`literag trace show <trace_id>`（各阶段 chunk、是否已过期、过期版本原文）、`literag lineage backfill <kb>`、`literag lineage gc`。
- HTTP：本期只在 `ask` 响应中返回 `trace_id`，search 端点不变；查询账本的 HTTP 接口不做（非目标，之后按需要加）。

## 7. 阶段

每个阶段结束都走 close-out，全部现有测试保持绿，`literag eval` 的数字不变（反目标 1）。

1. **S25a 通用账本**：`lite_rag/lineage/`、两套迁移、`PgLedger` / `SqliteLedger`、import 边界测试。合约测试对两个实现跑同一套用例。
2. **S25b 接入写入路径**：PG 与 Chroma store 的 upsert / delete / delete_kb 记账；`literag migrate` 调用账本迁移；backfill 命令。
3. **S25c 查询 trace**：两张表、`search_traced` / `search_kbs_traced` / `record_cited`、`ask` 埋点、`trace_id`、`trace show`。
4. **S25d gc 与收尾**：`lineage gc`、`lineage show/upstream/downstream`、README 写明第三方如何单独使用 `lite_rag.lineage`、更新 design §4 / §9 / §12。

## 8. 拆成独立 repo 的条件

出现任一即拆：TenderTED / Cairn / AdForge 等要使用账本；lite-mem 或 llm-steward 要记录自己的派生关系。拆分前提已由 5.4 保证，拆分 = 搬目录 + lite-rag 改 import + 发布 wheel + 改 pin。

## 9. 验收（自动化测试覆盖）

- 账本合约（PG、SQLite 各跑一遍）：`record` 相同内容复用、不同内容出新版本并 supersede 旧版本；"孤儿即结束"；`as_of` 返回指定时刻有效的版本；`upstream` / `downstream` 正确；`gc` 只清已结束且过期的 payload，行保留。
- 同一文档改一段（含让后面 chunk 编号和 offset 移动的修改、contextual 开启的情况）：只有原文变了的 chunk 出新版本，其他 chunk 仍是同一个 artifact，有两条父边（v1、v2），两条边上的 `chunk_id` / offset 各是当时的值。
- 文档里有两段相同原文：是两个 artifact（`k`=0、1），互不覆盖。
- 删除文档：文档和全部 chunk artifact 为 `deleted`；delete_kb 后该 kb namespace 下无 active artifact。
- 一致性：对已记账（写入时开启记账，或已回填）的文档，任一 upsert / delete 序列之后，store 里每个 chunk 都恰好对应一条从文档 active 版本出发的边，指向一个 active chunk artifact，且没有多余的 active artifact（PG 与 Chroma 后端都测）。
- ask 期间文档被修改（在检索之后、写 trace 之前改文档）：trace 记的是检索时读到的旧原文对应的 key，`trace show` 标为"当时有效、之后过期"。
- 未回填的 kb 开启 trace：`ask` 成功，来源显示"未记录"。
- `search` / `search_kbs` 在 trace 开启时也不写任何 trace 行，返回值不变；`search_kbs_traced` + `record_cited` 记下四个阶段；`record_cited` 传入不在 `in_context` 里的 chunk_id 时抛错，不写任何行。
- llm-steward 的方式（直接调 `lite_rag.db.migrate` 后建 store）也会建好账本表。
- Chroma 写入失败：账本中该文档为 `deleted`，与 store 一致。
- trace：能区分四个阶段；文档在 ask 之后被修改，`trace show` 标出被引用的 chunk "当时有效、之后过期"，并显示旧原文；`trace.enabled=false` 时不写任何行。
- 回填：对已有 kb 回填后满足一致性；重复回填 no-op。
- 边界：`lite_rag.lineage` 不 import lite-rag 其他模块。
- 行为不变：现有测试全绿；`-m ollama` 不要求，但 eval 配置跑 `FakeEmbedder` 的数字与改动前一致。

## 10. 留给实现者决定

- `PgLedger` / `SqliteLedger` 共享 SQL 的方式（参数风格适配或两份 SQL）。
- 递归查询的写法与深度保护。
- CLI 输出格式。
- 一致性测试是用随机操作序列（property-based）还是固定场景。
- llm-steward 改用 `search_kbs_traced` + `record_cited` 是 llm-steward 自己的后续工作（pin 升级之后），不在 S25 范围内。

## 12. 实现中的决定（2026-11-25，S25a–d）

实现时改动或补充了上文，以这里为准：

- **账本 API**：入口是 `migrate(conn)` / `pending(conn)` / `ledger_for(conn)`，按连接类型选 PG 或 SQLite（不是 `migrate_pg` 等四个函数）。另加 `record_many`（一组 chunk 一次检查父节点）、`count_active`（一致性检查）、`history`（某个 key 的全部版本）、`started_by`（某时刻已开始的最新版本，不论是否已结束）。PG / SQLite 的差异集中在 `lineage.dialect`，账本和 lite-rag 的 trace 表共用。
- **并发**：调用方必须串行化同一 namespace 的写入（lite-rag：PG 持有 kb 行锁，Chroma 是进程锁）。账本本身不加锁。
- **账本不记 contextual 上下文**：它是模型根据内容写的文字，存在边上会比删除活得久（终审 #1）。边上只记位置（`chunk_id`、`n`、`start`、`end`、`page`、`parent_n`）。
- **删除即清原文**：`deleted` 的产物立即清空 `payload` 和 `metadata`（文档的 title / source）（安全审查、终审 #1）；`superseded` 的保留到 `keep_text_days` 后由 gc 清。`trace show` 对已删除文档的 chunk 显示"过期"但没有原文。
- **文档 artifact 的 metadata** 记 `title`、`source`、`embedding`、`dim`；chunk artifact 的 metadata 为空（同一 kb 的 chunk 都一样，不重复记）。
- **backfill 修复关账本期间的删除**：账本里仍 active、store 里已没有的文档记为 `deleted`（终审 #2）。关账本期间整个 drop 掉的 kb 修不回来（store 里已无 kb，backfill 报 `KbNotFound`）：它的产物保持原样。
- **`chunks.lineage_key`**：写入时算好（与账本开关无关），检索时读出；trace 直接用它，不推算 `k`。
- **trace 存储**：`artifact_key` 可为 NULL（未记录）；`in_context` 不存，读取时取最后一个排序阶段（有 rerank 用 reranked，否则 retrieved）的前 `top_k`。Store 协议只有一个 `traces()`，返回在一个事务里的 `TraceLog`；读 trace 时 SQL 按 namespace 过滤。
- **trace 时间**：读完 chunk 之后立即取（不是模型回答之后）。`trace_report` 用 `started_by(key, 时间)` 找当时读到的版本：已结束即"过期"，仍有效即"有效"，找不到即"未记录"。
- **`record_cited(trace_id, cited: Sequence[Hit])`**：接收 Hit 而不是 chunk_id，因为 chunk_id 只在 kb 内唯一，跨 kb 的 trace 需要 kb 名。
- **gc**：`literag lineage gc` 不分 namespace，作用于整个 store；清原文和删 trace 都按 1000 条一批、每批一个事务（`lineage_artifacts_gc` 部分索引只含还有原文的已结束行）。`lineage.enabled: false` 时只删 trace，不动账本。不剪"稳定 chunk 指向旧文档版本"的边：那是来历本身，增长为每次编辑每个 chunk 一条边。
- **存储估算**：每条 trace 约 2×`top_k` 行（rerank 关，默认）或 2×`top_n`+`top_k` 行（rerank 开）。每天 1 万次、保留 90 天、rerank 开时，PG 约 5–10 GB。
- **合并前补充（2026-11-25）**：`record_cited` 每个 trace 只接受一次：它把 trace 的 status 从空（未结束的 search trace）条件更新为 `answered`，并在同一事务里写 cited 行，并发调用只有一个成功；`ask` 的 trace（含无引用、`no_sources`、`failed`）一律拒绝，传入空列表也算用掉这一次。`engine.purge_traces()` / `literag trace purge --yes` 分批删一个 namespace 的全部 trace（只删 trace，kb 用 `drop-kb` 删），索引 `ix_rag_query_traces_namespace (namespace, created_at, id)`（迁移 0009 / sqlite 0004）；`ask` 调用模型失败时也写 trace（`status: failed`，含检索阶段），然后照常抛出模型的错误（写 trace 本身失败时作为附注附在那个错误上，不替换它）；`record()` 也走 `record_many`（只有一条写入路径）：一次查出全部 key 的最新版本、分批结束被取代的、新版本与边各用一次 `executemany`（300 个 chunk 从约 900 次调用降到约 7 次）；内容没变的边不重写。
- **时钟**：版本时间取自写入方的时钟，trace 时间取自读取方的。共用一个 store 的多台主机需要时钟同步（NTP），否则可能把有效的 chunk 判为过期。不改用数据库时间，因为那样每次记账都要多一次查询。
- **未处理（不可达）**：文档 hash 不变但切出的 chunk 文本不同时，旧 chunk 不会结束。现在走不到这条路径（kb 的 chunker 固定，hash 覆盖全文）；要处理得给账本加"断开边"的接口。
