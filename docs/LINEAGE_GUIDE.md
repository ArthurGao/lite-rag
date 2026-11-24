# Lite Lineage 实现指导（给 Claude Code）

> 目标读者：在 lite-rag 仓库中工作的 Claude Code。
> 用法：**先分析现有代码，提交一份实现计划给我确认，再动手。** 本文给的是设计意图和约束，具体落地要以仓库现状为准。

---

## 0. 背景与目标

lite-rag 要做成跨项目复用的通用 RAG 模块。现在缺两样能力：

1. **增量重处理（Incremental Reprocessing）**：源文档、chunk 策略、embedding 模型变了，只重算受影响的部分，不做全量重建。
2. **可追溯（Lineage / Provenance）**：任一产物（chunk、embedding、摘要、生成的页面）都能查到上游来源；任一源变更都能查到下游影响。

另外，查询侧要能回答：这次回答用了哪些 chunk、哪个文档版本、当时这些数据是否已经过期。

**非目标**：不做 OpenLineage/Marquez 那样的企业级 lineage 平台；不做可视化 UI；不做分布式调度。

---

## 1. 分层（硬约束）

```
lite-lineage  (通用层，不认识 RAG)
    ▲
    │ 依赖（单向）
lite-rag      (RAG 层：chunk/embed step、KB、查询 trace)
```

- 通用层放在 lite-rag 仓库内的独立 package，例如 `packages/lineage/` 或 `lite_rag/lineage/`，以仓库现有结构为准。
- **通用层禁止 import 任何 RAG 代码。** `artifact.kind` 只是字符串（`"chunk"`、`"embedding"`、`"summary"`……），通用层不解释它。
- 将来会被其他项目复用（TenderTED、Cairn 签证材料核查、AI 学习系统、AdForge），所以接口设计要按"通用派生数据管道"来想。等第二个项目接入后，再把它拆成独立 repo。

---

## 2. 核心概念

| 概念 | 含义 |
|---|---|
| document | 原始输入（文件、URL、DB 记录），按 `content_hash` 区分版本 |
| step | 处理步骤的**定义**：name + version + config_hash（如 `chunk@v2`、`embed:bge-m3`） |
| processing_run | step 的**一次执行**：状态、时间、错误 |
| artifact | 某次执行产出的**不可变**产物，带 fingerprint |
| lineage_edge | 产物级的"输入 → 输出"派生关系 |

### 关键原则

1. **artifact 不可变**。内容变了就生成新的 artifact，旧的标记为 superseded。
2. **fingerprint 驱动增量**：
   ```
   fingerprint = hash(step.name + step.version + step.config_hash
                      + sorted(所有输入 artifact/document 的 content_hash))
   ```
   执行前先按 fingerprint 查询：已存在且 active 就直接复用（跳过）；不存在才计算。这和 Bazel、Nix 的缓存是同一个思路。
3. **run_id 管批次，lineage_edges 管单个产物**。一次 run 处理 1000 个文档、产出 5 万个 chunk 时，只靠 run_id 分不清哪个 chunk 来自哪个文档，所以必须有边。
4. **lineage_edges 只记录派生关系**。下面两种关系**不放进** lineage_edges：
   - KB 成员关系 → `kb_members`
   - Agent 或查询的消费关系 → `rag_query_sources`
5. **有效期**：artifact 带 `valid_from` 和 `superseded_at`，这样才能判断"查询那一刻用的数据是否已经过期"。

---

## 3. 表结构（通用层）

> 以 Postgres 为参考写法。如果 lite-rag 当前用 SQLite 或需要本地模式，请保持两者都能跑（递归 CTE 两边都支持；JSONB 在 SQLite 里退化为 TEXT JSON）。先确认仓库现有的 DB 访问方式（ORM 或原生 SQL），沿用它。

```sql
CREATE TABLE lineage_documents (
  id            UUID PRIMARY KEY,
  namespace     TEXT NOT NULL,          -- 租户/项目隔离
  source_uri    TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  version       INT  NOT NULL,
  status        TEXT NOT NULL DEFAULT 'active',  -- active/superseded/deleted
  metadata      JSONB,
  valid_from    TIMESTAMPTZ NOT NULL DEFAULT now(),
  superseded_at TIMESTAMPTZ,
  UNIQUE (namespace, source_uri, version)
);

CREATE TABLE lineage_steps (
  id          UUID PRIMARY KEY,
  name        TEXT NOT NULL,
  version     TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  config      JSONB,
  UNIQUE (name, version, config_hash)
);

CREATE TABLE lineage_runs (
  id          UUID PRIMARY KEY,
  step_id     UUID NOT NULL REFERENCES lineage_steps,
  status      TEXT NOT NULL,            -- running/succeeded/failed
  started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  error       TEXT,
  stats       JSONB                     -- added/skipped/failed 计数
);

CREATE TABLE lineage_artifacts (
  id            UUID PRIMARY KEY,
  namespace     TEXT NOT NULL,
  kind          TEXT NOT NULL,
  run_id        UUID REFERENCES lineage_runs,
  content_hash  TEXT NOT NULL,
  fingerprint   TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'active',  -- active/superseded/deleted
  payload_ref   TEXT,                   -- 指向向量 ID / 对象存储 / 文本表
  metadata      JSONB,                  -- page、offset 等
  valid_from    TIMESTAMPTZ NOT NULL DEFAULT now(),
  superseded_at TIMESTAMPTZ,
  UNIQUE (namespace, fingerprint)
);

CREATE TABLE lineage_edges (
  parent_id   UUID NOT NULL,
  parent_type TEXT NOT NULL,            -- 'document' | 'artifact'
  child_id    UUID NOT NULL REFERENCES lineage_artifacts,
  run_id      UUID REFERENCES lineage_runs,
  PRIMARY KEY (parent_id, child_id)
);
CREATE INDEX ON lineage_edges (child_id);
CREATE INDEX ON lineage_edges (parent_id);
```

**设计取舍（请评估后决定）：**
- documents 是否并入 artifacts（作为 `kind='source'`）？并入后 edges 只需要连 artifact，递归查询更简单。保留独立表则语义更清楚。请根据现有代码选一种并说明理由。
- chunk → embedding 这种一对一关系会让 edges 表膨胀。可以考虑一对一的关系直接在 artifact 上存 `parent_id`，只有多对多才写 edges。先评估数据量再定。

---

## 4. 通用层 API（建议签名，按仓库语言和风格调整）

```python
register_document(namespace, source_uri, content_hash, metadata) -> Document
    # content_hash 没变 → 返回现有版本；变了 → 新建版本，旧版本标 superseded

get_or_create_step(name, version, config) -> Step   # 内部计算 config_hash

compute_fingerprint(step, inputs: list[Document|Artifact]) -> str

should_recompute(namespace, step, inputs) -> tuple[bool, Artifact | None]

run_step(namespace, step, inputs, fn) -> list[Artifact]
    # 1. 算 fingerprint → 命中则直接返回，不调用 fn
    # 2. 未命中：建 run → 调用 fn(inputs) → 写 artifacts + edges → 结束 run
    # 3. 整个过程要幂等，fn 失败时 run 标为 failed，不留半成品 active artifact

upstream_of(artifact_id) -> list[Node]       # 递归 CTE 向上
downstream_of(node_id) -> list[Artifact]     # 递归 CTE 向下（影响分析）

mark_deleted(document_id) -> list[Artifact]  # 级联把下游 artifact 标 deleted，返回清单供上层清理向量
supersede(...)                               # 新版本生效时标记旧产物
gc(older_than, keep_metadata=True)           # 删除 superseded/deleted 的 payload，保留元数据行
```

影响分析的递归查询示例：

```sql
WITH RECURSIVE downstream AS (
  SELECT child_id FROM lineage_edges WHERE parent_id = :node_id
  UNION
  SELECT e.child_id FROM lineage_edges e
  JOIN downstream d ON e.parent_id = d.child_id
)
SELECT a.* FROM lineage_artifacts a JOIN downstream d ON a.id = d.child_id;
```

---

## 5. RAG 层（lite-rag 中实现）

### 5.1 把现有 ingestion 改造成 step

- 现有流程中的 load → parse → chunk → embed → 写入向量库，逐个包装成 `run_step` 调用。
- 每个 step 都要有显式的 version 和 config（chunk_size、overlap、embedding 模型名、维度……）。
- 写入向量库时，向量的 metadata 要带上 `artifact_id`。删除时，以 lineage 返回的 artifact 清单为准去删除向量。

### 5.2 RAG 专属表

```sql
CREATE TABLE kb_members (
  kb_id       UUID NOT NULL,
  document_id UUID NOT NULL,
  added_at    TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (kb_id, document_id)
);

CREATE TABLE rag_query_traces (
  id               UUID PRIMARY KEY,
  namespace        TEXT NOT NULL,
  user_id          TEXT,
  query_text       TEXT,              -- 可配置为只存 hash
  model_name       TEXT,
  prompt_version   TEXT,
  embed_model      TEXT,
  index_version    TEXT,
  retrieval_config JSONB,             -- top_k、filters、reranker
  answer_ref       TEXT,
  external_trace_id TEXT,             -- Langfuse 等外部 trace ID（可选）
  latency_ms INT, tokens_in INT, tokens_out INT,
  created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE rag_query_sources (
  trace_id    UUID NOT NULL REFERENCES rag_query_traces,
  artifact_id UUID NOT NULL REFERENCES lineage_artifacts,
  stage       TEXT NOT NULL,          -- retrieved/reranked/in_context/cited
  rank        INT,
  score       FLOAT,
  PRIMARY KEY (trace_id, artifact_id, stage)
);
```

- 文档版本**不要**冗余存进 sources，通过 `upstream_of(artifact_id)` 查。
- 判断查询时是否用了过期数据：`artifact.superseded_at IS NOT NULL AND artifact.superseded_at < trace.created_at`。
- trace 记录要做成**可关闭、可配置保留期**（例如 90 天后只保留聚合数据）。
- **普通问答的答案不写进 artifacts。** 只有会被持久化和复用的产出（缓存答案、生成的报告或 HTML 学习页）才作为 artifact 并写入 edges。

---

## 6. 实施阶段

1. **分析（先做，产出计划给我确认）**
   - 现有 ingestion 流程、存储（向量库、元数据库）、配置方式、语言和依赖。
   - 现在有没有任何 hash、去重、版本相关的逻辑。
   - 给出：package 放在哪、documents 是否并入 artifacts、一对一边怎么处理、需要改哪些现有文件。
2. **通用层**：表 + migration + API + 单元测试。
3. **接入 ingestion**：chunk/embed 改造为 step，向量带 artifact_id。
4. **删除与 GC**：源删除级联清理向量；superseded 产物的 GC。
5. **查询 trace**：两张 trace 表 + 检索流程埋点（四个 stage）。
6. **文档**：README 中写明用法和第三方项目如何接入通用层。

每个阶段完成后都要保证现有功能和测试不受影响。

---

## 7. 验收标准（需有自动化测试覆盖）

- [ ] 同一批文档重复 ingest：第二次 0 次 embedding 调用，全部 skipped。
- [ ] 修改 1 个文档：只有该文档的 chunk/embedding 被重算，其他全部 skipped。
- [ ] 修改 chunk 配置（version 或 config 变）：所有 chunk 及其下游重算，embedding step 本身的配置不变也会重算（因为输入 hash 变了）。
- [ ] 切换 embedding 模型：chunk 全部复用，只重算 embedding；新旧索引可以并存。
- [ ] 删除 1 个文档：下游 artifact 全部标记 deleted，对应向量从向量库删除，检索不再返回。
- [ ] `upstream_of(embedding_id)` 能返回到源文档及其版本。
- [ ] `downstream_of(document_id)` 返回完整的影响清单。
- [ ] step 执行失败：run 标记为 failed，没有残留 active 的半成品 artifact，重跑可以恢复。
- [ ] 查询 trace 能区分 retrieved/in_context/cited，并能判断当时是否用了已过期的 chunk。
- [ ] 通用层的 import 中不出现任何 RAG 模块（可以加一个测试或 lint 规则来保证）。

---

## 8. 参考（只借鉴思路，不引入依赖）

- LangChain Indexing API / RecordManager：hash 去重 + cleanup 模式。注意它的 incremental 模式不会清理已从语料中删除的源，这一点我们要做到。
- LlamaIndex IngestionPipeline + docstore：hash 去重与 upsert。
- CocoIndex：声明式增量引擎 + 端到端 lineage。如果以后数据源变多、需要实时同步，可以考虑迁移过去。
- OpenLineage：dataset / job / run 的概念模型，可参考命名，不要引入。

---

## 9. 需要你（Claude Code）在计划中明确回答的问题

1. 现有代码里哪些部分可以直接改造成 step，哪些需要重写？
2. DB 选型：只支持 Postgres，还是 SQLite + Postgres 都支持？
3. documents 是否并入 artifacts？一对一的边怎么存？
4. 向量库是否支持按 metadata 批量删除？不支持的话，删除怎么实现？
5. 现有的检索接口中，在哪里埋 trace 的四个 stage 改动最小？
6. 预估各阶段的改动范围和风险。
