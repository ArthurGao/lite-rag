# lite-rag 设计文档

> 状态：设计稿（2026-11-08）。实施按第 12 节的阶段推进；某个阶段的行为改变时，在同一次提交里更新本文。

## 1. 概述

lite-rag 是一个**可读、可改、可单独部署**的轻量 RAG 引擎，有两个用途：

1. **学习**：切块、embedding、BM25、融合、rerank、生成，每一步都是这个仓库里自己写的代码，可以单独替换，效果可以量化比较。
2. **通用 RAG**：既能作为 Python 库安装使用（`pip install`），也能作为独立服务运行（`literag serve`，或者直接用 Docker 镜像），通过 HTTP API 调用。使用方不需要拿到这个仓库的源码。[llm-steward](https://github.com/ArthurGao/llm-steward) 是第一个使用方，它负责治理，lite-rag 负责存储和检索。

### 目标

- **默认零成本**：Ollama 提供 embedding 和 LLM；存储用 PostgreSQL + pgvector，也可以选 Chroma 做纯嵌入式运行。
- **库和服务同一套核心**：`RagEngine` 是唯一实现，HTTP 服务只是在它外面包一层，两边的行为一致。
- **每个环节都可插拔**：`Chunker`、`Embedder`、`Store`、`Reranker`、`LLM` 都是 Protocol，换实现只改配置。
- **可以度量**：自带评测（hit@k、MRR、延迟），同一组问题在不同配置下并排比较。
- **LLM 可切换**：`ollama`、`openai`（以及任何 OpenAI 兼容的端点，例如 llm-gateway）、`anthropic`，只改配置。

### 非目标（v1）

- **不做治理**：审核、状态机和用户权限都交给 llm-steward 这类上层系统。lite-rag 只做 `kb` 级别的数据隔离，外加服务端的 API key 鉴权。
- **不做复杂文档解析**：OCR、版面分析、表格结构交给 RAGFlow 这类引擎。v1 只支持 txt、md；PDF 只抽取纯文本，作为可选 extra。
- **不做分布式**：一个 PG 实例就够用。目标规模是每个 kb 十万级 chunk。

## 2. 数据流

```
摄取：  文件/文本 → Loader → Document → Chunker → chunks → Embedder ─┐
                                                   └→ 分词(jieba) ────┴→ Store（一个事务：登记 + 向量 + 倒排）
检索：  query → Embedder → 向量 top-n ┐
        query → 分词 → BM25 top-n    ┴→ 融合(RRF) → [Reranker] → top-k Hit
回答：  Hit[] → prompt 模板(answer.j2) → LLM → Answer(text, citations, provenance)
```

## 3. 组件与接口

所有接口都是 `typing.Protocol`；测试用 fake 实现，不依赖网络和模型。

| 接口 | 方法 | 默认实现 | 其他实现 |
| --- | --- | --- | --- |
| `Loader` | `load(path) -> Document` | txt、md | pdf（extra `[pdf]`，只抽纯文本） |
| `Chunker` | `split(doc) -> list[Chunk]` | `recursive` | 见 §3.1：fixed、sentence、paragraph、markdown、semantic、parent-child，另有 contextual 增强 |
| `Embedder` | `embed_documents(texts)`、`embed_query(text)`、`dim`、`model` | `ollama`（`bge-m3`） | 见 §3.2：openai 兼容、voyage、cohere、gemini、fastembed、sentence-transformers、`FakeEmbedder` |
| `Store` | `upsert_document`、`delete_document`、`vector_query`、`keyword_query`、`get_document`、`stats` | `PgStore`（pgvector 存向量，PG 表存 BM25 倒排和文档登记） | `ChromaStore`（extra `[chroma]`）：向量放 Chroma，倒排和登记放 SQLite |
| `Reranker` | `rerank(query, hits) -> hits` | `NoRerank` | 本地 cross-encoder `bge-reranker`（extra `[rerank]`） |
| `LLM` | `complete(messages) -> Completion` | `OpenAICompatLLM`（Ollama `/v1`、OpenAI、llm-gateway） | `AnthropicLLM`（extra `[anthropic]`）、`FakeLLM` |

`Store` 把向量和倒排放在同一个接口里，这样 PG 后端能在一个事务里完成一篇文档的所有写入。BM25 的打分公式写在 `keyword.py` 里，两个后端共用，后端只负责存取倒排数据。

`RagEngine` 是库的入口，HTTP 服务调用的也是它：

```python
engine = RagEngine.from_config(cfg)
engine.ingest(kb, path)                                   # 文件 → 文档 id
engine.upsert_text(kb, doc_id, title, text, metadata={})  # 已存在则整篇替换
engine.delete(kb, doc_id)
engine.search(kb, query, top_k=5, mode="hybrid", rerank=False) -> list[Hit]
engine.ask(kb, question, top_k=5) -> Answer
engine.stats(kb)
```

### 3.1 切块策略（每个 kb 单独配置）

切块方式可以按 kb 配置，按名字从注册表里选实现；以后要加新策略，实现 `Chunker` Protocol 再注册即可。每个 chunk 都带上 `start`、`end` 字符偏移（用于引用定位）、`heading_path`（所在的标题层级）和 `tokens`（真实 token 数）。

| 策略 | 做法 | 适用 | 阶段 |
| --- | --- | --- | --- |
| `fixed` | 按固定长度切，带重叠 | 基线，用于对比 | S2 |
| `recursive`（默认） | 按分隔符层级递归切：标题 → 空行 → 换行 → 句末标点 → 空格，直到小于 `size`，再补上重叠 | 大多数文本 | S2 |
| `sentence` | 按中英文句末标点切句，再按窗口合并（每块 N 句，重叠 M 句） | 问答、短事实 | S2 |
| `paragraph` | 按空行切段，过短的段落向后合并，过长的再用 recursive 切 | 结构清晰的文章 | S2 |
| `markdown` | 按标题层级切，块内用 recursive；`heading_path` 作为前缀写进 embedding 文本 | 文档、手册、wiki | S2 |
| `semantic` | 先按句切，相邻句 embedding 的相似度跌破某个分位数（默认 95）时断开 | 主题频繁切换的长文 | S5b |
| `parent_child` | 小块（例如 200 token）建索引，命中后返回它所属的大块（例如 1000 token） | 需要精确命中、又需要上下文 | S5b |
| `contextual`（增强器） | 可以叠加在任意策略上：用 LLM 为每个 chunk 写一两句"这段在全文中讲什么"，拼在 chunk 前面再做 embedding 和 BM25 | 指代多、单看一段看不懂的文档；每个 chunk 多一次 LLM 调用 | S5b |

- **长度单位**：可以选 `chars` 或 `tokens`，默认 `tokens`，并且按 embedding 模型自己的 tokenizer 计数。KB 的教训是按词数切会失真（`rag-production-guide` §3）。拿不到 tokenizer 时用字符数估算，并记录这次是估算。
- **配置示例**：`chunking: {strategy: recursive, unit: tokens, size: 800, overlap: 120, min_size: 50}`。800/120 是 KB 里实测用过的起点，最终由 eval 决定。
- **改切块就要重建这个 kb**：`kbs` 表记录 chunker 配置，配置变了就要求 `literag reindex`。阈值也要重新校准，见 §5。
- 不在 v1 范围内：late chunking 和 proposition 切块。前者需要专门的长上下文 embedding 模型，后者每块都要一次 LLM 调用；等 eval 证明 S5b 不够用时再考虑。

### 3.2 Embedding（每个 kb 固定一个模型）

`Embedder` 分别提供 `embed_documents` 和 `embed_query`，因为很多模型要求区分文档和查询：例如 voyage 和 cohere 的 `input_type` 参数，以及 bge、e5、qwen3 这类模型给查询加的指令前缀。这些前缀写在每个模型的配置里，不写死在代码里。

| provider | 怎么调用 | 候选模型（S0/S5b 时按官方文档和 MTEB / C-MTEB 榜单核实后锁定） | 费用 |
| --- | --- | --- | --- |
| `ollama`（默认） | 本地 Ollama `/api/embed` | `bge-m3`（多语言，1024 维）、`qwen3-embedding`、`nomic-embed-text`、`mxbai-embed-large`、`snowflake-arctic-embed2` | 免费 |
| `fastembed` | 进程内 ONNX，不需要 Ollama | `BAAI/bge-small-zh-v1.5`、`intfloat/multilingual-e5-*` 等 | 免费 |
| `sentence_transformers` | 进程内 HuggingFace（extra，需要 torch） | 任意 HF embedding 模型 | 免费 |
| `openai` | OpenAI 兼容的 `/v1/embeddings` | OpenAI `text-embedding-3-large` / `-small`（支持 `dimensions` 截断）；同一个实现也能接 DashScope（通义 `text-embedding-v*`）、SiliconFlow、Jina、vLLM、TEI | 付费（各家免费额度不一） |
| `voyage` | 官方 SDK（extra） | `voyage-3-large` 等。KB 中 tenderted 实测用过 | 付费 |
| `cohere` | 官方 SDK（extra） | `embed-v4` 等多语言模型 | 付费 |
| `gemini` | 官方 SDK（extra） | `gemini-embedding-*`（支持截断维度） | 付费，有免费额度 |

- **"最好"由自己测**：榜单只能用来缩小候选范围。模型的取舍以本项目语料上的 `literag eval` 结果为准（AGENTS 规则 8）。所以 S5b 的验收是一张多模型对比表，列出 hit@k、MRR、延迟和每千 chunk 的费用。
- **一个 kb 只用一个模型**（`rag-production-guide` §4 规则 2）：`kbs` 表记录 `provider`、`model`、`dim` 和查询前缀；写入或查询时发现与配置不符，就拒绝执行。换模型等于新建一份索引后切换，不会在原地混写。
- **批量与限流**：`batch_size`、并发上限（默认 4）以及遇到 429/5xx 时的退避重试（默认 4 次）都可以配置。重试的是同一个模型，这不算回退链；用尽重试后直接抛错，绝不返回空结果。
- **维度截断**：模型支持截断（OpenAI `dimensions`、Gemini、Matryoshka 模型）时，可以配置 `dimensions`。PG 后端按维度分表（§4）。
- 付费 provider 的 key 只从环境变量读取。测试里一律用 fake，或者回放录制好的响应，不会真的调用接口。

## 4. 存储与 id

### PostgreSQL 后端（默认）

库（例如 `lite_rag`）里有一个 schema（默认 `lite_rag`，可配置）。这些表由 `literag migrate` 创建，迁移脚本放在包内：

| 表 | 内容 |
| --- | --- |
| `kbs` | `name`（主键）、`embedding_model`、`dim`、`chunker`（JSON）、`created_at` |
| `documents` | `kb`、`doc_id`、`title`、`source`、`content_hash`、`chunk_count`、`metadata` JSONB、`updated_at`；主键 `(kb, doc_id)` |
| `chunks` | `id`（`"{doc_id}#{n}"`）、`kb`、`doc_id`、`n`、`text`、`length`（词数）、`metadata` JSONB |
| `embeddings_<dim>` | `chunk_id`、`kb`、`embedding vector(<dim>)`，HNSW 索引（cosine）。每个维度一张表，首次使用时创建 |
| `postings` | `kb`、`term`、`chunk_id`、`tf`；索引 `(kb, term)` |
| `kb_stats` | `kb`、`n_chunks`、`total_length`（用来算 `avgdl`） |

- **一个事务**：`upsert_document` 在同一个事务里删掉旧的 chunk、embedding 和 posting，写入新的，再更新 `documents` 和 `kb_stats`。没有中间状态，也不需要恢复逻辑。
- **BM25 查询**：先在 SQL 里按 `(kb, term)` 取出 posting，同时拿到 `n(t)`、`N`、`avgdl`，再按第 5 节的公式打分。打分既可以在 Python 里算（`keyword.py`，便于阅读），也可以整段在 SQL 里算（快）。两种算法的结果用同一组测试核对。
- **向量查询**：`ORDER BY embedding <=> :q LIMIT n`，按 `kb` 过滤。

### Chroma 后端（可选）

- 数据目录 `LITERAG_DATA_DIR` 下有两部分：
  - `chroma/`：每个 kb 一个 collection；
  - `lite_rag.sqlite`：和 PG 后端同结构的 `kbs`、`documents`、`chunks`、`postings`、`kb_stats` 表。
- **写入不在一个事务里**：向量在 Chroma，其余在 SQLite。所以写入顺序固定为"登记 pending → 写 SQLite → 写 Chroma → 标记 ready"。打开引擎时，未完成的文档先清理掉再重写。
- 适合不想装 PG 的单机学习场景。合约测试对两个后端运行同一套用例。

### 通用规则

- **id 规则**：文档 id 由调用方给出（`upsert_text`），或者由文件路径的哈希生成（`ingest`）。chunk id 是 `"{doc_id}#{n}"`。
- **幂等**：`content_hash` 和 chunker 配置都没变时，`upsert_text` 直接返回；否则整篇替换。
- **换 embedding 模型**：`kbs` 表记录了每个 kb 的模型和维度。配置与记录不一致时拒绝打开，并提示运行 `literag reindex <kb>`。

## 5. 检索

- **向量**：先对 query 做 embedding，再取 top-n（n = `top_k × candidate_factor`，默认取 4 倍）。
- **BM25**：中文用 jieba 分词，英文转小写；停用词表可配置。采用标准公式：

  `score(q, d) = Σ IDF(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 − b + b·|d|/avgdl))`，
  `IDF(t) = ln(1 + (N − n(t) + 0.5)/(n(t) + 0.5))`，默认 `k1=1.5`、`b=0.75`。
- **融合**：默认用 RRF，`score = Σ 1/(k + rank)`，`k=60`。也可以选加权融合：两路分数先各自做 min-max 归一化，再按权重相加。
- **Rerank**（可选）：取融合后的前 n 条，用 cross-encoder 对每个 `(query, chunk)` 对重新打分，再取前 k 条。
- **模式**：`vector`、`bm25`、`hybrid`。每个 `Hit` 都带各阶段的分数 `scores={vector, bm25, fused, rerank}` 和排名，方便对比和调试。
- **过滤**：按 `kb` 隔离；`where` 支持对元数据做等值过滤，向量和 BM25 两路都会生效。

## 6. 生成

- **提示词模板** `prompts/answer.j2`：
  - 检索到的 chunk 依次编号为 `[1] … [k]`；
  - 要求只依据这些资料回答，并标注引用的编号；
  - 资料不足时，直接说明资料不足。
- **`Answer` 包含**：
  - 回答正文；
  - `citations`：引用了哪些 chunk id，代码会核对每个编号都在检索结果里；
  - `provenance`：`provider:model`；
  - token 用量。
- **没有回退链**：一次调用失败就抛出错误。
- **LLM 配置**：

```yaml
llm:
  provider: ollama        # ollama | openai | anthropic
  model: qwen3:8b
  base_url: http://127.0.0.1:11434/v1   # 仅 openai 兼容类；llm-gateway 用 http://127.0.0.1:8787/v1
  api_key_env: null       # openai: OPENAI_API_KEY；anthropic: ANTHROPIC_API_KEY
  temperature: 0
```

  `ollama` 走 Ollama 的 OpenAI 兼容接口，和 `openai` 共用一个实现；`anthropic` 用官方 SDK。key 只从环境变量读取。

## 7. HTTP 服务（`literag serve`）

FastAPI，默认地址 `127.0.0.1:8700`。和库一样，所有调用都经过 `RagEngine`。

| 端点 | 作用 |
| --- | --- |
| `GET /health` | 不需要鉴权；检查存储和 embedding 服务 |
| `PUT /v1/kbs/{kb}` · `GET /v1/kbs` · `GET/DELETE /v1/kbs/{kb}` | 建 kb（指定 chunker 与 embedding 模型）、列出、查看统计、删除 |
| `PUT /v1/kbs/{kb}/documents/{doc_id}` | upsert 一篇文本文档（`title`、`text`、`metadata`） |
| `POST /v1/kbs/{kb}/files` | 上传文件（multipart），返回 `doc_id` |
| `DELETE /v1/kbs/{kb}/documents/{doc_id}` | 删除文档 |
| `POST /v1/kbs/{kb}/search` | `{query, top_k, mode, rerank, where}` → hits（带各阶段分数） |
| `POST /v1/kbs/{kb}/ask` | `{question, top_k}` → answer（带引用和出处） |

- **鉴权**：`Authorization: Bearer <key>`。服务端只保存 key 的 `sha256`，配置为 `LITERAG_API_KEYS`（逗号分隔的哈希值）。每个 key 可以限定只能访问某些 kb，以及只读或读写。不配 key 时，服务只允许监听 `127.0.0.1`。
- **错误格式**：RFC 9457 Problem Details。
- **部署**：仓库自带 `Dockerfile`，CI 构建镜像（`ghcr.io/arthurgao/lite-rag`），用环境变量指向 PG 和 Ollama。仓库附带的 `docker compose` 示例包含 lite-rag + PG（pgvector）；Ollama 默认跑在宿主机上。
- **安装**：`pip install "lite-rag[server] @ git+https://github.com/ArthurGao/lite-rag"`。以后再发布到 PyPI。

## 8. 评测

- **语料**：`eval/` 下放一小份语料，中英文各一组，内容是自己写的公开知识。金标问题用 JSONL 存放，每行 `{question, expected_doc_ids[], expected_chunk_ids?[]}`。
- **对比**：`literag eval --config a.yaml --config b.yaml` 为每个配置建一个临时 kb，再跑全部问题。结果并排列出 hit@1、hit@5、MRR@10，以及检索延迟的 p50 和 p95。
- **用法**：这是学习的主循环。每换一种 chunker、加上 BM25、打开 rerank，都跑一次 eval，用数字判断效果有没有变好。

## 9. 配置

用 pydantic-settings 读取，环境变量前缀为 `LITERAG_`，也可以用 `--config file.yaml` 指定配置文件：

```yaml
store: {backend: postgres, database_url_env: LITERAG_DATABASE_URL, schema: lite_rag}   # 或 {backend: chroma, data_dir: ./data}
embedding: {provider: ollama, model: bge-m3, base_url: http://127.0.0.1:11434, batch_size: 32, max_concurrency: 4, query_prefix: ""}
chunking: {strategy: recursive, unit: tokens, size: 800, overlap: 120, min_size: 50, contextual: {enabled: false}}
retrieval: {mode: hybrid, top_k: 5, candidate_factor: 4, fusion: rrf, rrf_k: 60, bm25: {k1: 1.5, b: 0.75}}
rerank: {enabled: false, model: BAAI/bge-reranker-base, top_n: 20}
llm: {provider: ollama, model: qwen3:8b}
server: {host: 127.0.0.1, port: 8700}
```

## 10. 与 llm-steward 集成

llm-steward 新增一个 `LiteRagProvider`，实现它的 `KnowledgeProvider` Protocol。它支持两种接入方式：in-process 直接 import 库，或者 HTTP 调用服务。两种方式都会跑 llm-steward 的合约测试。

| llm-steward 调用 | lite-rag 实现 |
| --- | --- |
| `ensure_collection(collection)` | 建一个 kb，名为 `steward-<collection_id>`，返回 kb 名 |
| `upsert_entry(entry, kb, external_id)` | `upsert_text(kb, doc_id=str(entry_id), title, content)`，返回 `doc_id` |
| `remove(external_id)` | `delete(kb, doc_id)` |
| `search(query, kbs, top_k)` | 对每个 kb 调用 `search`，按 `doc_id` 合并结果（取最高分） |
| `health()` | 存储可用，且 embedding 服务可用 |

- capabilities 为 `{entry_upsert}`。`doc_parsing` 等 llm-steward P3 做文档时再加。
- 治理规则仍由 llm-steward 保证：只有生效的条目才会写入，检索结果要经过它的安全网。
- llm-steward 和 lite-rag 可以共用同一个 PG 实例，但各用各的库或 schema。

## 11. 项目结构

```
lite-rag/
  pyproject.toml            # uv；extras: [server] [chroma] [pdf] [rerank] [anthropic]
  Dockerfile  compose.example.yml
  src/lite_rag/
    config.py loaders.py chunking.py embedding.py keyword.py fusion.py rerank.py llm.py engine.py
    stores/ base.py pg.py chroma.py          # Store Protocol + 两个后端
    migrations/                              # PG schema（literag migrate）
    server/ app.py auth.py schemas.py        # FastAPI（extra [server]）
    cli.py prompts/answer.j2
  tests/   unit/（fakes）  integration/（PG：独立的测试 schema；Chroma：临时目录）  -m ollama 为可选
  eval/    语料、金标问题、报告
  docs/lite-rag-design.md
```

## 12. 阶段与验收

每个阶段结束时，ruff、ruff format、mypy、pytest 都要通过；每个新测试至少见过一次失败（red）。

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| S0 环境 | 安装 Ollama，拉取 `bge-m3` 和 `qwen3:8b`；在共享 PG 上建库 `lite_rag` 和对应角色；用文档核对 Ollama、pgvector 驱动、Chroma 和 rerank 包的 API 与版本，并锁定版本 | `ollama list` 列出两个模型；`psql` 能连上 `lite_rag`，`vector` 扩展可用；版本写进 `pyproject.toml` |
| S1 骨架 | uv、ruff、mypy、pytest、配置、fakes、CLI 骨架、`literag migrate` | `literag --help` 可用；迁移在测试 schema 上执行通过 |
| S2 摄取（PG） | Loader；chunker：`fixed`、`recursive`、`sentence`、`paragraph`、`markdown`；embedder：`ollama`、`openai` 兼容、`fastembed`；`PgStore`（单个事务完成 upsert 和删除）、`upsert_text`、`delete` | chunk 边界测试（中英文）；重复 upsert 不会重写；删除后 chunk、embedding、posting 全为空；故障注入：写入中途出错时整篇回滚 |
| S3 检索 | BM25（Python 版和 SQL 版）、RRF、三种模式、各阶段分数、`literag search` | BM25 与手算结果一致，Python 版和 SQL 版结果一致；融合顺序正确；`-m ollama` 真实往返测试通过 |
| S4 rerank 与回答 | cross-encoder、三种 LLM 实现、`answer.j2`、`literag ask`、引用核对 | FakeLLM 测试：越界的引用会被拒绝；切换 provider 只需改配置 |
| S5 评测 | 语料、金标问题、`literag eval` 对比表 | 用真实 Ollama 跑出一张对比表：fixed 对比 recursive 对比 markdown、vector 对比 hybrid、rerank 开与关 |
| S5b 进阶切块与更多 embedding | 进阶切块 `semantic`、`parent_child`、`contextual`；embedder `voyage`、`cohere`、`gemini`、`sentence_transformers` | 切块与 embedder 都有单测（fake 或录制响应）；用 eval 跑出两张表：切块策略对比、多个 embedding 模型对比（含费用列），结果记进 KB |
| S6 HTTP 服务 | FastAPI、API key 鉴权、Problem Details、Dockerfile、compose 示例、CI 构建镜像 | API 测试：未授权返回 401，越权访问 kb 返回 403；`docker compose up` 后能完成 upsert → search → ask；库和 API 的结果一致 |
| S7 Chroma 后端 | `ChromaStore` 加 SQLite 倒排，以及写入顺序与恢复逻辑 | 同一套 Store 合约测试在 PG 和 Chroma 上都通过；崩溃恢复测试 |
| S8 收尾 | `/simplify`、Codex review、修复问题，然后重跑全部检查 | review 结论记录在案 |
| S9 接入 llm-steward | `LiteRagProvider`（in-process 和 HTTP 两种）、合约测试、更新 llm-steward 设计文档 | llm-steward 的 KB 合约测试在两种接入方式下都通过；手动端到端：生效条目能检索到，废弃后检索不到 |

## 13. 待确认

- 各依赖的 license（Ollama、pgvector、Chroma、jieba、bge-m3、bge-reranker、qwen3）：锁定版本时逐个核对，写进 README。
- rerank 的打包方式：fastembed 的 cross-encoder 还是 sentence-transformers。S4 开始前查文档确认。前者更轻，不需要 torch。
- PDF 纯文本抽取用哪个库：S2 时确认。
- PG 驱动用 psycopg 3 还是 asyncpg，库是否提供 async API：S1 时决定。倾向于 sync 核心加 psycopg 3，服务层用线程池；这一点需要先评估吞吐。
