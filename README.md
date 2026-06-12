# Data-to-Text

食品安全抽检报告自动生成系统，基于 Agent + LLM 架构（Qwen3-14B），将抽检数据自动转化为专业的 Markdown/Word 分析报告。覆盖**监督抽检**和**风险监测**两大业务场景。

**技术栈**：Python 3.10+ / Pandas / Pydantic / python-docx / Matplotlib / SciPy / Ollama / vLLM / QLoRA 微调

## 系统架构

```
用户指令
  │
  ▼
┌──────────────┐
│  意图识别     │  IntentRecognizer (规则 + LLM 增强混合)
│  (intent.py)  │
└──────┬───────┘
       │ 结构化意图 (year, quarter, food_category, region, report_type, dimension...)
       ▼
┌──────────────┐
│  数据查询     │  DataQuerier → DataLoader (CSV 数据源)
│  (query.py)   │
└──────┬───────┘
       │ 多张 DataFrame (prov_trend, risk_items, market_inspection, high_rate...)
       ▼
┌──────────────┐
│  统计计算     │  StatisticsEngine (卡方检验、趋势分析、Wilson CI、异常检测、组间比较)
│(statistics.py)│
└──────┬───────┘
       │ 统计指标 (mean_rate, chi2, p_value, trend_direction...)
       ▼
┌──────────────┐
│  上下文组装   │  ContextAssembler → ReportContext (Pydantic 模型)
│ (context.py)  │
└──────┬───────┘
       │ 结构化 Prompt 文本
       ▼
┌──────────────┐
│  报告生成     │  ReportGenerator (Ollama / vLLM 后端, 含降级模式)
│ (generator.py)│
└──────┬───────┘
       │ Markdown 报告文本
       ▼
┌──────────────┐
│  事实校验     │  FactChecker (数值偏差检测, tolerance=5%)
│(fact_checker) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Word 文档    │  ReportTemplate + ChartGenerator + MarkdownToDocx
│  生成         │  (图表生成 + Markdown→Word 转换)
└──────────────┘
```

## 目录结构

```
Data-to-Text/
├── pyproject.toml                   # 项目配置、依赖、ruff/pytest 配置
│
├── src/                             # 核心源码
│   ├── cli.py                       # CLI 入口（d2t 命令）
│   ├── config.py                    # 全局配置（路径、环境变量、CSV映射、Schema元数据）
│   │
│   ├── agent/                       # Agent 层：意图识别、数据查询、统计、生成
│   │   ├── intent.py                # 意图识别（规则引擎 + LLM 增强混合策略）
│   │   ├── query.py                 # 数据查询（根据意图加载并过滤多张数据表）
│   │   ├── statistics.py            # 统计引擎（卡方检验、趋势分析、Wilson CI、异常检测、ANOVA）
│   │   ├── generator.py             # 报告生成（Ollama/vLLM 双后端 + 降级模式 + 事实校验集成）
│   │   ├── fact_checker.py          # 事实校验（提取生成文本中的数值并与上下文比对）
│   │   └── context.py               # 上下文组装（将查询结果+统计结果组装为 ReportContext）
│   │
│   ├── data/                        # 数据层：加载、字典、模型
│   │   ├── loader.py                # 数据加载器（CSV 读取、缓存、列校验、多表映射）
│   │   ├── dictionary.py            # 数据字典（读取 Excel 字典文件，字段查询）
│   │   └── models.py                # Pydantic 数据模型（BatchDetail, TrendData, RiskItem, ReportContext 等）
│   │
│   ├── report/                      # 报告输出层
│   │   ├── template.py              # 报告模板（生成图表 + 组装 Markdown + 转 Word）
│   │   ├── converter.py             # Markdown→Word 转换器（标题、表格、列表、图片、引用、代码块）
│   │   └── charts.py                # 图表生成器（柱状图、趋势折线图、饼图、同比对比图）
│   │
│   └── training/                    # 训练数据层
│       ├── prepare_data.py          # 训练数据准备（DOCX 解析、CSV 对、JSONL、去重、验证）
│       ├── shared.py                # 训练数据格式化（Alpaca 格式、ChatML 格式、章节指令模板）
│       └── evaluate.py              # 模型评估（词重叠、事实准确率、结构完整性、综合评分）
│
├── scripts/                         # 工具脚本（独立运行，需 GPU 环境）
│   ├── finetune_cloud.py            # 云端 QLoRA 微调脚本（Qwen3-14B, AutoDL）
│   ├── inference_cloud.py           # 云端推理对比测试（基础模型 vs LoRA 微调）
│   ├── deploy_to_autodl.py          # AutoDL 部署命令生成器
│   └── optimize_training_data.py    # 训练数据质量优化脚本
│
├── prompts/                         # Prompt 模板
│   ├── intent_recognition.txt       # 意图识别系统提示词
│   └── system_report.txt            # 报告生成系统提示词
│
├── training_data/                   # 训练数据（JSONL 格式）
│   ├── training_data.jsonl          # 原始训练数据
│   └── training_data_formatted.jsonl # 微调格式数据（ChatML messages）
│
├── tests/                           # 单元测试
│   ├── test_intent.py               # 意图识别测试
│   ├── test_query.py                # 数据查询测试
│   ├── test_statistics.py           # 统计计算测试
│   ├── test_generator.py            # 报告生成测试
│   ├── test_fact_checker.py         # 事实校验测试
│   ├── test_context.py              # 上下文组装测试
│   └── test_loader.py               # 数据加载测试
│
└── output/                          # 运行时输出目录（报告、图表）
```

## 核心模块详解

### 意图识别 — `src/agent/intent.py`

**类**：`IntentRecognizer`

**策略**：规则引擎优先 + LLM 增强混合

- **规则引擎**：通过正则表达式和预定义词典提取实体
  - 年份：`/(\d{4})\s*年/` 或纯数字匹配
  - 季度：中文/数字季度 + `Q1-Q4` 格式
  - 食品大类：31 个已知品类 + 别名映射（如"辣椒类"→"辣椒"）
  - 地区：34 个省级行政区 + 简称映射（如"粤"→"广东"）
  - 报告类型：监督抽检 / 风险监测 / 综合分析（含模糊匹配）
  - 维度：全省 / 大类食品 / 细类食品 / 地市 / 环节 / 抽样场所

- **LLM 增强**：当规则识别结果不完整（缺年份、报告类型或食品大类）时，调用 LLM 补全
  - 支持 Ollama 和 vLLM/OpenAI 两种后端
  - 通过环境变量 `INTENT_USE_LLM=true` 开启

### 数据查询 — `src/agent/query.py`

**类**：`DataQuerier`

根据意图从 `DataLoader` 加载并过滤数据，返回多张 DataFrame。

| 键名 | 数据内容 | 过滤条件 |
|------|---------|---------|
| `prov_trend` | 全省三年趋势 | — |
| `prov_batch_detail` | 全省批次明细 | 按年份 |
| `risk_items` | 违规项/风险项 | 按食品大类/细类/项目 |
| `market_inspection` | 市场抽检明细 | 按食品大类/细类/项目 + 年份 |
| `high_rate` | 高不合格率项目 | 按食品大类/细类/项目 |
| `exceedance` | 超标倍数 | 按食品大类/细类/项目 + 年份 |
| `near_limit` | 近限值 | 按食品大类/细类/项目 + 年份 |
| `category_trend` | 大类食品趋势 | 按食品大类/细类/项目 |
| `item_trend` | 检测项目趋势 | 按检测项目名 |
| `seasonal` | 季节性风险 | 按食品细类/大类 |

过滤逻辑：`_filter_by_food()` 方法按 `sp_s_17`→`sp_s_18`→`sp_s_19`→`sp_s_20` 四级食品分类列逐级匹配

### 统计引擎 — `src/agent/statistics.py`

**类**：`StatisticsEngine`（全部静态方法）

| 方法 | 功能 | 关键输出 |
|------|------|---------|
| `chi_square_test()` | 卡方检验 | chi2, p_value, significant, conclusion |
| `yoy_change()` | 同比变化计算 | yoy_change, yoy_change_pct, direction |
| `trend_analysis()` | 线性回归趋势分析 | slope, r_squared, p_value, direction |
| `wilson_confidence_interval()` | Wilson 置信区间 | rate, lower, upper |
| `summarize_rates()` | 不合格率汇总统计 | mean/max/min/median/std_rate, overall_rate |
| `detect_anomalies()` | 异常值检测（IQR/Z-score） | 异常值索引列表 |
| `compare_groups()` | 多组比较（单因素方差分析） | f_statistic, p_value, significant |

### 上下文组装 — `src/agent/context.py`

**类**：`ContextAssembler`

将查询结果和统计结果组装为 `ReportContext` Pydantic 模型，组装的数据模块：

- `province_summary`：全省概况（总批次、不合格批次、不合格率、三年趋势、同比变化、P值）
- `category_details`：分类抽检详情（前 20 条）
- `trend_analysis`：三年趋势分析（前年/去年/今年批次和不合格率、slope、chi2 等）
- `risk_items`：风险项目（前 15 条，含三年结果/率、趋势）
- `seasonal_analysis`：季节性风险（前 20 条，Q1-Q4 结果）
- `high_rate_items`：高不合格率项目（前 15 条）
- `statistics`：统计指标（来自 `StatisticsEngine`）

输出：`to_prompt_text()` 生成结构化 Markdown 文本供 LLM 使用

### 报告生成 — `src/agent/generator.py`

**类**：`ReportGenerator`

1. 调用 `ContextAssembler.assemble()` 组装上下文
2. 调用 `ContextAssembler.to_prompt_text()` 生成 Prompt
3. 调用 LLM 生成报告（Ollama 或 vLLM 后端）
4. 调用 `FactChecker.check()` 进行事实校验
5. 记录生成指标（成功率、降级率、事实校验通过率、平均耗时）

**降级模式**：当 LLM 调用失败时，从上下文文本中正则提取关键数据，生成结构化但较简略的报告，并标注"降级模式"

LLM 参数：temperature=0.3, num_predict/max_tokens=4096

### 事实校验 — `src/agent/fact_checker.py`

**类**：`FactChecker`

检查生成报告中的数值是否与上下文数据一致：

1. 从生成文本中提取数值事实（不合格率、抽检批次、P值、趋势方向等）
2. 从上下文中提取参考数值
3. 比对偏差，超过 tolerance（默认 5%）则记录警告

判定标准：准确率 ≥ 90% 通过 / ≥ 70% 警告 / < 70% 不通过

### 数据加载 — `src/data/loader.py`

**类**：`DataLoader`

- `resolve_csv_mapping()`：自动匹配带时间戳后缀的 CSV 文件名
- 列校验：对关键表检查必需列是否存在
- 38+ 张数据表的映射关系（见 `config.py` 中的 `CSV_FILE_MAPPING`）

### 数据模型 — `src/data/models.py`

| 模型 | 用途 |
|------|------|
| `BatchDetail` | 单年度批次明细 |
| `TrendData` | 三年趋势数据 |
| `RiskItem` | 风险项目 |
| `YearComparison` | 三年对比 |
| `SeasonalRisk` | 季节性风险 |
| `TrainingSample` | 训练样本 |
| `ReportContext` | 报告上下文（核心模型，含验证逻辑） |

### 报告输出 — `src/report/`

- **`template.py`** — `ReportTemplate`：生成图表 + 组装 Markdown + 转 Word
- **`converter.py`** — `MarkdownToDocx`：支持标题、表格、列表、引用、代码块、图片、行内格式
- **`charts.py`** — `ChartGenerator`：柱状图、趋势折线图、饼图、同比对比图，自动中文字体适配

### 训练数据 — `src/training/`

- **`prepare_data.py`** — `TrainingDataPreparer`：从 DOCX 报告中提取"数据表-分析文本"对，支持两种提取模式，自动去重、验证、追加
- **`shared.py`**：ChatML/Alpaca 格式化 + 14 套章节指令模板（7 种章节 × 2 种分析类型）
- **`evaluate.py`** — `Evaluator`：词重叠率（25%）、事实准确率（40%）、结构完整性（35%）

### 云端脚本 — `scripts/`

- **`finetune_cloud.py`**：Qwen3-14B QLoRA 微调脚本（r=16, alpha=32, 4-bit NF4 量化）
- **`inference_cloud.py`**：云端推理对比测试（基础模型 vs LoRA 微调，CLI 参数 `--base-only` / `--lora-only` / `--num-samples N`）
- **`deploy_to_autodl.py`**：AutoDL 部署命令生成器
- **`optimize_training_data.py`**：训练数据质量优化脚本

## CLI 命令

入口：`d2t`（安装后可用）或 `python -m src.cli`

```bash
# 生成报告
d2t generate "帮我生成一份2025年广西监督抽检报告" --docx --name "抽检分析报告"

# 查看可用数据表
d2t explore

# Markdown 转 Word
d2t convert report.md -o report.docx

# 训练数据管理
d2t training stats -f training_data.jsonl
d2t training import input.jsonl --output-name training_data.jsonl --mode a
d2t training export -i training_data.jsonl -o training_data_formatted.jsonl

# 模型评估
d2t evaluate test_data.jsonl -o eval_results.json --model qwen3:14b --backend ollama

# 云端微调（在 AutoDL 上运行）
python scripts/finetune_cloud.py
python scripts/finetune_cloud.py --prepare-only
```

## 环境配置

### 依赖安装

```bash
pip install -e .
```

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `qwen3:14b` | Ollama 模型名称 |
| `VLLM_API_URL` | — | vLLM/OpenAI 兼容 API 地址 |
| `VLLM_API_KEY` | — | vLLM API 密钥 |
| `CLOUD_API_URL` | — | 云端 API 地址 |
| `CLOUD_API_KEY` | — | 云端 API 密钥 |
| `INFERENCE_BACKEND` | `ollama` | 推理后端：ollama 或 vllm |
| `INTENT_USE_LLM` | `false` | 是否启用 LLM 增强意图识别 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### Ollama 部署

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:14b
ollama serve
```

## AutoDL 云端环境

微调训练和推理测试全程在 **AutoDL** 云 GPU 平台完成。

| 项目 | 值 |
|------|-----|
| 工作目录 | `/root/autodl-tmp/data-to-text/` |
| 基础模型 | `/root/autodl-tmp/Qwen3-14B` |
| LoRA 权重 | `./qwen3-lora-finetuned/` |
| 合并模型 | `./qwen3-lora-merged/` |
| 训练数据 | `training_data/training_data.jsonl` (250 条) |
| GPU | 32GB 显存的 NVIDIA GPU |

### AutoDL 环境初始化

```bash
source /etc/network_turbo
python -c "import torch; print(torch.__version__)"
pip install transformers peft datasets accelerate bitsandbytes
pip uninstall torchvision -y
```

### 往 AutoDL 传文件

使用 `scripts/deploy_to_autodl.py` 生成 heredoc 格式的部署命令：

```bash
python3 scripts/deploy_to_autodl.py           # 输出到终端
python3 scripts/deploy_to_autodl.py output.sh  # 输出到文件
```

### AutoDL 数据持久化

- `/root/autodl-tmp/` 下的文件在实例关机后保留（数据盘）
- 系统盘（`/root/` 其他目录）在关机后会清空

## 数据说明

### 数据源

- `自动分析报告数据模板/数据表/`：省级数据（CSV 格式，38+ 张表）
- `自动分析报告agent数据/数据表/`：市级数据（按城市分目录，含 xlsx 和 csv）

### 数据表命名规则

- 省级表：`jdcj_prov_{类型}_{时间戳}.csv` 或 `sheng_jdcj_{维度}_{类型}_{时间戳}.csv`
- 市级表：`fxjc_city_{类型}.xlsx` 或 `shi_fxjc_{维度}_{类型}.xlsx`

### 关键列名

| 列名 | 含义 |
|------|------|
| `niandu` | 年度 |
| `jdpc` | 监督抽检批次 |
| `hgpc` | 合格批次 |
| `bhgpc` | 不合格批次 |
| `bhgl` | 不合格率 |
| `sp_s_17` | 大类食品 |
| `sp_s_18` | 食品亚类 |
| `sp_s_19` | 食品次亚类 |
| `sp_s_20` | 细类食品 |
| `xiangmumingcheng` | 检测项目名称 |
| `xiangmufenlei` | 检测项目分类 |
| `trend_3y` | 三年趋势方向 |
| `p_value` | P 值 |
| `chi2_3y` | 三年卡方值 |
| `slope` | 线性回归斜率 |
| `r_squared` | 决定系数 |
| `wilson_ci_95` | Wilson 95% 置信区间 |
| `rate_trend` | 率趋势 |

完整字段说明见 `自动分析报告数据模板/sheng_jdcj_dictionary数据字典.xlsx`，可通过 `DataDictionary` 类查询。

## QLoRA 微调记录

2026-06-07 在 AutoDL 上完成 Qwen3-14B 4-bit QLoRA 微调：

| 参数 | 值 |
|------|-----|
| LoRA r | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| 学习率 | 2e-4 |
| Epochs | 3 |
| Batch size | 1 |
| 梯度累积 | 16 |
| 最大序列长度 | 1024 |
| 量化 | 4-bit NF4 (BitsAndBytes) |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 优化器 | paged_adamw_8bit |
| 训练时间 | 19分17秒 |
| 训练步数 | 48 步 |
| Loss 变化 | 0.9474 → 0.2319 |

## 关键技术笔记

### transformers 新版本 API 变化

训练脚本已适配新版 transformers 的以下 API 变更：
- `tokenizer=tokenizer` → `processing_class=tokenizer`（Trainer 参数）
- `torch_dtype=...` → `dtype=...`（from_pretrained 参数）
- `warmup_ratio=...` → `warmup_steps=...`（TrainingArguments 参数）

### CUDA OOM 解决方案

训练过程经历了多次 OOM，最终通过以下组合解决：
- 4-bit 量化（BitsAndBytes NF4）
- 移除 `prepare_model_for_kbit_training()`
- max_seq_length 从 2048 减到 1024
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `gradient_checkpointing_kwargs={"use_reentrant": False}`
- `optim="paged_adamw_8bit"`

### Qwen3 思考模式（Thinking Mode）

- Qwen3 默认在 chat template 中启用思考模式（在 `<think>` 标签内推理）
- SYSTEM_PROMPT 中不应包含"在心中推理"/"思考"等触发词
- 推理时必须：`tokenizer.apply_chat_template(..., enable_thinking=False)`
- 兜底：用 `re.sub(r'<\s*/?\s*think\s*>', '', generated)` 清理输出

### 推理方案

推理和训练统一使用 AutoDL 云 GPU 平台，不探索本地推理方案。用户本地 GPU 为 RTX 3050 Ti (4GB)，仅用于代码开发和文件准备。

## 测试

```bash
pytest tests/ -v
```

测试覆盖 7 个核心模块：意图识别、数据查询、统计计算、报告生成、事实校验、上下文组装、数据加载。

## 代码规范

- **Linter**：Ruff（配置见 `pyproject.toml`，target-version: py310, line-length: 120）
- **类型检查**：Python 类型注解 + Pydantic 模型
- **日志**：统一使用 `setup_logging()` 获取 logger

```bash
ruff check src/ tests/
```

## 已知问题

1. **数据源耦合**：当前数据源为静态 CSV 文件，未接入数据库
2. **市级数据适配**：`DataLoader` 主要针对省级 CSV 数据设计，市级 xlsx 数据需要额外适配
3. **意图识别**：规则引擎覆盖常见场景，但复杂/模糊指令仍依赖 LLM 增强
4. **降级模式**：当 LLM 不可用时降级生成的报告质量有限
5. **事实校验**：当前仅做数值偏差检测，未覆盖语义层面的逻辑一致性校验
6. **图表样式**：中文字体依赖系统安装的字体，不同环境可能显示异常
7. **训练数据**：当前训练数据量有限（250 条），微调效果有待更多高质量样本验证

## 快速上手

```bash
# 1. 安装依赖
pip install -e .

# 2. 确保 Ollama 运行（默认 localhost:11434）
ollama serve

# 3. 查看可用数据
d2t explore

# 4. 生成报告
d2t generate "帮我生成一份2025年广西监督抽检报告"

# 5. 查看输出
# Markdown: output/report.md
# Word: output/抽检分析报告.docx
# 图表: output/charts/
```
