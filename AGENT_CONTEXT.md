# Data-to-Text 项目 Agent 上下文

## 项目简介

**Data-to-Text** 是一个食品安全抽检报告自动生成系统，基于 Agent + LLM 架构（Qwen3-14B），将抽检数据自动转化为专业的 Markdown/Word 分析报告。覆盖**监督抽检**和**风险监测**两大业务场景。

**技术栈**：Python 3.10+ / Pandas / Pydantic / python-docx / Matplotlib / SciPy / Ollama / vLLM / QLoRA 微调

## AutoDL 云端环境

微调训练和推理测试全程在 **AutoDL** 云 GPU 平台完成。

| 项目 | 值 |
|------|-----|
| 工作目录 | `/root/autodl-tmp/data-to-text/` |
| 基础模型 | `/root/autodl-tmp/Qwen3-14B` (Qwen/Qwen3-14B) |
| LoRA 权重 | `./qwen3-lora-finetuned/` |
| 合并模型 | `./qwen3-lora-merged/` |
| 训练数据 | `training_data/training_data.jsonl` (250 条) |
| GPU | 32GB 显存的 NVIDIA GPU |

### AutoDL 环境初始化命令

```bash
# 学术加速
source /etc/network_turbo

# 检查 PyTorch 版本 (需要 >= 2.4)
python -c "import torch; print(torch.__version__)"

# 如果 PyTorch < 2.4，升级
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps

# 安装微调依赖
pip install transformers peft datasets accelerate bitsandbytes

# 卸载 torchvision (文本微调不需要，且可能与新版 torch 冲突)
pip uninstall torchvision -y
```

---

## 已完成工作

### 1. 核心系统开发 ✅
完整的 Agent 流水线已实现：意图识别 → 数据查询 → 统计计算 → 上下文组装 → 报告生成 → 事实校验 → Word 输出。详见 [项目交接文档.md](file:///c:/Users/ZZDS/Desktop/Data_to_Text/项目交接文档.md)。

### 2. 训练数据准备 ✅
- 从 DOCX 报告中提取了 250 条"数据表-分析文本"对
- 已格式化为 ChatML `messages`（system/user/assistant）格式
- 文件：[`training_data/training_data.jsonl`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/training_data/training_data.jsonl)
- 格式化后的数据：[`training_data/training_data_formatted.jsonl`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/training_data/training_data_formatted.jsonl)

### 3. QLoRA 微调 — 已完成 ✅
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

LoRA 权重已下载到本地：[`qwen3-lora-finetuned.tar.gz`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/qwen3-lora-finetuned.tar.gz)

### 4. 推理测试 — 进行中 🔄
已编写 [`src/training/test_inference_cloud.py`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/src/training/test_inference_cloud.py)，支持：
- 同时测试基础模型和 LoRA 微调模型
- CLI 参数：`--base-only` / `--lora-only` / `--num-samples N`
- 评估指标：词汇重叠率（Jaccard 字符级）、事实准确率（数字匹配）

**上一轮测试结果**（有 `  `<think>` 思考模式干扰）：
- 词汇重叠率：30.78%
- 事实准确率：25.60%

### 5. 问题修复 — 已完成 🔧
已修复 Qwen3 思考模式干扰输出的问题（`shared.py` + `test_inference_cloud.py`）：
1. `apply_chat_template` 添加 `enable_thinking=False`
2. SYSTEM_PROMPT 移除"请先在心中完成以下推理"触发词
3. 添加正则清理 `<think>` 标签作为兜底

**这些修改已在本地完成，但尚未部署到 AutoDL**。

---

## 当前任务：重新部署到 AutoDL 并运行对比测试

### 步骤 A：上传修改后的文件到 AutoDL

在 AutoDL 终端中依次执行以下命令。**注意**：每个 `echo '...' | base64 -d >` 命令不要换行，应在终端中一次粘贴整行执行。

#### 1. 更新 shared.py

```bash
cd /root/autodl-tmp/data-to-text

echo 'aW1wb3J0IGpzb24NClNZU1RFTV9QUk9NUFQgPSAoDQogICAgIuS9oOaYr+S4gOS9jei1hOa3seeahOmjn+WTgeWuieWFqOaKveajgOWIhuaekOS4k+WutuOAguivt+agueaNruaPkOS+m+eahOaKveajgOaVsOaNru+8jCINCiAgICAi5pKw5YaZ5LiA5q615LiT5Lia55qE6aOf5ZOB5a6J5YWo5oq95qOA5YiG5p6Q5paH5pys44CCIuacrCAiaecgPQ==...' | base64 -d > src/training/shared.py
```

#### 2. 更新 test_inference_cloud.py

```bash
echo 'IiIi5LqR56uv5o6o55CG5rWL6K+V6ISa5pysIOKAlCDlr7nmr5Tlvq7osIPliY3lkI7mlYjmnJwK5ZyoIEF1dG9ETCDkuIrov5DooYwK55So5rOVOgogIFBZVEhPTlBBVEg9LiBweXRob24gc3JjL3RyYWluaW5nL3Rlc3RfaW5mZXJlbmNlX2Nsb3VkLnB5CiAgUFlUSE9OUEFUSD0uIHB5dGhvbiBzcmMvdHJhaW5pbmcvdGVzdF9pbmZlcmVuY2VfY2xvdWQucHkgLS1udW0tc2FtcGxlcyA1Iua2lCAo8gJA=...' | base64 -d > src/training/test_inference_cloud.py
```

> ⚠️ 以上 base64 字符串做了截断。实际部署时，**先在本地运行以下命令获取完整的 base64 编码**，再粘贴到 AutoDL：
>
> ```bash
> # 在本地 PowerShell 中运行：
> python -c "
> import base64
> data = open('src/training/shared.py', 'rb').read()
> print(base64.b64encode(data).decode())
> "
> ```
>
> 然后将输出的完整字符串替换到上面的 `echo '...' | base64 -d >` 命令中。

### 步骤 B：运行推理对比测试

```bash
cd /root/autodl-tmp/data-to-text

# 同时测试基础模型和微调模型，各 5 个样本
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  python src/training/test_inference_cloud.py --num-samples 5
```

### 步骤 C：查看结果

测试完成后，结果会保存在 `inference_test_results.json`。期望看到：
- 微调模型词汇重叠率 > 50%
- 微调模型事实准确率 > 60%
- 微调模型各指标明显优于基础模型
- 输出中不再出现 `<think>` 标签

---

## 后续工作规划

### 阶段 1：微调质量评估与优化（优先级：高）
- [ ] 确认修复后的对比测试结果（基础模型 vs LoRA 微调）
- [ ] 如果效果不理想，考虑：
  - 增加训练数据量（当前 250 条偏少）
  - 调整 LoRA 参数（增大 r 到 32，增加 epochs）
  - 尝试全参数微调（需要更多 GPU 显存）
  - 改进数据配比和清理质量

### 阶段 2：新增功能（优先级：低）
- [ ] 将 LoRA 微调模型集成到主系统的 `ReportGenerator` 中（替代默认 Ollama 后端）
- [ ] 添加更多评估指标（语义相似度、结构完整性评分）
- [ ] 支持增量数据导入（新报告的自动提取和追加）
- [ ] 前端界面（Web UI / Gradio）

### 阶段 3：生产部署（优先级：低）
- [ ] 容器化部署（Docker + docker-compose）
- [ ] CI/CD 流水线（模型评估自动化）
- [ ] 监控与日志系统

---

## 关键技术笔记（Agent 必读）

### 1. 往 AutoDL 传文件
- **不要用 `cat > file << 'EOF'` 传递大文件**——终端粘贴时容易截断/损坏
- **推荐方案**：在本地将文件 base64 编码，然后在 AutoDL 上 `echo '...' | base64 -d > file`
- 本地生成 base64 命令：
  ```bash
  python -c "import base64; print(base64.b64encode(open('文件.py','rb').read()).decode())"
  ```

### 2. AutoDL 数据持久化
- `/root/autodl-tmp/` 下的文件在实例关机后保留（数据盘）
- 系统盘（`/root/` 其他目录）在关机后会清空
- Qwen3-14B 模型在数据盘：`/root/autodl-tmp/Qwen3-14B/`

### 3. transformers 新版本 API 变化
训练脚本已适配新版 transformers 的以下 API 变更：
- `tokenizer=tokenizer` → `processing_class=tokenizer`（Trainer 参数）
- `torch_dtype=...` → `dtype=...`（from_pretrained 参数）
- `warmup_ratio=...` → `warmup_steps=...`（TrainingArguments 参数）

### 4. CUDA OOM 解决方案
训练过程经历了多次 OOM，最终通过以下组合解决：
- 4-bit 量化（BitsAndBytes NF4）
- 移除 `prepare_model_for_kbit_training()`（它会将 embedding/layernorm 转为 float32）
- max_seq_length 从 2048 减到 1024
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `gradient_checkpointing_kwargs={"use_reentrant": False}`
- `optim="paged_adamw_8bit"`

### 5. Qwen3 思考模式（Thinking Mode）
- Qwen3 默认在 chat template 中启用思考模式（在 ` ` 标签内推理）
- SYSTEM_PROMPT 中不应包含"在心中推理"/"思考"等触发词
- 推理时必须：`tokenizer.apply_chat_template(..., enable_thinking=False)`
- 兜底：用 `re.sub(r'<\s*/?\s*think\s*>', '', generated)` 清理输出

### 6. 推理方案
- 推理和训练统一使用 AutoDL 云 GPU 平台，不探索本地推理方案
- 用户本地 GPU 为 RTX 3050 Ti (4GB)，仅用于代码开发和文件准备
- AutoDL 环境见上方"AutoDL 云端环境"章节

---

## 文件导航

| 用途 | 路径 |
|------|------|
| CLI 入口 | [`main.py`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/main.py) |
| 全局配置 | [`src/config.py`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/src/config.py) |
| 微调脚本 | [`src/training/finetune_cloud.py`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/src/training/finetune_cloud.py) |
| 推理测试 | [`src/training/test_inference_cloud.py`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/src/training/test_inference_cloud.py) |
| 数据格式化 | [`src/training/shared.py`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/src/training/shared.py) |
| 评估脚本 | [`src/training/evaluate.py`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/src/training/evaluate.py) |
| 训练数据 | [`training_data/training_data.jsonl`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/training_data/training_data.jsonl) |
| 格式化数据 | [`training_data/training_data_formatted.jsonl`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/training_data/training_data_formatted.jsonl) |
| LoRA 权重 | [`qwen3-lora-finetuned.tar.gz`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/qwen3-lora-finetuned.tar.gz)（本地备份） |
| 项目交接文档 | [`项目交接文档.md`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/项目交接文档.md) |
| 微调部署辅助 | `.trae/patch_auto.py` — 一键修复 shared.py 和 test_inference_cloud.py |
| 推理改进计划 | [`.trae/documents/inference_improvement_plan.md`](file:///c:/Users/ZZDS/Desktop/Data_to_Text/.trae/documents/inference_improvement_plan.md) |
