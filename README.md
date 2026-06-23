# 实训作业：面向复杂规律推理的多模态大模型后训练

## 1. 课题背景与任务

本课题关注科学发现中的一个核心问题：**模型能否从图像、数据点、曲线等多模态观测信息中，自动归纳出背后的数学公式与科学规律。**

本次实训以 **「根据数据点和函数图像进行公式发现」** 为核心任务：

> 给定一条未知函数 `y = f(x)` 的**曲线图像**和一组**采样数据点 `(x, y)`**，模型需要推理出其背后的**解析表达式**（numpy 风格，如 `2 * np.sin(3 * x + 1)`）。

你需要围绕**数据构造、任务设计、后训练方法、评估**展开实践：构造不同类型/难度的函数规律样本，设计从简单函数到组合函数、从干净数据到含噪数据的难度层级，并探索 **SFT / RL 等后训练方式**如何提升模型的多步推理与假设迭代能力，最终在隐藏测试集上**刷榜**。

---

## 2. 模型限制（重要）

- **最终提交方案必须基于 `Qwen3-VL-8B`**（base / instruct / thinking 版本均可）。
- **数据合成、蒸馏、拒绝采样等辅助环节：可使用任意开源模型**。
  - 但这些模型**仅可用于构造/筛选训练数据**，**不能作为最终提交的方案模型**。
  - **不能通过权重编辑等手段**把其他模型的能力直接注入最终模型。
- **禁止调用外部模型 API（如 GPT、Claude 等）辅助训练。**
- 实训期间鼓励多用大模型协助：头脑风暴、写代码、debug 等都可以。

---

## 3. 数据集

所有数据**均包含函数图像**，**目标均为预测公式**；**数据语言为英文**。

| 集合 | 是否公开 | 规模 | 说明 |
|---|---|---|---|
| **训练集** | 自备 | 不限 | 可用互联网公开免费资料或**自行合成**，来源与数量不限 |
| **验证集 (dev)** | ✅ 题目+答案 | 300 | **本发布包提供**，可自行评测 |

最终榜单分数在**隐藏测试集**上计算，通过提交模型由官方统一评测得到。

### dev 数据格式
`data/task/dev/samples.jsonl`，每行一个 JSON 样本，关键字段：

| 字段 | 含义 |
|---|---|
| `id` | 样本唯一 ID |
| `image_path` | 曲线图相对路径（`images/<id>.png`） |
| `data_points_text` | 给模型看的采样点 `[[x, y], ...]` |
| `function_hints` | 候选函数提示（**可能含干扰项，并非全部用到**） |
| `expression_str` / `expression_numpy` | **真值表达式**（裸 / numpy 风格） |
| `test_points` | 评测用留出点 `[[x, y], ...]`（最多 50 个），R² 在此计算 |
| `image_x_range` | 图像 x 轴范围 |

---

## 4. 评价指标

**每道题的评价指标为正确率 `R²@0.99`**：用模型预测的表达式在该题 `test_points` 的 `x` 上求值，与真值 `y` 计算决定系数 R²，**`R² ≥ 0.99` 记为答对，否则记为答错**。最终分数 = 答对题数 / 总题数。

R² 定义：`R² = 1 - SS_res / SS_tot`，其中 `SS_res = Σ(ŷ-y)²`，`SS_tot = Σ(y-ȳ)²`。
预测无法求值 / 抽取失败 / 出现 nan/inf 的样本，记为答错。

---

## 5. 评测方式（官方 `eval.py`）

本发布包内的 `eval.py` 即**官方评测脚本**（基于 **vLLM**）。其工作方式：

1. 用 vLLM 加载你提交的模型，对每条样本送入**曲线图 + 文本（候选函数 + 采样点）**。
2. 通过 **tool calling**（`submit_expression(expression)` 工具）让模型提交 numpy 表达式；
   配合 vLLM 的 reasoning / tool-call parser 解析 `<think>` 与工具调用，并有正则兜底。
3. 在 `test_points` 上计算 R²，输出 `acc@0.99`（及 0.95/0.90/0.80 参考）、mean/median R²。

**默认 prompt 模板**（可被 `prompt.txt` 覆盖，见第 6 节）包含三个占位符：
`{function_hints}`、`{data_points}`、`{axis_note}`。

### 在 dev 上自评
```bash
# 1) 安装依赖（如未安装）
pip install vllm

# 2) 评测你的模型（在 dev 集上）
python eval.py <你的模型权重目录> --split dev \
    --tp 1 \
    --reasoning-parser qwen3 \
    --tool-call-parser hermes
```
结果写入 `eval_outputs/<模型名>/`：`eval_summary_dev.json`（分数摘要）与 `eval_results_dev.jsonl`（逐题 `predicted_expr` / R²，便于排错），终端同时打印分档表。
> parser 名需与你的模型对话/工具格式匹配；Qwen3 系一般用 `--reasoning-parser qwen3 --tool-call-parser hermes`。

---

## 6. 提交要求（务必遵守）

- **提交模型权重路径**。如需自定义 prompt，**在模型目录下放 `prompt.txt`**，评测脚本默认读取（占位符见第 5 节，示例见 `prompt_example.txt`）。
- **不能修改、不能 hack 评测脚本**。评测由官方统一执行，必须按要求提交**模型 + 可选 `prompt.txt`**，不要依赖对脚本的改动。
- **复现环境与官方一致**。为保证榜单及时更新，你的模型在 **dev 规模数据（300 条）、单卡** 下需在 **30 分钟内**输出结果。

---

## 7. 时间线

- **最终提交截止：【6.25 24:00】**

---

## 8. 最终交付：报告&模型路径

提交一份**实训报告（report）**，包含但不限于：
1. **总体介绍**：最终方案简述、创新点、实验结果。
2. **方法实现**：多方案请拆分；每个方案含思路、方法描述、具体细节（数据来源、提示词、超参等）。无效但重要的方法也可提及，并注明是否最终采用。
3. **实验及结果**：实验设计、结果、验证集分数。
4. **总结**：本次发现；建议额外总结方法的应用价值与潜在缺陷。
5. **分工（重要）**：详细列举每人贡献。

---

## 9. 发布包内容
```
HW3_release/
  README.md            作业说明
  eval.py              评测脚本
  prompt_example.txt   自定义 prompt 示例
  data/task/dev/
    samples.jsonl      300 条验证样本 (含真值) 
    images/            300 张曲线图
```

---

## 10. 本仓库当前干净工作流

本仓库只保留原始 dev 数据、通用数据生成器、通用 LoRA 训练/合并脚本和评测分析工具。新实验从这里开始：

```bash
bash scripts/generate_v2_data.sh
bash scripts/run_lora_sft.sh
bash scripts/merge_lora.sh
```

核心技术路线见 `docs/technical_route.md`；完整命令说明见 `docs/clean_workflow.md`；新开对话的上下文提示词见 `docs/new_conversation_prompt.md`。

当前主线不是把任务做成 29 类模板分类，而是训练模型形成稳定的符号回归流程：先用图像判断周期性、对称性、零点/极值、包络和趋势等粗结构，再生成多个候选函数族与参数，最后用 Reference points 做数值代入检验，误差足够小才提交 tool call。

---

## 11. RL / DPO 后训练启动方式

当前推荐使用 guarded DPO 流程继续做 RL 后训练。该流程会把训练拆成多个短 phase，每个 phase 训练后自动 merge、dev eval，并只接受超过当前最好 `acc@0.99` 的 adapter；如果分数低于当前 SFT baseline，会自动停止，避免长时间训练把模型训坏。

由于项目盘空间紧张，推荐使用 global storage wrapper。它会把 preference 数据、adapter checkpoint、merge 后完整模型、eval logs 和 eval summary 全部写到：

```text
/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/
```

### 一键启动推荐配置

如果终端或 SSH 会话不稳定，优先用后台启动方式：

```bash
cd /inspire/hdd/project/generative-large-model/public/ywy/hw3-of-lpf
git pull origin rl

source envs/rl_gpu/activate.sh

CUDA_VISIBLE_DEVICES=0 \
RUN_NAME=guarded_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_rl_dpo_global_guarded_detached.sh
```

启动后按终端提示查看日志：

```bash
tail -f /inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/run_full.log
tail -f /inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/guarded_status.jsonl
```

如果需要前台运行，使用：

```bash
cd /inspire/hdd/project/generative-large-model/public/ywy/hw3-of-lpf
git pull origin rl

source envs/rl_gpu/activate.sh

CUDA_VISIBLE_DEVICES=0 \
RUN_NAME=guarded_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_rl_dpo_global_guarded_train.sh
```

前台脚本同样会把完整 stdout/stderr 记录到：

```text
/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/run_full.log
```

注意：checkpoint、merge 模型、eval 输出和 cache 都会写到 global storage；`TMPDIR` 会使用较短的 `/tmp/hw3rl_*` 路径，因为 vLLM/ZMQ 的 IPC socket 路径长度不能超过约 107 字符。

默认配置面向单张 140G GPU：

```text
PHASES=8
PHASE_STEPS=10
MAX_PREF_SAMPLES=3000
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-7
DPO_BETA=0.02
SFT_LOSS_COEF=0.05
REJECTION_MODE=hardest
```

等价地，effective batch size 是 `8 x 4 = 32`。这里故意使用较小 learning rate、较小 DPO beta，并混入 SFT loss，目标是让 RL 稳定微调当前最好 SFT checkpoint，而不是快速偏离原模型。

### 长时间训练配置

如果前几轮 phase 稳定不掉点，可以增加 phase 数量：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_NAME=guarded_long_$(date +%Y%m%d_%H%M%S) \
PHASES=20 \
PHASE_STEPS=10 \
MAX_PREF_SAMPLES=3000 \
PER_DEVICE_TRAIN_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=4 \
LEARNING_RATE=2e-7 \
DPO_BETA=0.02 \
SFT_LOSS_COEF=0.05 \
bash scripts/run_rl_dpo_global_guarded_train.sh
```

不建议一开始直接把 `PHASE_STEPS` 或 `LEARNING_RATE` 开大。之前 `lr=5e-6, beta=0.1, 100 steps` 的 DPO 会导致 dev 明显掉点；当前策略是小步训练、频繁 eval、只保留提升的 phase。

### 查看结果

每次运行会生成一个独立 run 目录，例如：

```text
/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/guarded_20260623_120000/
```

重点文件：

```text
guarded_status.jsonl                    每个 phase 的分数与 accept/reject 决策
phase_*/adapter/                        每个 phase 训练出的 LoRA adapter
phase_*/eval_logs/                      merge 与 eval 日志
merged_models/qwen3_vl_dpo_guarded_*    每个 phase 的 merge 后完整模型
eval_outputs/*/eval_summary_dev.json    dev summary
eval_outputs/*/eval_results_dev.jsonl   逐样本预测与 R²
```

实时看训练进度：

```bash
tail -f /inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/guarded_status.jsonl
```

### 手动评测某个 adapter

如果需要单独评测某个 adapter，也把 merge 和 eval 输出放到 global storage：

```bash
cd /inspire/hdd/project/generative-large-model/public/ywy/hw3-of-lpf
source envs/rl_gpu/activate.sh

ADAPTER_DIR=/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/phase_1/adapter \
MERGED_DIR=/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/manual_eval/merged_model \
EVAL_OUTPUT_ROOT=/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/manual_eval/eval_outputs \
LOG_DIR=/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/manual_eval/logs \
bash scripts/run_rl_dpo_eval.sh
```

当前最好 SFT dev 分数约为 `acc@0.99 = 68.0%`。RL phase 只有超过这个基线并被 guarded script 标记为 `accept`，才值得作为新的候选提交模型继续观察。
