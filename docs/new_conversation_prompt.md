# New Conversation Prompt

你现在接手一个 Qwen3-VL-8B 多模态符号回归作业仓库。请不要沿用旧的零散 stage 训练思路，而是把它当作一个小型现代后训练项目来系统推进。

任务：输入一张函数曲线图、function hints、reference points，最终用 tool call `submit_expression(expression)` 输出 numpy 表达式。官方 dev 有 300 条，指标是 test points 上 `R2 >= 0.99` 的准确率。最终提交模型必须基于 Qwen3-VL-8B，不能用外部 API 训练，其他开源模型只能用于数据生成、蒸馏或筛选。

当前认知：
- 这不是单纯记忆公式模板，而是“看图识别函数族 + 猜参数 + 用 reference points 验证 + 自我修正”的任务。
- SFT 必须教会任务协议、合法 tool-call、基础函数族知识、视觉曲线特征、以及代入 reference points 检查答案的操作。
- 只在旧 LoRA 上继续叠训练上限有限。主线应从 Qwen3-VL-8B-Instruct 重新做一版干净、系统、高秩、视觉侧参与的 LoRA SFT。
- 旧模型结果可以作为 baseline 或 warm-start 对照，但不要让历史 stage 脚本主导新实验。
- 数据比模型本身更关键：需要覆盖基础 29 个函数族、hard negatives、多个候选函数族、近邻参数混淆、视觉风格扰动、以及 dev 错误族 repair/replay。

推荐路线：
1. 保持仓库干净：只使用通用脚本，不再新增 stage2/stage3/stageN 式脚本。
2. 生成 `data/sft_v2`：混合 tool-only 和 multi-family point-check 数据。
3. 从 Qwen3-VL-8B-Instruct fresh train 一版高秩 LoRA：`r=32/64`，bf16，4 x H100，视觉侧 LoRA 打开。
4. 评 dev，按函数族、难度、R2、null rate 做分析。
5. 用当前模型 self-sampling 生成多个候选表达式，程序自动打分，做 rejection-SFT 或 DPO/RFT。
6. 可用更强开源 VLM 做蒸馏，但 teacher 只提供候选搜索思路；最终表达式必须用程序验算筛选。

干净仓库中的关键命令：

```bash
cd /inspire/hdd/project/generative-large-model/public/hw3-of-lpf
source /inspire/hdd/project/generative-large-model/public/envs/hw3/bin/activate
git pull --ff-only origin main

bash scripts/generate_v2_data.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
LOAD_IN_4BIT=0 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_TRAIN_EPOCHS=1 \
LEARNING_RATE=1e-5 \
SAVE_STRATEGY=no \
bash scripts/run_lora_sft.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/merge_lora.sh

CUDA_VISIBLE_DEVICES=0 python eval.py \
  /inspire/hdd/project/generative-large-model/public/hw3-of-lpf/outputs/qwen3_vl_sft_merged \
  --split dev \
  --tp 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser hermes \
  --enforce-eager
```

工程原则：
- 不要把生成数据、模型权重、checkpoint 提交进 git。
- 所有训练输出放在 repo 内 `outputs/`，并定期删除 `checkpoint-*`。
- 每次实验都保留命令、数据 manifest、eval summary 和 per-sample analysis。
- 优先做可验证改进：数据配比、视觉侧 LoRA、hard-negative 质量、自动评分后训练。
- 不要为了短期 dev 分数牺牲最终泛化；dev 可以指导错误分布，但不能写死 dev 答案。
