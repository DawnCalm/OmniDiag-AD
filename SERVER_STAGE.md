# Server Training Plan

更新日期：`2026-05-20`

本文档把服务器阶段落成两个可执行入口：

- Stage 2: `Q-Former` 对齐
- Stage 3: `LoRA / QLoRA` 联训

## 1. 目标

### Stage 2

冻结：

- `BEVFusion` 主干
- 视觉编码器
- 文本 LLM

只训练：

- `Q-Former connector`

目标是先让 `BEV tokens + crop/render` 对齐到 LLM embedding space。

### Stage 3

冻结：

- `BEVFusion` 主干
- 视觉编码器

训练：

- `Q-Former connector`
- `LLM LoRA adapters`

可选：

- `QLoRA` 4-bit 量化加载 LLM

目标是把 Stage 2 学到的 BEV 对齐能力继续接到真正的 LLM 指令微调上。

## 2. 现有脚本入口

- Stage 2:
  - [tools/train_bev_vlm_stage2_server.py](/root/bevfusion/tools/train_bev_vlm_stage2_server.py)
- Stage 3:
  - [tools/train_bev_vlm_stage3_server.py](/root/bevfusion/tools/train_bev_vlm_stage3_server.py)

相关模型与训练工具：

- [bev_vlm/server_models.py](/root/bevfusion/bev_vlm/server_models.py)
- [bev_vlm/server_training.py](/root/bevfusion/bev_vlm/server_training.py)

## 3. 输入数据

当前推荐直接复用本地阶段已经生成好的数据：

- [outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl)
- [outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl)

这些样本已经包含：

- `question`
- `answer`
- `task_type`
- `images`
- `bev_features`
- `metadata`

## 4. 服务器依赖

按 Hugging Face 官方文档的常见组合，服务器建议至少安装：

```bash
pip install -U transformers accelerate peft
```

如果要启用 `QLoRA`，再安装：

```bash
pip install -U bitsandbytes
```

如果要加载某些 tokenizer / model 还可能需要：

```bash
pip install -U sentencepiece einops
```

## 5. 推荐环境变量

先在服务器上明确三件事：

```bash
export DATA_JSONL=/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl
export LLM_MODEL=<your-llm-model>
export VISION_MODEL=<your-vision-encoder>
```

说明：

- `LLM_MODEL`：文本 LLM，例如指令模型
- `VISION_MODEL`：独立视觉编码器，例如 ViT / SigLIP / CLIP 类模型

当前代码设计是：

- 独立视觉编码器负责 `crop / render`
- `Q-Former` 负责聚合 `BEV tokens + visual tokens`
- LLM 负责文本生成

## 6. Stage 2 命令

### 6.1 最小 smoke run

```bash
python tools/train_bev_vlm_stage2_server.py \
  "$DATA_JSONL" \
  --llm-model "$LLM_MODEL" \
  --vision-model "$VISION_MODEL" \
  --output-dir outputs/bev_vlm/stage2_server_smoke \
  --epochs 1 \
  --batch-size 1 \
  --max-samples 32 \
  --smoke-steps 8 \
  --gradient-checkpointing \
  --torch-dtype float16 \
  --device cuda
```

### 6.2 正式对齐训练

```bash
python tools/train_bev_vlm_stage2_server.py \
  "$DATA_JSONL" \
  --llm-model "$LLM_MODEL" \
  --vision-model "$VISION_MODEL" \
  --output-dir outputs/bev_vlm/stage2_server \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --gradient-checkpointing \
  --torch-dtype float16 \
  --device cuda
```

Stage 2 训练完成后，重点保留：

- `stage2_server_best.pt`
- `qformer_connector.pt`
- `stage2_server_config.json`

如果 Stage 3 只需要 connector，优先用：

- `qformer_connector.pt`

## 7. Stage 3 命令

### 7.1 LoRA 联训

```bash
python tools/train_bev_vlm_stage3_server.py \
  "$DATA_JSONL" \
  --llm-model "$LLM_MODEL" \
  --vision-model "$VISION_MODEL" \
  --stage2-checkpoint outputs/bev_vlm/stage2_server/qformer_connector.pt \
  --output-dir outputs/bev_vlm/stage3_server_lora \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --gradient-checkpointing \
  --torch-dtype float16 \
  --device cuda
```

### 7.2 QLoRA 联训

```bash
python tools/train_bev_vlm_stage3_server.py \
  "$DATA_JSONL" \
  --llm-model "$LLM_MODEL" \
  --vision-model "$VISION_MODEL" \
  --stage2-checkpoint outputs/bev_vlm/stage2_server/qformer_connector.pt \
  --output-dir outputs/bev_vlm/stage3_server_qlora \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --gradient-checkpointing \
  --torch-dtype float16 \
  --use-qlora \
  --device cuda
```

Stage 3 训练完成后，重点保留：

- `stage3_server_best.pt`
- `qformer_connector.pt`
- `lora_adapter/`
- `stage3_server_config.json`

## 8. 当前脚本支持的关键参数

### Stage 2 / Stage 3 通用

- `--prompt-mode {question_only,structured}`
  - `question_only`：更接近最终服务器路线
  - `structured`：更适合 smoke test 或排查
- `--gradient-checkpointing`
- `--gradient-accumulation-steps`
- `--torch-dtype {float16,bfloat16,float32}`
- `--max-samples`
- `--smoke-steps`

### Stage 3 专有

- `--use-qlora`
- `--lora-r`
- `--lora-alpha`
- `--lora-dropout`
- `--lora-target-modules`
- `--stage2-checkpoint`

## 9. 当前实现约束

### 9.1 这不是最终多模态大模型封装

当前服务器设计仍是：

- 单独视觉编码器
- 单独 `Q-Former`
- 单独文本 LLM

它不是直接调用现成 VLM 的统一 processor / chat template。

### 9.2 本地 fallback 不应带到服务器

本地阶段为了避免塌缩，在 [tools/predict_bev_vlm_stage2.py](/root/bevfusion/tools/predict_bev_vlm_stage2.py) 中对 `trust / miss_summary` 使用了 `structured_fallback`。

服务器阶段不要继续依赖这个逻辑。

服务器阶段应让：

- `trust`
- `miss_summary`
- `scene`
- `attribution_object`

全部回到真实 LLM 生成。

### 9.3 还没有服务器推理脚本

当前已经有：

- Stage 2 训练入口
- Stage 3 训练入口

但服务器侧还没有单独的推理/评测脚本，需要后续补：

- 服务器版 prediction export
- 服务器版 generate + eval
- 服务器版样例面板

## 10. 推荐开发顺序

1. 先在服务器装依赖
2. 跑 Stage 2 smoke
3. 跑 Stage 2 正式训练
4. 用 `qformer_connector.pt` 启动 Stage 3
5. 先尝试 LoRA，再尝试 QLoRA
6. 补服务器推理脚本和评测脚本

## 11. 当前阶段结论

服务器阶段现在已经具备“可执行入口”的最低条件：

- Stage 2 对齐入口已具备
- Stage 3 LoRA/QLoRA 入口已具备
- Stage 2 -> Stage 3 的 connector checkpoint 衔接已具备
- tokenizer / image_processor / adapter 保存路径已具备

下一位 Codex 不需要再从零搭脚手架，可以直接开始：

- 安装依赖
- 选择服务器模型
- 运行 Stage 2
- 运行 Stage 3

