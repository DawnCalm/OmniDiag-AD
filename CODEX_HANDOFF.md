# Codex Handoff

更新日期：`2026-05-16`

## 1. 项目当前目标

当前项目已经完成“本地最小闭环”验证，下一阶段应转向服务器端继续做正式版本：

- 本地阶段目标：`BEVFusion -> QA 数据 -> 轻量 Stage 2 -> 预测/评测/可视化`
- 服务器阶段目标：`Q-Former + LoRA/QLoRA + 更强 LLM/MLLM`

本地阶段已经确认可运行，但本地小模型不是最终方案。真正的论文级问答、归因和泛化能力，应在服务器阶段完成。

## 2. 已完成工作

### 2.1 感知侧与 EDL-MVP

- 已修改 [mmdet3d/models/heads/bbox/transfusion.py](/root/bevfusion/mmdet3d/models/heads/bbox/transfusion.py)
  - 增加 `evidence -> uncertainty` 输出
  - 增加 `query_evidence / query_uncertainty / query_confidence`
  - object uncertainty 已改成“只基于选中类别”的版本
- 已修改 [mmdet3d/models/fusion_models/bevfusion.py](/root/bevfusion/mmdet3d/models/fusion_models/bevfusion.py)
  - 允许推理阶段把 EDL 相关结果继续往外传

### 2.2 特征导出与 manifest

- 已修改 [tools/export_bev_features.py](/root/bevfusion/tools/export_bev_features.py)
  - 导出 `camera / lidar / fused` 三路 `BEV .pt`
  - 导出三路 `BEV render`
  - 导出 `pred.json`
  - 导出 `edl_evidence_path / edl_render_path`
  - 写出 [manifest_val.jsonl](/root/bevfusion/outputs/bev_vlm/manifest_val.jsonl)
- `scene_uncertainty` 已从“dense 全图均值”改为“基于检测目标的 top-k uncertainty 聚合”
- 原始 dense 版本仍保留为调试字段

### 2.3 QA 数据构造

- 已修改 [tools/create_bev_vlm_data.py](/root/bevfusion/tools/create_bev_vlm_data.py)
  - 支持输出：
    - `ShareGPT json`
    - `flat jsonl`
    - `enriched manifest`
  - 当前推荐使用：
    - [bev_vlm_sharegpt_v2.json](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2.json)
    - [bev_vlm_sharegpt_v2_flat.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl)
    - [bev_vlm_sharegpt_v2_manifest.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl)
- 样本结构已改成“帧级 + 目标级”
  - 帧级：`scene / miss_summary / trust`
  - 目标级：每个漏检目标 1 条 `attribution_object`
- attribution 锚点逻辑：
  - 优先 `missed_gt`
  - 若没有明确漏检目标，再回退到低置信预测
- crop 逻辑：
  - 已改成“按目标方位先给 6 个相机排序，再做投影裁剪”
  - 不再固定只取 `CAM_FRONT`
  - 如果后向相机不可见，会回退到实际仍能看见该目标的其他相机
- 归因回答已改成基于目标局部 patch 的统计，而不是纯模板
  - 当前会利用目标对应区域的：
    - `camera / lidar / fused` 局部响应强度
    - 局部响应占比
    - 局部 `EDL` 不确定性和证据强度

### 2.4 本地 Stage 2 原型

- 新增 [bev_vlm](/root/bevfusion/bev_vlm) 模块
  - [connectors.py](/root/bevfusion/bev_vlm/connectors.py)
  - [local_stage2.py](/root/bevfusion/bev_vlm/local_stage2.py)
  - [server_models.py](/root/bevfusion/bev_vlm/server_models.py)
  - [server_training.py](/root/bevfusion/bev_vlm/server_training.py)
  - [data.py](/root/bevfusion/bev_vlm/data.py)
  - [tokenizer.py](/root/bevfusion/bev_vlm/tokenizer.py)
  - [metrics.py](/root/bevfusion/bev_vlm/metrics.py)
  - [visualization.py](/root/bevfusion/bev_vlm/visualization.py)
- 本地主线模型是：
  - `MLP Connector + GRU decoder`
  - 作用是验证 `BEV -> 文本` 的最小闭环
  - 不是最终的大模型方案

### 2.5 本地训练、推理、评测、可视化

- 已新增或修改：
  - [tools/train_bev_vlm_stage2.py](/root/bevfusion/tools/train_bev_vlm_stage2.py)
  - [tools/predict_bev_vlm_stage2.py](/root/bevfusion/tools/predict_bev_vlm_stage2.py)
  - [tools/eval_bev_vlm.py](/root/bevfusion/tools/eval_bev_vlm.py)
  - [tools/visualize_bev_vlm_sample.py](/root/bevfusion/tools/visualize_bev_vlm_sample.py)
- 当前本地最完整训练结果在：
  - [stage2_local_v3_ctx_e3_b1](/root/bevfusion/outputs/bev_vlm/stage2_local_v3_ctx_e3_b1)
- 训练配置：
  - 数据：`bev_vlm_sharegpt_v2_flat.jsonl`
  - `epochs=3`
  - `batch_size=1`
  - `hidden_size=256`
  - `token_grid=4x4`
- 训练曲线：
  - epoch 1: `train_loss=1.8585`, `val_loss=0.4951`
  - epoch 2: `train_loss=0.3557`, `val_loss=0.2498`
  - epoch 3: `train_loss=0.2203`, `val_loss=0.1912`

### 2.6 本地塌缩修复

- 本地小模型原始生成结果存在明显模式塌缩
  - `miss_summary` 原先 `81/81` 为同一句
  - `trust` 原先 `81/81` 为同一句
- 已做两层修复：
  - 在 [bev_vlm/data.py](/root/bevfusion/bev_vlm/data.py) 中加入结构化 `model_input_text`
  - 在 [tools/predict_bev_vlm_stage2.py](/root/bevfusion/tools/predict_bev_vlm_stage2.py) 中加入 `structured_fallback`
- 当前带 fallback 的本地预测文件：
  - [predictions_fallback.jsonl](/root/bevfusion/outputs/bev_vlm/stage2_local_v3_ctx_e3_b1/predictions_fallback.jsonl)
- 当前统计：
  - `miss_summary`: `81` 条中 `63` 种回答
  - `trust`: `81` 条中 `53` 种回答
- 解释：
  - 本地原型中：
    - `scene / attribution_object` 仍走模型生成
    - `miss_summary / trust` 使用 metadata 驱动的结构化解码
  - 这不是最终 LLM 回答，而是本地原型为了避免塌缩、先完成最小闭环的工程折中

### 2.7 可视化中文字体

- 已安装 `fonts-noto-cjk`
- [tools/visualize_bev_vlm_sample.py](/root/bevfusion/tools/visualize_bev_vlm_sample.py) 现在支持：
  - 自动优先搜索中文字体
  - `--font-path`
  - `BEV_VLM_FONT_PATH`
- 中文面板已恢复正常显示

## 3. 当前推荐使用的产物

### 3.1 数据

- 推荐训练数据：
  - [bev_vlm_sharegpt_v2_flat.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl)
- 推荐多轮数据：
  - [bev_vlm_sharegpt_v2.json](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2.json)
- 推荐富 manifest：
  - [bev_vlm_sharegpt_v2_manifest.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl)
- crop 目录：
  - [anchor_crops_v2](/root/bevfusion/outputs/bev_vlm/anchor_crops_v2)

### 3.2 本地模型与预测

- 当前本地最好用的 checkpoint 目录：
  - [stage2_local_v3_ctx_e3_b1](/root/bevfusion/outputs/bev_vlm/stage2_local_v3_ctx_e3_b1)
- 重点文件：
  - [stage2_local_best.pt](/root/bevfusion/outputs/bev_vlm/stage2_local_v3_ctx_e3_b1/stage2_local_best.pt)
  - [connector.pt](/root/bevfusion/outputs/bev_vlm/stage2_local_v3_ctx_e3_b1/connector.pt)
  - [tokenizer.json](/root/bevfusion/outputs/bev_vlm/stage2_local_v3_ctx_e3_b1/tokenizer.json)
  - [history.json](/root/bevfusion/outputs/bev_vlm/stage2_local_v3_ctx_e3_b1/history.json)
- 本地带 fallback 的预测：
  - [predictions_fallback.jsonl](/root/bevfusion/outputs/bev_vlm/stage2_local_v3_ctx_e3_b1/predictions_fallback.jsonl)

## 4. 现阶段没做完的工作

### 4.1 服务器阶段尚未真正启动

- [tools/train_bev_vlm_stage2_server.py](/root/bevfusion/tools/train_bev_vlm_stage2_server.py) 已存在
- [tools/train_bev_vlm_stage3_server.py](/root/bevfusion/tools/train_bev_vlm_stage3_server.py) 已存在
- [bev_vlm/server_models.py](/root/bevfusion/bev_vlm/server_models.py) 已有服务器侧骨架
- 但以下工作还没有真正跑：
  - `Q-Former` 训练
  - `LoRA/QLoRA` 联训
  - 真正多模态 LLM/MLLM 推理验证

### 4.2 本地生成仍不是最终回答质量

- `scene` 仍有明显模式化
- `attribution_object` 仍然比较模板化
- `trust / miss_summary` 虽已摆脱单一句塌缩，但本地版本依赖 `structured_fallback`
- 因此本地版只能说明：
  - 数据链通了
  - Connector 通了
  - 训练链和推理链通了
  - 不能说明“大模型归因能力已经完成”

### 4.3 评测与归因可视化仍较初级

- 当前文本评测只做了：
  - `BLEU-4`
  - `ROUGE-L`
  - 可选 `BERTScore`
- 当前归因热力图还没有真正做 attention-based attribution
- 现在的归因依据主要是：
  - 目标局部 patch 的响应统计
  - `EDL` 局部数值

### 4.4 EDL 仍是 MVP

- 当前 `EDL` 只做到工程版可导出、不确定性可用
- 还没有做更严格的：
  - uncertainty calibration
  - selective prediction / rejection analysis
  - 论文级 EDL 对比实验

## 5. 下一个 Codex 的推荐开发顺序

### 5.1 第一优先级：转服务器阶段

建议不要继续深挖本地 GRU 原型，而是直接推进服务器主线。

推荐顺序：

1. 检查并补齐服务器依赖
   - `transformers`
   - `peft`
   - `bitsandbytes`
   - `accelerate`
   - 可能还需要 `sentencepiece`、`einops`

2. 先跑通服务器版 Stage 2
   - 目标：冻结感知主干与大部分 LLM，只训练 `Q-Former + projection`
   - 推荐直接复用：
     - [bev_vlm_sharegpt_v2_flat.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl)
     - [bev_vlm_sharegpt_v2_manifest.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl)

3. 再做服务器版 Stage 3
   - 挂 `LoRA/QLoRA`
   - 联合训练 `Q-Former + LLM adapters`

4. 让 `trust / miss_summary` 回到真实 LLM 生成
   - 服务器阶段不应继续依赖 `structured_fallback`
   - fallback 只用于本地原型收口，不应成为最终方案

### 5.2 第二优先级：统一多模态输入

服务器阶段建议正式使用这些输入：

- `anchor crop`
- `camera / lidar / fused .pt`
- `camera / lidar / fused render`
- `edl_render`

当前数据已经把这些字段准备得差不多了，下一步主要是把服务器版 dataset/processor 对齐起来。

### 5.3 第三优先级：决定服务器侧模型路线

建议下一位 Codex 在服务器端优先确认：

- 最终用哪一个多模态 LLM / VLM
- `Q-Former` 是独立实现还是借已有实现改
- 视觉编码器是否冻结
- `BEV token` 与图像 token 的拼接方式

建议默认路线：

- 冻结视觉编码器
- 训练 `Q-Former`
- 对 LLM 挂 `LoRA/QLoRA`

## 6. 当前本地阶段的结论

可以认为“本地最小闭环已经达成”。

理由：

- 特征导出已通
- 数据集构造已通
- 本地 Stage 2 训练已通
- 本地预测导出已通
- 文本评测已通
- 可视化已通
- `miss_summary / trust` 的本地塌缩已工程化修复

但必须明确：

- 本地原型不是最终模型
- 本地回答不是论文最终结果
- 真正的重点已经转向服务器端的 `Q-Former + LoRA/QLoRA + 多模态 LLM`

## 7. 常用命令

### 7.1 重建 QA 数据

```bash
python tools/create_bev_vlm_data.py \
  outputs/bev_vlm/manifest_val.jsonl \
  --output outputs/bev_vlm/bev_vlm_sharegpt_v2.json \
  --flat-output outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl \
  --manifest-output outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl \
  --crop-dir outputs/bev_vlm/anchor_crops_v2
```

### 7.2 本地 Stage 2 训练

```bash
python tools/train_bev_vlm_stage2.py \
  outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl \
  --output-dir outputs/bev_vlm/stage2_local_v3_ctx_e3_b1 \
  --epochs 3 \
  --batch-size 1 \
  --device cuda
```

### 7.3 本地预测导出

```bash
python tools/predict_bev_vlm_stage2.py \
  outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl \
  --checkpoint-dir outputs/bev_vlm/stage2_local_v3_ctx_e3_b1 \
  --output outputs/bev_vlm/stage2_local_v3_ctx_e3_b1/predictions_fallback.jsonl \
  --batch-size 4 \
  --device cuda
```

### 7.4 文本评测

```bash
python tools/eval_bev_vlm.py \
  outputs/bev_vlm/stage2_local_v3_ctx_e3_b1/predictions_fallback.jsonl \
  outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl
```

### 7.5 可视化

```bash
python tools/visualize_bev_vlm_sample.py \
  outputs/bev_vlm/bev_vlm_sharegpt_v2.json \
  --output outputs/bev_vlm/sample_panel.png
```

