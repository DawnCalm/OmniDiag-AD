# [OmniDiag-AD] 🚗 👁️ 🧠
A VLM-based explainable autonomous driving system. It bridges BEVFusion features with LLMs via a lightweight connector for multi-modal perception failure diagnostics.
## 📌 Project Overview (项目简介)
This repository contains the official implementation of a **VLM-based Perception Trustworthiness and Explainability Evaluation System** for autonomous driving. 

While traditional 3D detection models (like BEVFusion) act as "black boxes", our system introduces a powerful diagnostic brain (powered by Large Vision-Language Models like Qwen2-VL) to dissect, interpret, and attribute perception failures. By intercepting intermediate multi-modal BEV features (Camera, LiDAR, and Fused) and aligning them with the LLM's text space via a lightweight connector, the system outputs human-readable, multi-granularity diagnostics.

本项目是一个面向自动驾驶的**基于 VLM 的感知可信性与可解释性评估系统**。不同于传统 BEV 模型的黑盒输出，本系统通过截获单模态与融合态的底层 BEV 特征，利用轻量级 Connector 将其对齐到大语言模型（如 Qwen2-VL）的特征空间。系统能够像“主治医师”一样，输出多粒度的场景描述、感知失败的模态归因（如究竟是视觉逆光还是雷达稀疏导致漏检），以及基于证据深度学习（EDL）的可信度建议。

---

## ✨ Key Features (核心特性)

* **🔍 Multi-Modal Feature Interception**: Hooks into the BEVFusion backbone to extract pure camera, pure LiDAR, and fused BEV feature maps before the detection head.
* **⚡ Lightweight BEV-to-Language Connector**: An efficient projection module (≤ 5M parameters) that translates high-dimensional BEV tensors into LLM-understandable tokens without overwhelming VRAM.
* **💬 Multi-Granularity Explainability**: Generates DriveLM/OmniDrive-style QA conversations covering:
    1.  *Scene Description*: What is happening in the environment?
    2.  *Perception Attribution*: Why did a detection fail? (e.g., "Camera missed the object due to backlighting, but LiDAR captured the reflection").
    3.  *Trustworthiness Assessment*: Uncertainty evaluation powered by Evidential Deep Learning (EDL).
* **🚀 Three-Stage Training Pipeline**: Structured training including BEVFusion+EDL pre-training, Connector alignment, and QLoRA instruction fine-tuning to prevent catastrophic forgetting.

---

## 🧭 Workflow

### Local conservative path (`4GB VRAM`)

1. Export BEV features, predictions, renders, and EDL maps:

```bash
python tools/export_bev_features.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_mini.yaml \
  ./pretrained/bevfusion-det.pth \
  --output-dir outputs/bev_vlm \
  --split val \
  --max-samples 8
```

2. Build ShareGPT samples, flat `jsonl`, and an enriched manifest with missed-GT anchors:

```bash
python tools/create_bev_vlm_data.py \
  outputs/bev_vlm/manifest_val.jsonl \
  --output outputs/bev_vlm/bev_vlm_sharegpt.json
```

3. Train the local Stage 2 prototype with the lightweight MLP connector:

```bash
python tools/train_bev_vlm_stage2.py \
  outputs/bev_vlm/bev_vlm_sharegpt_flat.jsonl \
  --output-dir outputs/bev_vlm/stage2_local \
  --epochs 3 \
  --batch-size 2
```

4. Run lightweight evaluation and visualization:

```bash
python tools/eval_bev_vlm.py predictions.jsonl outputs/bev_vlm/bev_vlm_sharegpt_flat.jsonl
python tools/visualize_bev_vlm_sample.py outputs/bev_vlm/bev_vlm_sharegpt.json --output outputs/bev_vlm/sample_panel.png
```

### Server enhancement path

Use the same flat dataset and enriched manifest, then switch to the server-stage scripts:

- `tools/train_bev_vlm_stage2_server.py`
- `tools/train_bev_vlm_stage3_server.py`

These scripts are designed for a server environment with `transformers`, `peft`, and optionally `bitsandbytes` installed.
