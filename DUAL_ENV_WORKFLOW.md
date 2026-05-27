# Dual-Environment Workflow

更新日期：`2026-05-27`

## 1. 结论

当前项目建议采用“双环境两阶段”架构，不要把老 `BEVFusion` 感知环境和新大模型训练环境强行混在一起。

原因：

- 感知侧依赖老环境：
  - `Python 3.8`
  - `torch 1.10.1`
  - `CUDA 11.3`
  - 旧版 `mmcv / mmdet3d / spconv`
- 大模型侧需要新环境：
  - 新版 `transformers / peft / accelerate`
  - `bitsandbytes` for `QLoRA`
  - 对新架构 GPU 更友好的新 `PyTorch`

对于 Blackwell GPU，不建议在同一个环境里同时兼容这两套依赖。

## 2. 当前验证结果

本地老环境已确认：

- 旧环境版本仍可正常运行：
  - `torch 1.10.1`
  - `cuda 11.3`
- 已存在完整数据产物：
  - [outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl)
  - [outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl)
  - [outputs/bev_vlm/anchor_crops_v2](/root/bevfusion/outputs/bev_vlm/anchor_crops_v2)
  - [outputs/bev_vlm/val](/root/bevfusion/outputs/bev_vlm/val)
- 已额外做过一次重新导出 smoke run：
  - `export_bev_features.py` 在新目录成功导出 `2` 个样本
  - `create_bev_vlm_data.py` 在新目录成功生成 `ShareGPT / flat / manifest`

因此当前判断是：

- 老环境可以继续稳定负责“感知推理 + 中间结果导出”
- 服务器不必强行在大模型环境里再跑一次老版 `BEVFusion`

## 3. 环境分工

### 环境 A：老 BEVFusion 环境

职责：

- `BEVFusion` 推理
- EDL/MVP 导出
- 中间结果导出
- QA 数据构造

应产出的标准资产：

- `camera / lidar / fused BEV .pt`
- `pred.json`
- `edl_evidence`
- `camera/lidar/fused BEV render`
- `edl_render`
- `anchor crop`
- `bev_vlm_sharegpt_v2_flat.jsonl`
- `bev_vlm_sharegpt_v2_manifest.jsonl`

### 环境 B：新 LLM/MLLM 环境

职责：

- `Q-Former` 对齐训练
- `LoRA / QLoRA` 联训
- 解释模型推理
- 服务器端自动评测

不负责：

- 老 `BEVFusion` 训练或推理

## 4. 老环境执行顺序

### 4.1 导出感知资产

```bash
python tools/export_bev_features.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_mini.yaml \
  ./pretrained/bevfusion-det.pth \
  --output-dir outputs/bev_vlm \
  --split val
```

### 4.2 构造 QA 数据

```bash
python tools/create_bev_vlm_data.py \
  outputs/bev_vlm/manifest_val.jsonl \
  --output outputs/bev_vlm/bev_vlm_sharegpt_v2.json \
  --flat-output outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl \
  --manifest-output outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl \
  --crop-dir outputs/bev_vlm/anchor_crops_v2
```

### 4.3 导出前做资产校验

```bash
python tools/validate_bev_vlm_outputs.py --root outputs/bev_vlm
```

如果校验通过，说明：

- manifest 路径完整
- flat 样本路径完整
- crop / render / pt / pred / edl 都存在

## 5. 迁移到服务器的最小文件集

推荐迁移整个：

- `outputs/bev_vlm/`

如果只搬最关键资产，至少包括：

- `outputs/bev_vlm/val/`
- `outputs/bev_vlm/anchor_crops_v2/`
- `outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl`
- `outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl`
- `outputs/bev_vlm/bev_vlm_sharegpt_v2.json`

注意：

- `flat.jsonl` 和 `manifest.jsonl` 里现在写的是绝对路径
- 如果服务器路径和本地不同，迁移后需要：
  - 保持相同目录结构
  - 或者重写这些路径

最简单的方法是：

- 在服务器上仍然使用 `/root/bevfusion/...` 这个相同目录前缀

## 6. 新环境执行顺序

服务器阶段直接复用这些资产：

- [outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_flat.jsonl)
- [outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl](/root/bevfusion/outputs/bev_vlm/bev_vlm_sharegpt_v2_manifest.jsonl)

然后按 [SERVER_STAGE.md](/root/bevfusion/SERVER_STAGE.md) 走：

1. Stage 2：
   - 训练 `Q-Former connector`
2. Stage 3：
   - 训练 `Q-Former + LoRA/QLoRA`

## 7. 最终推理建议

最终推理也建议保持两阶段：

1. 老环境：
   - 输入原始传感器数据
   - 导出 `BEV / pred / EDL / crop / render`
2. 新环境：
   - 读取这些中间资产
   - 生成 `scene / miss_summary / trust / attribution_object`

可以后续封装成一个总控脚本，但内部仍然应该是：

- 环境 A 调用一次
- 环境 B 调用一次

而不是把所有依赖塞进一个 Python 环境。

## 8. 如果服务器必须跑 BEVFusion

这件事应该被拆成一条独立迁移任务，不要和大模型训练任务混做一件事。

建议单独立项：

- 目标：老版 `BEVFusion` 在 Blackwell GPU 上适配运行
- 范围：
  - 新 `PyTorch`
  - 新 `CUDA`
  - 旧 `mmcv / mmdet3d / spconv`
  - 自定义算子兼容
- 风险很高，不应阻塞当前大模型训练主线

当前主线更推荐：

- 本地或老服务器负责感知导出
- 新服务器负责解释器训练

## 9. 当前建议

当前最合理的下一步是：

1. 在老环境再次生成并校验 `outputs/bev_vlm`
2. 把这批标准资产迁到服务器
3. 在服务器新环境中按 [SERVER_STAGE.md](/root/bevfusion/SERVER_STAGE.md) 启动：
   - `Stage 2 Q-Former`
   - `Stage 3 LoRA/QLoRA`

不要把“Blackwell 适配老 BEVFusion 环境”作为当前训练主线的前置条件。

