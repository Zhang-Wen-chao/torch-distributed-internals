# 12-activation-checkpoint — 激活检查点

目标：走读 `torch.utils.checkpoint` 并用 FSDP 组合实测显存收益——
用计算换显存（反向时重算中间激活，不保存）。

## 本章要回答的问题

1. checkpoint 的原理？（前向不保存激活，反向时重算）
2. `use_reentrant=True/False` 两种实现的差异？
3. FSDP + checkpoint 组合能再省多少显存？

## 目录

```text
chapters/12-activation-checkpoint/
├── README.md      # 本章入口（本文件）
└── demos/
    └── demo_checkpoint.py  # FSDP ± checkpoint 峰值显存对比
```

## 走读摘要（checkpoint.py:349）

- `checkpoint(function, *args, use_reentrant=False)`：前向时函数在
  `torch.no_grad` 下执行、**不保存中间激活**（只保存输入）；
- 反向时 `CheckpointFunction.backward` **重跑一次前向**恢复中间激活再算梯度
  （`checkpoint.py:228`）；
- `use_reentrant=True`：前向完全不建图（no_grad 跑完），反向整体重算；
  `False`：保留部分图结构 + **early stop**（`:393`，重算到够用即停）。
- 代价：前向计算量 ×2（前向+反向各一次）；收益：激活显存按 checkpoint
  粒度降低。

## L20 验证记录

| 配置 | 峰值显存 | 说明 |
| --- | --- | --- |
| FSDP 2 卡（无 checkpoint） | 17.00 GB | B=16, S=1024：激活占主导 |
| FSDP 2 卡 + checkpoint | **4.98 GB** | 省显存 **70.7%** |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
2026-08-15。loss 验证：两版本 3 步 loss 完全一致（训练等价）。

**结论（绑定本次配置）**：
1. checkpoint 把激活相关的峰值显存几乎清零（17.0→5.0GB，省 70.7%）——
   "用计算换显存"的核心卖点实测成立。
2. 小 batch 时激活占比低，checkpoint 收益不明显（B=4 时仅 ~3.5%）——
   收益取决于激活在峰值中的占比。
3. 吞吐数据受 benchmark 顺序效应（第二次运行已 warmup）污染，不作结论；
   理论上前向计算量 ×2（前向+反向重算）。

## 源码地图

走读基线：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`。
文件：`torch/utils/checkpoint.py`（1666 行）。

| 位置 | 内容 |
| --- | --- |
| `:228` | `CheckpointFunction`：autograd.Function，反向重算 |
| `:349` | `checkpoint(function, *args, use_reentrant=None)` 入口 |
| `:517` | `checkpoint_sequential`：顺序模块按段包装 |
| `:1484` | `_checkpoint_without_reentrant_generator`：非 reentrant 实现 |
| `:393-403` | reentrant vs 非 reentrant 差异（early stop / 图记录） |

## 与 FSDP 的组合

- FSDP 分片参数 + checkpoint 省激活 → 两者正交叠加（大模型标配）。
- 注意事项：checkpoint 包装**必须包含整个 FSDP 单元的 forward**
  （在 FSDP 模块内部按层包装即可），且 `use_reentrant=False` 与 FSDP2 的
  `torch.compile` 兼容性更好。
