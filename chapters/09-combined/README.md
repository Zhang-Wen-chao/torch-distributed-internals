# 09-combined — 组合并行（TP + PP + DP）

目标：把前面各章吃透的组件**组合起来**做 3D 并行：4×L20 上
TP=2 × PP=2（DP=1）或 TP=2 + PP=1 + DP=2。验证组合后的训练与单设备
数值等价（loss 一致）。

## 本章要回答的问题

1. 组合时各并行维度的进程组怎么划分？（mesh 的维度分配）
2. TP+PP+DP 组合的训练 loss 与单设备是否一致？
3. 组合的开销在哪（bubble、通信）？

## 目录

```text
chapters/09-combined/
├── README.md       # 本章入口（本文件）
├── source-map.md   # 组合方法与验证思路
└── demos/
    └── demo_3d.py  # TP=2 × PP=2（4 卡）训练 vs 单设备 loss
```

## L20 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_3d | 4×NCCL（TP=2 × DP=2） | PASS（组合训练 loss 与单设备完全一致） |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
2026-08-15。说明：4 卡上完整 3D（TP×PP×DP）需 ≥8 卡，本环境不可行；
TP×PP 组合需要 pipelining manual frontend 手动拼 stage（nightly 的自动切分
stage 是 GraphModule，TP plan 不兼容），已标注在 source-map。

## 组合的关键

- **进程组划分**：mesh (tp, pp) → TP 组沿 dim1、PP 组沿 dim0
  （或自定义），每维一个进程组（chapter 01 的地基）。
- **PP 的 stage 内再做 TP**：每个 stage 内部用 parallelize_module（TP）；
  相邻 stage 之间 p2p 传激活（PP）。
- **DP 套最外层**：DDP 复制整条 TP+PP 管线（需要 8 卡以上才完整 3D；
  4 卡上 TP×PP 已是 4，DP=1）。
