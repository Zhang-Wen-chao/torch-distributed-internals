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
stage 是 GraphModule，TP plan 不兼容），已标注在 README。

## 组合的关键

- **进程组划分**：mesh (tp, pp) → TP 组沿 dim1、PP 组沿 dim0
  （或自定义），每维一个进程组（chapter 01 的地基）。
- **PP 的 stage 内再做 TP**：每个 stage 内部用 parallelize_module（TP）；
  相邻 stage 之间 p2p 传激活（PP）。
- **DP 套最外层**：DDP 复制整条 TP+PP 管线（需要 8 卡以上才完整 3D；
  4 卡上 TP×PP 已是 4，DP=1）。

## 源码地图

## 4 卡上的可行组合

| 组合 | 卡数 | 说明 |
| --- | --- | --- |
| TP=2 × PP=2 | 4 | 每 stage 内部 2 卡 TP；stage 间 p2p |
| TP=2 + DP=2 | 4 | DDP 包 TP（DDP 官方支持 device_mesh） |
| TP=2 × PP=1 + DP=2 | 4 | DDP 复制 TP 模型 |
| 完整 3D（TP×PP×DP） | ≥8 | 本仓库环境不可行，标注 |

## demo_3d 设计（TP=2 × PP=2）

```
模型（6 个 Linear）：切 2 个 stage（每 stage 3 个 Linear）
每个 stage 内部：parallelize_module（TP=2）：第一个 Linear Colwise、
最后一个 Rowwise、中间层的接续（全量/分片注意布局）
PP 通信：stage 间 p2p（官方 pipelining 的 Schedule1F1B 支持
TP 组：需要 stage 的 forward/backward 由 TP 组内两个 rank 并行执行
```

实现（官方组件）：
1. `init_device_mesh("cuda", (2, 2), mesh_dim_names=("pp", "tp"))`
2. 模型按层切 2 个 stage：stage0 = [Linear0, ReLU, Linear1, ReLU]，stage1 = [Linear2]
   （用 `pipe_split()` 或 `split_spec` 标注）
3. **每个 stage 内部先 TP 化**（`parallelize_module(stage_mod, mesh["tp"], plan)`）
4. `pipeline()` + `Schedule1F1B`，PP 组 = mesh["pp"]
5. 训练与单设备 loss 对比

注意：官方 pipelining + TP 组合需要 stage 用 TP mesh 建（官方有
`pipeline(..., group=mesh["pp"])` + stage 内 TP 的示例模式）。

## 验证标准

- PP=2 训练 loss == 单设备 loss（同 seed 同数据同 micro-batch 累积）。
- 记录 bubble 与吞吐（可选）。
