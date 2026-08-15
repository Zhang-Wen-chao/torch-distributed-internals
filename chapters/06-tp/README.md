# 06-tp — 官方张量并行（torch.distributed.tensor.parallel）

目标：读透官方 `parallelize_module` + ColwiseParallel / RowwiseParallel——
它是建立在 **DTensor** 之上的 TP 实现（权重用 `distribute_tensor` 打上
Shard 布局，计算仍走原生算子，DTensor 负责通信）。与 mini-megatron 手写的
Column/Row ParallelLinear 对照。

## 本章要回答的问题

1. `parallelize_module(module, tp_mesh, {"w1": ColwiseParallel()})` 做了什么？
2. Colwise / Rowwise 的权重布局与输入输出布局（Replicate/Shard）？
3. 官方 TP 与 mini-megatron 手写实现（Column/Row + all-reduce）的对应关系？
4. 官方 TP 的前向结果 == 单设备模型前向结果（单设备语义保持）？

## 目录

```text
chapters/06-tp/
├── README.md       # 本章入口（本文件）
├── source-map.md   # 源码地图
└── demos/
    └── demo_tp.py  # 官方 TP 语义验证（与单设备模型输出一致）
```

## L20 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_tp | 2×NCCL | PASS（TP 输出与单设备一致，Colwise 权重切分正确） |
| demo_tp | 2×Gloo cpu | PASS |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
2026-08-15。

## 使用手册（本层关键坑）

- `parallelize_module` 只接受 **1D mesh**（多维要先切片 `mesh["tp"]`，
  api.py:30-31）。
- Colwise 输出默认 Shard(-1)，配 Rowwise 输入（Shard(-1)）才能无缝衔接；
  中间夹非 shard-aware 算子时要手动处理布局。
- `use_local_output=True`（默认）时输出是本地张量；需要继续作为 DTensor
  传播布局时设 False。
- 与 DDP 组合：TP 是模型并行，DDP 套在外面做数据并行（`_data_parallel_utils`
  处理 state_dict 转换）。
