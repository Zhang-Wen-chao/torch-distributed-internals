# 03-zeroredundant — 官方 ZeroRedundancyOptimizer（ZeRO-1）

目标：读透 `torch.distributed.optim.ZeroRedundancyOptimizer`——PyTorch 自带的
ZeRO-1 实现（只分片优化器状态），与 `mini-deepspeed` 的 ZeRO-1 交叉对照。

## 本章要回答的问题

1. 官方 ZeRO-1 的参数分片算法？（贪心：最大参数优先给累计最少的 rank）
2. 每步 `step()` 的通信是什么？（本地优化器更新分片 → 广播参数全量）
3. 它与 mini-deepspeed 的 ZeRO-1 在语义上有什么异同？
4. `overlap_with_ddp` / `parameters_as_bucket_view` 分别是什么？

## 目录

```text
chapters/03-zeroredundant/
├── README.md       # 本章入口（本文件）
├── source-map.md   # 源码地图
├── notes/
│   └── 01-zro-internals.md  # 分片 / 本地优化器 / 参数同步
└── demos/
    └── demo_zero1.py        # 官方 ZeRO-1 vs 全量 AdamW 数值等价
```

## L20 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_zero1 | 2×NCCL | 待验证 |

## 使用手册（本层关键坑）

- `step()` 内部会做全量参数同步（每个 rank 广播自己的分片），**替代**了
  DDP 的梯度 all-reduce 吗？不——它假设梯度已经是**全量平均后的**（通常配
  DDP 用），它只负责"分片优化器状态 + 更新后同步参数"。
- 每次 `step()` 后所有 rank 参数恢复一致（`_sync_params`），这是它和
  mini-deepspeed ZeRO-1 共同的不变量。
- 不配 DDP 单独用时，需要自己保证梯度已跨 rank 平均（演示脚本里显式
  all-reduce 梯度）。
- `state_dict()` 只返回**本 rank 已知的**全局状态（经 `consolidate_state_dict`
  后才是完整状态），直接保存会丢分片信息。
