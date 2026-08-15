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
├── notes/
│   └── 01-zro-internals.md  # 分片 / 本地优化器 / 参数同步
└── demos/
    └── demo_zero1.py        # 官方 ZeRO-1 vs 全量 AdamW 数值等价
```

## L20 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_zero1 | 2×NCCL | PASS（与全量 AdamW 逐元素一致；本地优化器各持 2/4 参数） |
| demo_zero1 | 4×NCCL | PASS（各持 1/1/1/3 参数） |
| demo_zero1 | 2×Gloo cpu | PASS |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，2026-08-15。

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

## 源码地图

走读基线：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`。
文件：`torch/distributed/optim/zero_redundancy_optimizer.py`（1679 行）。

| 位置 | 内容 |
| --- | --- |
| `:289` | `class ZeroRedundancyOptimizer(Optimizer, Joinable)` |
| `:376` | `__init__`：校验 → `Optimizer.__init__` → 进程组 → 本地优化器构造 → 分桶 |
| `:650` | `_partition_parameters`：贪心分片（参数按 numel 从大到小，给累计最小的 rank，`:688-696`） |
| `:722` | `_param_to_rank`：参数 → 归属 rank 映射（缓存） |
| `:758` | `_broadcast_params_from_rank`：单个 rank 向全体广播它的分片 |
| `:810` | `_sync_params`：**所有 rank 轮流广播自己的分片**（step 后恢复全量一致） |
| `:1112` | `step()`：`_local_step`（本地优化器更新分片）+ `_sync_params()` |
| `:1211` | `state_dict`：本 rank 已知的全局状态（需 `consolidate_state_dict`） |
| `:1173` | `load_state_dict`：只载入本 rank 分片的 state |
| `:895` | DDP bucket 对齐的子集分配（`overlap_with_ddp=True` 用） |

## 与 mini-deepspeed ZeRO-1 对照

| 维度 | 官方 ZeroRedundancyOptimizer | mini-deepspeed Stage 1 |
| --- | --- | --- |
| 分片对象 | 优化器状态（Adam moments） | 优化器状态 |
| 分片算法 | 贪心按大小（`:688-696`） | 参数展平后等长切片 |
| 参数同步 | 每 rank 广播自己的分片（`broadcast`，`:810-823`） | 更新后 all-gather |
| 梯度处理 | 不处理（假设已平均） | all-reduce 均值 |
| 与 DDP 集成 | `overlap_with_ddp` / bucket 对齐 | 无 |
| 保存/恢复 | `state_dict` + `consolidate_state_dict` | 显式拒绝（无格式） |

结论：两者是同一 ZeRO-1 语义（分片优化器状态、更新后恢复全量参数），但官方
版粒度是"参数级 + 广播"，mini-deepspeed 是"扁平向量 + all-gather"；官方版还
多了与 DDP 的深度集成（bucket 对齐），这是 mini-deepspeed 明确不做的。
