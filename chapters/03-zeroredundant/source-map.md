# 03-zeroredundant 源码地图

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
