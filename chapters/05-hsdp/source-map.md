# 05-hsdp 源码地图

走读基线：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`。HSDP 不是独立文件，
是 FSDP 的 HYBRID_SHARD 策略，主要差异在 `_runtime_utils.py`。

## 关键行号索引（已在 chapter 04 读过）

| 位置 | 内容 |
| --- | --- |
| `fsdp.py:117` | `class FullyShardedDataParallel`：`device_mesh` 参数解析 |
| `_runtime_utils.py:831` | `_reduce_grad`：HYBRID_SHARD 分支（`:863-878`） |
| `_runtime_utils.py:863-878` | **HSDP 梯度路径**：组内 reduce-scatter 后，`dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)`（`:872`），再 post-divide |

## HSDP 梯度路径（reducer 视角）

```
post_backward（flat 全量梯度）:
  ├─ reduce_scatter（分片组内）→ 每个 rank 拿到自己 shard 的梯度和
  ├─ all_reduce（复制组间）→ 所有复制副本的分片梯度求和
  ├─ 除以 world_size（pre/post divide 因子）
  └─ 累积进 shard 梯度

forward（参数还原）:
  all_gather（分片组内）→ 本复制组内每个 rank 还原全量
  复制组之间参数相同（不需要跨组 gather）
```

对比 FSDP：FSDP 只有"一个组"（全体 rank 分片），HSDP 多一层复制组。
复制组的含义：组内各 rank 持有**相同的完整参数**（前向 gather 后），
梯度 reduce-scatter 后再跨复制组 all-reduce 分片梯度。
