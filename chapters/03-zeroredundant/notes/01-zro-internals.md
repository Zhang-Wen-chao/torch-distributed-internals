# 01 — ZeroRedundancyOptimizer 内部

> 走读版本：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`
> 走读日期：2026-08-15
> 文件：`torch/distributed/optim/zero_redundancy_optimizer.py`

## 一句话总结

**ZeRO-1：每个 rank 只持有并更新一部分参数的优化器状态；step() 后把各自
更新的参数分片广播给所有人，恢复全量一致性。** 梯度仍需跨 rank 平均
（通常由 DDP 完成），它不碰梯度。

## 初始化（`:376`）

```
ZeroRedundancyOptimizer(params, optimizer_class=AdamW, ...)
  ├─ 校验参数类型/形状一致（`:387-388`）
  ├─ Optimizer.__init__（保留完整 param_groups，但本地优化器是另一份）
  ├─ process_group / world_size / rank / global_rank               :417-426
  ├─ _init_local_optimizer()：为「本 rank 分到的参数」建一个真正的
  │    optimizer_class 实例（`:435`，self.optim）
  └─ _build_param_buckets()（parameters_as_bucket_view=True 时把分片
       参数展平成连续桶，:449）
```

## 分片算法（`:650-702`）

```
_partition_parameters():
  sizes = [0] * world_size
  for param_group in param_groups:
    params_sorted = sorted(params, key=numel, reverse=True)   # 大到小
    for param in params_sorted:
      rank = argmin(sizes)        # 给当前累计字节最少的 rank
      assign(param, rank); sizes[rank] += param.numel()
```

- 贪心"最大优先 + 最小负载"装箱，目标是各 rank 分到的**字节数**尽量均匀。
- 注意：参数**不跨 rank 拆分**（粒度是参数级，不是扁平分片）——这是它和
  mini-deepspeed（扁平等长切片）的最大结构差异。
- 分片结果缓存在 `_partition_parameters_cache`，所有 rank 用相同算法得到
  相同结果，无需通信。

## step() 的完整路径（`:1112`）

```
step():
  ├─ loss = _local_step(closure, **kwargs)
  │     → self.optim.step()：本地优化器只更新本 rank 分到的参数
  │        （Adam moments 只存在 owner rank 上 → 内存省 (N-1)/N）
  └─ _sync_params():                                              :810
       handles = []
       for rank in range(world_size):
         handles += _broadcast_params_from_rank(rank)   # 每 rank 广播分片
       for h in handles: h.wait()
```

`_broadcast_params_from_rank`（`:758`）：src=rank 把自己分到的参数逐个
`broadcast()` 给组内所有人。所以每步通信量 = **全量参数 × (world_size-1)
个广播**（等价一次全量 all-gather 的通信量），与 DDP 的梯度 all-reduce
量级相同。

## 关键不变量

1. 初始化后：所有 rank 参数一致（ZRO 的 `_init_optimizer_state` 里对每个分片
   做了一次 broadcast）。
2. 每次 `step()` 后：所有 rank 参数一致（`_sync_params`）。
3. 每个参数只由 owner rank 更新（其 Adam moments 只有 owner 持有）。

## 两个可选模式

- `parameters_as_bucket_view=True`：分片参数展平成连续桶（`:449`），减少
  通信调用次数。
- `overlap_with_ddp=True`：本地优化器延迟初始化，把分片与 DDP 的梯度 bucket
  对齐（`:432-443`，`:895` 的 `_assign_bucket_subset_to_rank`），让参数更新
  发生在 DDP 桶重建之后——省掉一次"广播初始化"的同步。

## 与 mini-deepspeed 的语义对照结论

官方 ZRO 与 mini-deepspeed ZeRO-1 是**同一 ZeRO-1 语义**的两个实现：
- 状态归属：参数级分片（官方） vs 扁平向量切片（mini）；
- 参数同步：广播（官方） vs all-gather（mini）；
- 梯度：不处理（官方，假设 DDP 已平均） vs 内部 all-reduce（mini）；
- 附加：官方有 DDP bucket 对齐和 consolidate state_dict，mini 显式不做。
