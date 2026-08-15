# 02 — FSDP2（fully_shard）与 DTensor

> 走读版本：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`
> 走读日期：2026-08-15
> 文件：`torch/distributed/fsdp/_fully_shard/_fully_shard.py`（746 行）、
> `_fsdp_param.py`（966 行）、`torch/distributed/tensor/_api.py`（1376 行）

## FSDP1 vs FSDP2：一句话差异

**FSDP1 用自定义的 FlatParameter + C++/Python 运行时管理通信；FSDP2 把
"分片"本身表达成 DTensor（`Shard(0)`），通信由 DTensor 算子系统自动完成。**

```
FSDP1: FlatParameter（自定义叶子）→ all-gather/reduce-scatter 由 _runtime_utils 手动编排
FSDP2: param 是 DTensor(Shard(0)) → forward pre-hook 调 all_gather（unshard）
       变为普通 Tensor；反向 reduce-scatter 回 DTensor
```

## FSDP2 的 fully_shard（`:87`）

```
fully_shard(module, mesh, reshard_after_forward, ...)
  ├─ 模块前向 pre-hook：all-gather 分片参数（unshard）→ 普通 Tensor
  ├─ 模块前向 hook：释放全量（reshard_after_forward=True 时）
  ├─ backward hook：reduce-scatter 梯度 → DTensor shard
  └─ 分片粒度 = 每次 fully_shard 调用一个通信组（自底向上调用，
       下层已分片的参数归入下层组，`:121-131`）

mesh 语义（`:136-143`）：
  1D mesh → FSDP：placements = (Shard(0),)
  2D mesh → HSDP：placements = (Replicate(), Shard(0))
```

`_fsdp_param.py` 的 `_init_sharded_param`（`:260`）：把参数构造为
`DTensorSpec(Shard(0))` 的 DTensor（`:333`），`is_dtensor = isinstance(param,
DTensor)`（`:289`），支持参数原本已是 DTensor（TP 叠加，`:291` 的 `_tp_spec`）
——**TP+FSDP 在 FSDP2 里是天然组合**：TP 参数（Shard(1)）再套 FSDP 的
Shard(0)，二维分片。

## DTensor 基础（tensor/_api.py）

```
distribute_tensor(tensor, mesh, [Shard(0)])           :692
  ├─ 只接受叶子（`:757-760`）
  ├─ 默认 src_data_rank=0：单设备语义（scatter/broadcast）  :697/:723-728
  └─ 返回 DTensor（含 _local_tensor + DTensorSpec）

redistribute(placements)：DTensor 方法
  Shard(0) → Replicate：沿分片维 all-gather（通信发生点）
  Replicate → Shard(0)：scatter
  Shard(0) → Shard(1)：all-to-all（转置式）
```

**关键认知**：DTensor 的算子（mm/add/...）通过 `__torch_dispatch__` 知道每个
张量的布局，自动在需要时插入通信（redistribute）。TP 里 Colwise 的
"输出 Shard(-1)" 到 Rowwise 的 "输入 Shard(-1)" 无需显式通信，正是这个
机制（chapter 06 的 `_prepare_input_fn` 只是声明布局）。

## 与本章前文的关系

- chapter 01 的 DeviceMesh 是 DTensor 的地基（mesh 的每维一个进程组）；
- chapter 06 的官方 TP 就是"把参数 distribute_tensor 成 DTensor 的普通
  模块"；
- FSDP2 = 同一条路径用于数据并行：参数分片成 Shard(0) DTensor，
  `to_local()` 得到本地分片，`redistribute(Replicate)` 即全量还原。

## 下一步

演示：DTensor 布局/redistribute 语义 + fully_shard 与 FSDP1/手写版数值等价。
