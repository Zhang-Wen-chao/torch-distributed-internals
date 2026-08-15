# 03 — DeepSpeed ZeRO-3 vs PyTorch FSDP

> 走读版本：DeepSpeed 0.19.3 `runtime/zero/stage3.py`（3814 行）+
> `parameter_offload.py` vs torch 2.10 `fsdp/_fully_shard/`、`_runtime_utils.py`
> 走读日期：2026-08-15

## 数学同类，工程完全不同

两者都是"参数/梯度/优化器状态分片 + 按需物化 + 用完释放"（全状态分片）。
但读了两边源码后，差异非常具体：

## 差异 1：参数精度结构（最大的差异）

**DeepSpeed**：双副本——**fp16 训练参数 + fp32 分区主副本**：

```
stage3.py:2544 step() 里：
  _prepare_sub_group → fp32 分区拿到梯度（_prepare_fp32_grad_for_sub_group，
       :2370 附近：fp32_partitioned_groups_flat[sub].grad = 展平梯度）
  _optimizer_step(sub_group_id)（:1151）：在 fp32 分区上更新
       → 再把 fp32 copy 回 fp16 参数
  _release_sub_group（:2388）：释放 fp32 梯度、swap out
```

**PyTorch FSDP**：单副本——参数就是 FlatParameter 切片，mixed precision
通过 `MixedPrecisionPolicy`（param_dtype 转换）实现，**没有独立的 fp32
分区主副本**（FSDP2 `_fsdp_param.py` 的 shard 就是唯一存储）。

含义：ZeRO-3 常驻显存 = fp16 分片 + fp32 分区（1/N）；FSDP 常驻 =
参数分片 + 优化器状态分片。ZeRO-3 的 fp32 分区在 optimizer 阶段还要
swap in/out（ZeRO-Infinity 可放到 NVMe）。

## 差异 2：参数物化机制

**DeepSpeed**：模块级 hooks + 参数协调器——

```
parameter_offload.py:279 setup_zero_stage3_hooks():
  每个子模块注册:
    register_forward_pre_hook   → pre_sub_module_forward_function（gather 本模块参数）
    register_forward_hook       → post_sub_module_forward_function（释放）
    register_full_backward_hook → post_sub_module_backward_function（梯度处理+释放）
  coordinator 维护 hierarchy（:2316 _get_param_coordinator().hierarchy）
  step 时 partition_all_parameters()（parameter_offload.py:218）：
    release_and_reset_all —— 保证所有参数回到分片状态
```

**PyTorch FSDP**：FlatParameter + FSDP 单元的 pre/post forward hook
（`_runtime_utils.py:348 _pre_forward` / `:701 _post_backward_hook`），
prefetch 由 `_prefetch_handle` 调度。

物化粒度对比：ZeRO-3 是"每个子模块一个 gather 单元"（粒度更细，
`stage3_max_live_parameters` 可再调）；FSDP 是"每个 FSDP 包装单元"
（用户决定粒度）。

## 差异 3：梯度路径

**DeepSpeed**：ipg_buckets（in-place gradient bucketing）→
`reduce_independent_p_g_buckets_and_remove_grads`（`:1437`）→
`averaged_gradients`（分区梯度）→ fp32 分区。

**FSDP**：flat param 的全量梯度 → `_reduce_grad`（`:831`）的
reduce-scatter（含 padding，`:890`）。

## 差异 4：offload 能力

- DeepSpeed：`optimizer_swapper` / `parameter_offload` / **NVMe swapper**
  （`stage3.py:1205` 的 `nvme_swapper.swap_out_partitioned_params`）——
  ZeRO-Infinity 的完整三级（GPU→CPU→NVMe）。
- FSDP：CPU offload（参数/梯度/优化器状态可 offload 到 CPU），无 NVMe。

## 差异 5：外部参数 / 生态集成

- DeepSpeed 有 **external parameter 注册机制**（`parameter_offload.py:361-381`
  的 `register_external_parameter`，处理 MoE 等"不属于任何模块"的参数）——
  这是配合 DeepSpeed MoE 的产物。
- FSDP 无此概念（用 DTensor/ignored_params 表达）；但 FSDP2 与 torch.compile、
  DCP（分布式 checkpoint）原生集成是它的强项。

## 总结表

| 维度 | DeepSpeed ZeRO-3 | PyTorch FSDP |
| --- | --- | --- |
| 参数存储 | fp16 参数 + **fp32 分区主副本**（双副本） | 单一 FlatParameter 分片 |
| 物化 | 模块级 hooks + coordinator hierarchy | FSDP 单元 hooks + prefetch |
| 优化器 | 按 sub_group 循环，swap in/out | 直接在分片上 |
| offload | GPU→CPU→**NVMe** 三级 | GPU→CPU |
| 外部参数 | 有（MoE 配套） | 无（DTensor 表达） |
| 原生集成 | HF/MoE/推理生态 | torch.compile/DTensor 2D/DCP |

**为什么开源大模型两条 recipe 都有**：ZeRO-3 赢在"极限显存下的自由度"
（双副本 + NVMe + 细粒度物化），FSDP 赢在"原生生态里的简洁"（单副本 +
DTensor + compile）。这就是上一轮说的"同数学、不同工程"的源码级证据。
