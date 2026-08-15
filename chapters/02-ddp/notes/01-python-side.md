# 01 — DDP Python 侧：初始化与 forward 编排

> 走读版本：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`
> 走读日期：2026-08-14
> 文件：`torch/nn/parallel/distributed.py`

## `__init__`（`:653`）做了什么

```
DistributedDataParallel(module, ...)
  ├─ 进程组：process_group 或 device_mesh（仅 1D）→ get_group(0)   :688-703
  │     root mesh 切片出的子 mesh → _pre_dp_module_transform（TP 融合）:705-717
  ├─ 收集参数（排除 _ddp_params_and_buffers_to_ignore）            :729-733
  ├─ 无 requires_grad 参数 → 报错（除非 delay_all_reduce）         :734-742
  ├─ 多设备模块检查（跨 GPU 参数 → is_multi_device_module）        :750-752
  ├─ _verify_model_params_across_processes()  [关键]               :~
  │     ├─ 每 rank 把所有参数展平拼接成一个 tensor
  │     ├─ all_gather 拼接后逐 rank 比对
  │     └─ 不匹配 → "Check that the optimizer's ... 或模型初始化不同"报错
  │        （对比失败不一定是灾难：正是 DDP 能"跨 rank 参数必须一致"
  │          的保证机制）
  ├─ bucket 大小：bucket_cap_mb → bucket_bytes_cap                  :831-834
  ├─ 构造 Reducer（torch._C._distributed_c10d.Reducer）
  │     bucket_indices 由参数大小从大到小贪心装桶（对齐 256B）
  └─ _sync_params_and_buffers：rank 0 → broadcast 初始参数
        （保证所有 rank 从同一权重开始，见笔记 02 的"不变量"）
```

要点：

- **`_verify_model_params_across_processes` 是全量对比**，不是抽样。对比的是
  展平后的参数向量，任何 rank 顺序/初值不一致都会立刻失败——这是 DDP
  "SPMD 契约"的第一道防线。
- `bucket_cap_mb=None` 时用 C++ 默认 `kDefaultBucketBytesCap = 25MB`
  （reducer.hpp:30）。
- 参数按**尺寸从大到小**装桶、桶内字节对齐（256B），保证每个桶内参数数量
  少而通信粒度粗。

## `forward`（`:1660`）与反向的衔接

```
forward(*inputs)
  ├─ _pre_forward → 内部触发 reducer.prepare_for_backward()       :1662
  ├─ module.forward(*inputs)（或 _run_ddp_forward：延迟同步时不跑） :1663-1667
  └─ _post_forward：记录 used params / 广播 buffer                :1668
```

关键：**`prepare_for_backward` 发生在 forward 里**（`torch.distributed.autograd`
文档所说的 "DDP 在 forward 时注册好 hook 等待 backward"）。反向一旦开始，
`autograd_hook` 按梯度就绪顺序被调用。

## 常用参数对应的内部开关

| 参数 | 内部影响 |
| --- | --- |
| `find_unused_parameters=True` | `dynamic_graph_find_unused()`：每步 `search_unused_parameters` 扫计算图 + `all_reduce_local_used_map` |
| `static_graph=True` | 只扫一次；之后 `skip_all_reduce_unused_params` 可跳过 unused 桶 |
| `gradient_as_bucket_view=True` | 桶内存直接作为 `param.grad` 的视图，省一次 `copy_bucket_to_grad` |
| `bucket_cap_mb` | 桶粒度；越小通信越细（overlap 好）但次数越多 |

## 下一步

C++ Reducer：分桶结构、梯度 hook 驱动、桶就绪 → all-reduce（笔记 02）。
