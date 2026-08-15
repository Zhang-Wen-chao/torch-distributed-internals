# 02-ddp — DDP：数据并行的官方实现

目标：读透 DDP——它是用得最多的并行 wrapper，也是理解 FSDP/HSDP 的跳板。
DDP 的核心不在 Python 而在于 C++ 的 `Reducer`：参数分桶、autograd 梯度 hook、
桶就绪即 all-reduce、与 optimizer.step() 的契约。

## TL;DR

DDP 的全部秘密 = **梯度分桶 + 桶满即 all-reduce**（通信与反向计算重叠）。
仓库里的 `ManualDDP` 用 150 行手写了这个机制，与官方 DDP 训练参数逐元素一致。

## 本章要回答的问题

1. `DistributedDataParallel(module)` 初始化时做了什么？（参数排序、跨 rank
   校验、Reducer 创建、bucket 组装）
2. 反向时"每个参数梯度就绪 → 桶满 → all-reduce"的完整机制？
3. bucket 是怎么分的？`bucket_cap_mb` 怎么影响通信粒度？
4. `find_unused_parameters` / `static_graph` / `gradient_as_bucket_view` 各自
   解决了什么问题？
5. 手写一个最小 DDP（autograd hook + 分桶 + all-reduce）能否与官方数值一致？

## 验证记录

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`。

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_ddp_mechanism | 2×NCCL | PASS（3 步训练跨 rank 参数一致） |
| demo_ddp_mechanism | 4×NCCL | PASS |
| demo_ddp_mechanism | 2×Gloo cpu | PASS |
| compare_ddp_manual_vs_official | 2×NCCL | PASS（3 步参数逐元素一致，rtol=1e-5） |
| compare_ddp_manual_vs_official | 4×NCCL | PASS |
| compare_ddp_manual_vs_official | 2×Gloo cpu | PASS |

实测发现（详见 notes/02 末尾）：本版本 `register_post_accumulate_grad_hook`
的 grad 参数数值不可靠（~12.5x），需从 `p.grad` 取梯度；桶 pending 计数必须
每步重置。

## 使用手册（本层关键坑）

- DDP 要求所有 rank 的模型**参数顺序一致**、初始值一致（初始化时从 rank 0
  broadcast，`_sync_params_and_buffers`）；否则静默训练错模型。
- 反向必须从 DDP 包装后的模块输出开始（forward 里做了 `prepare_for_backward`）。
- `find_unused_parameters=True` 有性能代价（每步扫描计算图 + local_used_map
  all-reduce）；模型结构稳定时用 `static_graph=True` 或干脆不用该参数。
- `gradient_as_bucket_view=True` 让 `param.grad` 直接指向 bucket 内存，省一次
  拷贝，但要求"每个参数只注册一个 hook"（`:1685-1692` 的检查）。
- DDP 不做梯度累积的除法修正以外的任何魔法：all-reduce 求和后除以 world_size
  （`set_divide_factor`，`reducer.cpp`）。梯度裁剪、AMP 都在 DDP 外部。

## 源码地图

走读基线：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806` +
`pytorch/pytorch@a36e1d39eb`。

## Python 侧：`torch/nn/parallel/distributed.py`（2434 行）

| 位置 | 内容 |
| --- | --- |
| `:328` | `class DistributedDataParallel(Module, Joinable)` |
| `:653` | `__init__`：进程组选择（含 device_mesh 1D 支持）、参数收集、`_verify_model_params_across_processes`、bucket 大小、Reducer 创建 |
| `:1660` | `forward`：`_pre_forward`（触发 `_prepare_for_backward`）→ 模型 forward → `_post_forward` |
| `:1740` | `_match_all_reduce_for_bwd_pass`（join 模式用） |
| `:2034` | `_register_builtin_comm_hook`：AllReduce / FP16 压缩 hook |
| `:2358` | `_set_static_graph` |
| `:1182-1250` | bucket 大小推断（`bucket_cap_mb` → `bucket_bytes_cap`） |

## C++ 侧：`torch/csrc/distributed/c10d/reducer.cpp`（2501 行）+ `reducer.hpp`

| 位置 | 内容 |
| --- | --- |
| `reducer.hpp:30` | `kDefaultBucketBytesCap = 25MB`（默认桶上限） |
| `:90` | `Reducer::Reducer`：校验 + `initialize_buckets` |
| `:1064` | `initialize_buckets`：按 `bucket_indices` 组装桶（含 `bucket_views_in/out`） |
| `:649` | `autograd_hook`：参数梯度就绪入口 |
| `:876` | `mark_variable_ready`：累计就绪、`--bucket.pending == 0` 时 `mark_bucket_ready` |
| `:955` | `all_reduce_bucket`：构造 `GradBucket` 并触发 comm hook |
| `:939` | `run_comm_hook`：默认 `run_allreduce_hook`（`_AllReduceBySumCommHook`） |
| `:1529` | `prepare_for_backward`：重置计数、可选未使用参数扫描 |
| `:1611` | `finalize_bucket_dense`：写回 `param.grad`（bucket_view / copy） |
| `:1705` | `finalize_backward`：等所有桶的 `future_work`、统一异常处理 |

## 推荐阅读顺序

1. Python `__init__`（`:653`）→ 弄清参数顺序、broadcast、bucket 大小
2. reducer.cpp `initialize_buckets`（`:1064`）→ bucket 怎么拼
3. `autograd_hook → mark_variable_ready → mark_bucket_ready → all_reduce_bucket`
   （`:649/:876/:955`）→ 梯度就绪驱动的通信
4. `finalize_bucket_dense`（`:1611`）→ 归约结果如何回到 `param.grad`
5. Python `forward`（`:1660`）→ 与反向的衔接
