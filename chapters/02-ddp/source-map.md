# 02-ddp 源码地图

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
