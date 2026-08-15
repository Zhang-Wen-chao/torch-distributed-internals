# 06-tp 源码地图

走读基线：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`。
目录：`torch/distributed/tensor/parallel/`。

| 文件 | 职责 |
| --- | --- |
| `api.py`（142 行） | `parallelize_module`：按 plan（style 或 {FQN: style}）递归应用 |
| `style.py`（810 行） | `ColwiseParallel`/`RowwiseParallel`/`SequenceParallel`/`PrepareModuleInput/Output` |
| `_data_parallel_utils.py` | DDP 组合：state_dict 布局转换 |
| `loss.py` | 词表并行交叉熵（配合 embedding 分片） |
| `input_reshard.py` | 输入重切分（SequenceParallel 用） |

## 关键行号索引

### api.py

| 位置 | 内容 |
| --- | --- |
| `:14` | `parallelize_module(module, device_mesh, parallelize_plan)` |
| `:71` | 无 mesh 时取当前 mesh（`_mesh_resources.get_current_mesh`） |
| `:86` | 单个 style → `style._apply(module, mesh)` |
| `:87-128` | dict plan：按 FQN（支持 fnmatch 通配）递归应用 |

### style.py

| 位置 | 内容 |
| --- | --- |
| `:45` | `ColwiseParallel`：weight Shard(0)，输入 Replicate，输出 Shard(-1)（`:90-96`） |
| `:118-128` | `_partition_linear_fn`：`distribute_tensor(param, mesh, [Shard(0)])` |
| `:140-146` | `_prepare_output_fn`：输出 redistribute 到 Shard(-1)，`to_local()` |
| `:179` | `RowwiseParallel`：weight Shard(1)，输入 Shard(-1)，输出 Replicate |
| `:427` | `PrepareModuleInput`：输入 reshape/redistribute |
| `:595` | `PrepareModuleOutput`：输出布局转换 |

## 与 mini-megatron 对照

| 维度 | 官方 TP | mini-megatron |
| --- | --- | --- |
| 权重切分 | DTensor Shard(0)/Shard(1)（`distribute_tensor`） | 手工沿列/行切片 |
| 输入布局 | DTensor Replicate / 手动 redistribute | 每 rank 持全量输入 |
| 输出合并 | Shard(-1) 的 DTensor 算子自动 all-gather | 手写 all-reduce |
| 通信触发 | DTensor 算子内部（redistribute） | 显式 dist.all_reduce |
| 前向语义 | 与单设备一致（官方保证） | 与 Megatron 一致 |

**本质**：官方 TP = "权重打上 Shard 布局的普通模块" + DTensor 自动补齐通信；
mini-megatron = 显式手写同一套数学。两者是同一 TP 语义的两个层次。
