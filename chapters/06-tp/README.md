# 06-tp — 官方张量并行（torch.distributed.tensor.parallel）

目标：读透官方 `parallelize_module` + ColwiseParallel / RowwiseParallel——
它是建立在 **DTensor** 之上的 TP 实现（权重用 `distribute_tensor` 打上
Shard 布局，计算仍走原生算子，DTensor 负责通信）。与 mini-megatron 手写的
Column/Row ParallelLinear 对照。

## 本章要回答的问题

1. `parallelize_module(module, tp_mesh, {"w1": ColwiseParallel()})` 做了什么？
2. Colwise / Rowwise 的权重布局与输入输出布局（Replicate/Shard）？
3. 官方 TP 与 mini-megatron 手写实现（Column/Row + all-reduce）的对应关系？
4. 官方 TP 的前向结果 == 单设备模型前向结果（单设备语义保持）？

## 目录

```text
chapters/06-tp/
├── README.md       # 本章入口（本文件）
└── demos/
    └── demo_tp.py  # 官方 TP 语义验证（与单设备模型输出一致）
```

## L20 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_tp | 2×NCCL | PASS（TP 输出与单设备一致，Colwise 权重切分正确） |
| demo_tp | 2×Gloo cpu | PASS |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
2026-08-15。

## 使用手册（本层关键坑）

- `parallelize_module` 只接受 **1D mesh**（多维要先切片 `mesh["tp"]`，
  api.py:30-31）。
- Colwise 输出默认 Shard(-1)，配 Rowwise 输入（Shard(-1)）才能无缝衔接；
  中间夹非 shard-aware 算子时要手动处理布局。
- `use_local_output=True`（默认）时输出是本地张量；需要继续作为 DTensor
  传播布局时设 False。
- 与 DDP 组合：TP 是模型并行，DDP 套在外面做数据并行（`_data_parallel_utils`
  处理 state_dict 转换）。

## 源码地图

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
