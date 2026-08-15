# 06-tp — TP：把权重切开

目标：读透官方 `parallelize_module` + ColwiseParallel / RowwiseParallel——
它是建立在 **DTensor** 之上的 TP 实现（权重用 `distribute_tensor` 打上
Shard 布局，计算仍走原生算子，DTensor 负责通信）。并对照手工实现的
Column/Row ParallelLinear。

## TL;DR

TP = 把一层的大权重沿行/列切开分给多卡（Colwise / Rowwise）。官方实现就是
"给权重贴上 DTensor 标签"，通信自动发生；前向输出与单设备完全一致。

## 本章要回答的问题

1. `parallelize_module(module, tp_mesh, {"w1": ColwiseParallel()})` 做了什么？
2. Colwise / Rowwise 的权重布局与输入输出布局（Replicate/Shard）？
3. 官方 TP 与手工实现（Column/Row + all-reduce）的对应关系？
4. 官方 TP 的前向结果 == 单设备模型前向结果（单设备语义保持）？

## 验证记录

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

## 两种实现层次

同一套 TP 数学（沿列/行切权重 + 归约合并）有两种写法：

| 维度 | 官方 TP（DTensor 层） | 手工 TP（原语层） |
| --- | --- | --- |
| 权重切分 | DTensor Shard(0)/Shard(1)（`distribute_tensor`） | 手工沿列/行切片 |
| 输入布局 | DTensor Replicate / redistribute | 每 rank 持全量输入 |
| 输出合并 | DTensor 算子自动通信 | 显式 dist.all_reduce |
| 前向语义 | 与单设备一致（官方保证） | 与官方一致（本仓库验证） |

**本质**：官方 TP = "权重打上 Shard 布局的普通模块"，通信由 DTensor 自动
补齐；手工版 = 用 chapter 00 的原语显式写同一套数学。想从底层理解 TP，
先写手工版；想在生产中用，用官方版。
