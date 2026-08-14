# 00-primitives — c10d 通信原语

目标：读透 `torch.distributed` 最底层——进程组（ProcessGroup）、collective 原语、
NCCL/Gloo 差异、异步与多 stream 重叠。这一章是后续所有章节的地基：DDP 的梯度
all-reduce、FSDP 的 all-gather/reduce-scatter、TP 的切分通信，全部建立在这些原语上。

## 本章要回答的问题

1. `init_process_group` 发生了什么？（rendezvous → backend 选择 → 进程组对象）
2. 一个 `dist.all_reduce(t)` 调用从 Python 到 C++（c10d）到 NCCL 的完整路径？
3. collective 的同步/异步语义是什么？`async_op=True` 返回的 work 对象怎么用？
4. NCCL 和 Gloo 在什么时候用、行为差异在哪？（设备、原生 op、P2P 语义）
5. CUDA 多 stream：collective 与计算如何重叠？`torch.cuda.stream` 怎么配合？

## 目录

```text
chapters/00-primitives/
├── README.md         # 本章入口（本文件）
├── source-map.md     # 源码地图：官方文件 + 版本 + 职责
├── notes/            # 逐段走读笔记（待写）
└── demos/
    ├── demo_allreduce.py   # collective 语义 + op 类型 + 分组
    └── demo_async_stream.py  # 异步 + 多 stream 重叠（待写）
```

## 演示目标

| 演示 | 证明什么 |
| --- | --- |
| `demo_allreduce.py` | all-reduce 的 SUM/PRODUCT/MIN/MAX 语义、进程组内 rank 子集、结果与全局正确性 |
| `demo_async_stream.py` | `async_op=True` 的非阻塞语义；单独 CUDA stream 上通信与计算重叠 |

## 使用手册（本层关键坑）

- collective 是**集体操作**：组内所有 rank 必须用相同顺序调用相同 op，否则挂死或
  未定义行为；进程组 timeout 只能把死等变成报错。
- `init_process_group` 的 backend 参数：CPU tensor 用 `gloo`，CUDA tensor 用
  `nccl`。`nccl` 后端不原生支持 CPU tensor（会报错或退化）。
- `torch.distributed.barrier()` 是 collective：只用于同步点，不是调试工具。
- 单进程也要 `destroy_process_group()`；多 rank 结束时同样要销毁。
- NCCL 的 buffer 与 stream 语义：collective 在调用时所在的 stream 上排队，
  与计算重叠需要显式 stream 编排（见 `demo_async_stream.py`）。
