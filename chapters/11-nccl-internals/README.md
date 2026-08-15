# 11-nccl-internals — NCCL 通信算法（ring/tree/pipe）实测

目标：实测 NCCL 的算法选择对 all-reduce 性能的影响——不同消息大小、
不同 `NCCL_ALGO`（Ring/Tree/PatRing）下的耗时，理解"小消息 latency 主导、
大消息 bandwidth 主导"以及算法的适用区间。

## 本章要回答的问题

1. NCCL 的算法有哪些？各自的适用区间？（ring：带宽最优；tree：节点数多时
   延迟低）
2. `NCCL_ALGO` / `NCCL_PROTO` / `NCCL_DEBUG=INFO` 怎么用？
3. 4 卡 PCIe（无 NVLink）上不同算法的实测差异？

## 目录

```text
chapters/11-nccl-internals/
├── README.md      # 本章入口（本文件）
└── demos/
    └── bench_allreduce.py  # all-reduce benchmark（size × algo）
```

## L20 验证记录

| 消息大小 | Ring | Tree | 结论 |
| --- | --- | --- | --- |
| 1 KB | 0.16 ms | **0.09 ms** | 小消息 Tree 延迟低（树深度低） |
| 1 MB | 2.42 ms | **1.13 ms** | 中等消息 Tree 胜 |
| 16 MB | **10.1 ms** (1.66 GB/s) | 15.6 ms (1.08 GB/s) | 大消息 Ring 带宽优势 |
| 256 MB | **163 ms** (1.64 GB/s) | 257 ms (1.04 GB/s) | Ring 明确胜出 |

环境：4×L20 PCIe（无 NVLink），NCCL 2.29.2，`NCCL_SHM_DISABLE=1`
（容器 shm 仅 1GB，NCCL 走 socket）。

**结论（绑定本次配置）**：
1. 算法切换点：~1MB 以下 Tree 延迟低，以上 Ring 带宽优——与 NCCL 启发式
   设计一致（小消息延迟主导、大消息带宽主导）。
2. **实测带宽上限 ~1.6 GB/s**：远低于 PCIe 理论（32 GB/s 双向）——因为
   `NCCL_SHM_DISABLE=1` 强制走 TCP socket（容器 shm 限制，见 handoff）。
   同卡 NVLink 集群的 ring 带宽可达几十 GB/s。
3. PatRing：NCCL 2.29.2 中 `NCCL_ALGO=PatRing` 报 invalid usage——新版
   ring 已默认并入 pattern 感知实现。
4. 对训练的含义：本机多卡 all-reduce 瓶颈在通信链路（socket），印证
   handoff 的"TP/PP 无性能收益"结论。

## 背景（source-map 摘要）

- **Ring**：每个 rank 只与邻居通信，带宽利用率高，延迟随规模线性增长
  （O(N) 步）；
- **Tree**：log 深度（O(logN) 步），大集群延迟低，但带宽利用率不如 ring；
- **PatRing**（2.19+）：pattern 感知 ring，多消息并行时优化；
- NCCL 按消息大小启发式选算法：小消息（<~512B）用直接/共享内存；
  Tree 在节点数多时胜出；单节点内 ring 通常最优。
- `NCCL_ALGO=Ring,Tree` 限定候选；`NCCL_DEBUG=INFO` 打印实际选用的算法。

## 源码地图

## NCCL 算法（官方文档要点，走读 `docs.nvidia.com/deeplearning/nccl`）

| 算法 | 机制 | 适用 |
| --- | --- | --- |
| Ring | 环状流水，每 rank 与邻居收发，数据分 W 块 | 带宽受限的大消息；单节点 |
| Tree | 二叉/多叉树，根收集再广播 | 延迟受限；节点数多（跨节点） |
| PatRing | pattern 感知 ring（nccl 2.19+），聚合多个消息 | 多并行消息（如分布式训练梯度） |
| CollNet/Oneshot | 集合通信加速网（SHARP）/单次通信 | InfiniBand SHARP 环境 |

- **启发式**：NCCL 按消息字节数 + 拓扑选择算法（`ncclTopoTuneModel`）。
  小消息走 Direct/LL（低延迟协议），大消息走 SIMPLE（高带宽协议）。
- **协议（PROTO）**：LL（低延迟，half 带宽）、LL128（SM 优化）、SIMPLE
  （全带宽）。`NCCL_PROTO` 可覆盖。
- 消息大小 × 算法 × 协议构成"调参空间"，实测是理解性能边界的最快方式。

## 本仓库环境

- 4×L20 PCIe 4.0 x16（~32GB/s 双向），无 NVLink；NCCL 2.29.2。
- 预期：单节点内 ring 最优；tree 的 log 深度优势在 4 卡上不明显；
  小消息（<1MB）latency 主导，算法差异小；大消息带宽主导，ring 胜出。
