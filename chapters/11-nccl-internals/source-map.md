# 11-nccl-internals 背景

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
