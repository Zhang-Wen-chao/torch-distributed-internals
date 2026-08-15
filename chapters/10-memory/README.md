# 10-memory — 显存实测：到底省多少

目标：验证 FSDP 的核心卖点——**省显存**。用足够大的模型（355M 级别，
单卡放不下全量训练）在 4×L20 上实测峰值显存：
单卡（放不下则报 OOM）、DDP、官方 FSDP1、手写 FSDP，并记录 tok/s。

## TL;DR

实测（202M 模型，4×L20）：FSDP 比 DDP 省显存 25%（2 卡）/ 37%（4 卡）；
**手写版没做 reshard 反而比 DDP 更占显存**——省显存靠"用完就放"。

## 本章要回答的问题

1. FSDP 相比 DDP 实际省多少显存？（模型参数 + 梯度 + 优化器状态 3 部分）
2. 手写 FSDP 的显存与官方 FSDP 差距多大？
3. 显存省下来的代价（通信量/吞吐）是什么？

## 验证记录

| 配置 | 峰值显存 | tok/s | 说明 |
| --- | --- | --- | --- |
| 单卡（202M, B=2, S=512） | 4.07 GB | 18,314 | 基线 |
| DDP 2 卡 | 4.88 GB | 5,170 | 每卡复制全量（参数+梯度+Adam） |
| 官方 FSDP1 2 卡 | **3.67 GB** | 5,069 | 分片 + reshard |
| 官方 FSDP1 4 卡 | **3.06 GB** | 3,307 | 分片更细 |
| 手写 FSDP 2 卡 | 5.69 GB | 3,084 | **未做 reshard，全量常驻** |

**结论（绑定本次配置）**：
1. FSDP 比 DDP 省 ~25% 显存（2 卡）→ 4 卡省 37%；模型小、激活主导时
   收益有限，大模型（参数/优化器状态主导）收益接近 1/W。
2. **手写版显存反而最高**——它只有分片语义（forward gather + backward
   reduce_scatter），没有 post-forward reshard 释放。这正好证明：
   **"FSDP 省显存的关键不是分片本身，而是 reshard（用完即释放全量）"**。
   官方 FSDP 的 reshard 生命周期（chapter 04 笔记 01）是显存收益的落点。
3. 手写版吞吐最低：无通信/计算 overlap（同步 collective）。
4. 多卡吞吐低于单卡：符合 handoff 的 L20 结论（PCIe 通信、无 NVLink、
   小 batch 时通信无法隐藏）。

## 测量规范

- `torch.cuda.reset_peak_memory_stats()` + 训练循环里
  `torch.cuda.max_memory_allocated()`。
- 固定 batch/seq，记录峰值与吞吐；标注模型尺寸、world size。
- 结论只绑定本次配置（handoff 里 L20 的 MFU 天花板 ~14% 的背景）。
