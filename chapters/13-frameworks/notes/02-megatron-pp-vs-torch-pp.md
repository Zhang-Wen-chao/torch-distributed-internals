# 02 — Megatron PP vs PyTorch pipelining

> 走读版本：megatron-core 0.18.0 `pipeline_parallel/schedules.py`（2409 行）vs
> torch 2.10 `distributed/pipelining/schedules.py`（3438 行）
> 走读日期：2026-08-15

## 调度公式对比（都读到了源码）

**warmup（非交错 1F1B）**：

| | 公式 | 出处 |
| --- | --- | --- |
| Megatron | `pp_size - pp_rank - 1` | `schedules.py:834` |
| PyTorch | `min(n_mb, num_stages - stage_index)` | `schedules.py:873-876` |

数学同构（两者都是"第一个 stage 做最多 warmup"），差一个索引口径
（Megatron 的 rank 从 0 起、PyTorch 的 stage_index 也从 0 起，但 warmup
计数差 1——PyTorch 最后一个 stage 也做 1 个 warmup forward，Megatron 不做）。

**交错（interleaved / V 型）**：

- Megatron 有 `forward_backward_pipelining_with_interleaving`（`:896`），
  warmup 公式复杂（`:841-842`）：
  `(pp_size - pp_rank - 1) * 2 + (num_model_chunks - 1) * group_size`
- PyTorch 有 `ScheduleInterleaved1F1B`（`schedules.py:2493`）

## Megatron 独有的东西（PyTorch 没有/刚有）

读 `schedules.py` 目录结构发现的：

```
forward_backward_no_pipelining          —— CUDA graph 捕获兼容的 PP=1 路径（:849-851）
forward_backward_pipelining_without_interleaving   —— 标准 1F1B
forward_backward_pipelining_with_interleaving      —— 交错 1F1B
combined_1f1b.py                        —— PP+CP（context parallel）组合调度
hybrid_cp_schedule.py                   —— 混合上下文并行调度
fine_grained_activation_offload.py      —— 细粒度激活 offload
multimodule_communicator.py / bridge_communicator.py —— 多模块通信抽象
schedule_table（:866）                  —— (microbatch_id, model_chunk_id) 调度表，
                                           交错调度的核心数据结构
```

对照 PyTorch：`ScheduleGPipe / Schedule1F1B / ScheduleInterleaved1F1B /
ScheduleLoopedBFS / ScheduleZBVZeroBubble / ScheduleDualPipeV`——覆盖也不小，
但 **combined（PP+CP 组合）、MoE 专家并行感知、CUDA graph 兼容、激活
offload** 这些深度优化是 Megatron 多年打磨的结果。

## 通信层对比

- Megatron：`P2PCommunicator`（`p2p_communication.py`）——按 dtype/shape
  缓存 send/recv buffer、批量 P2P、与 TE 集成。
- PyTorch：`_batch_p2p`（`schedules.py` 里）+ `PipelineStage` 的
  get_fwd_send_ops——机制类似，buffer 管理和 dtype 变体更少。

## 结论

| 维度 | Megatron | PyTorch |
| --- | --- | --- |
| 成熟度 | 训练 175B+ 模型的实战品 | 新晋者，快速演进（2.10 大改 API） |
| 覆盖 | 1F1B + interleaved + PP×CP 组合 + CUDA graph + MoE 感知 | 1F1B + interleaved + zero-bubble/dualpipe（名字新，成熟度待验证） |
| 使用方式 | 深度绑定 Megatron 模型结构 | fx 自动切分，任意模型可上 |

一句话：**PP 这一层 Megatron 明显领先——它把流水线调度当成了一个持续
20 年的研究课题；PyTorch 这两年才补课。** 学调度算法读 Megatron 源码，
做通用框架用 PyTorch。
