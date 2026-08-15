# 07-pipelining 源码地图

走读基线：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`。
目录：`torch/distributed/pipelining/`。

| 文件 | 职责 |
| --- | --- |
| `_IR.py` | `PipelineStage` 的图表示 + `split_module`（自动切分） |
| `stage.py`（1588 行） | `PipelineStage`：前向/反向 chunk、p2p 收发 op |
| `schedules.py`（3438 行） | 调度器：`ScheduleGPipe` / `Schedule1F1B` / `ScheduleInterleaved1F1B` / `ScheduleLoopedBFS` |
| `microbatch.py` | micro-batch 拆分（`split_args_kwargs`） |
| `_backward.py` | 反向的 loss 传播 |

## 关键行号索引（schedules.py）

| 位置 | 内容 |
| --- | --- |
| `:727` | `ScheduleGPipe`：全 forward 后全 backward（GPU 利用率低） |
| `:846` | `Schedule1F1B`：warmup + 1B1F 交替 |
| `:873-876` | **warmup 公式**：`min(n_microbatches, num_stages - stage_index)` |
| `:885-914` | warmup 阶段：fwd recv → forward → fwd send |
| `:918-961` | 1B1F 阶段：bwd recv+send 融合 → backward → forward |
| `:963-964` | cooldown：收尾 bwd sends |
| `:2493` | `ScheduleInterleaved1F1B`：多 stage 交错（v-shape） |

## 与 mini-megatron 1F1B 对照

| 维度 | 官方 pipelining | mini-megatron |
| --- | --- | --- |
| warmup | `min(n_mb, num_stages - stage_index)` | `pp_size - pp_rank - 1` |
| 切分 | `split_module` 自动（fx 图，按 `examples` 样例） | 手工按层数分配 |
| 通信 | `_batch_p2p`（send/recv 批量融合，`:888-924`） | 逐 micro-batch send/recv |
| loss | 仅最后 stage，`loss_fn` 传入 | 最后 stage 计算后广播 |
| 调度实现 | 集中式（stage 外显式驱动） | 每 stage 内循环 |

**本质**：官方是"图切分 + 集中式调度器"的工程化 1F1B；mini-megatron 是
"手工切层 + 阶段内循环"的最小复刻。数学调度一致，工程结构不同。
