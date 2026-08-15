# 07-pipelining — PP：把层切开，流水线跑

目标：读透官方 pipelining——`PipelineStage`（阶段）+ `Schedule1F1B`（调度）。
并对照手工实现的串行/1F1B 流水线。官方基于**自动图切分**（`split_spec`
标注切分点），手工版按层数直接切。

## TL;DR

PP = 按层切开模型分给多卡，数据像流水线流过；**1F1B 调度**（一次前向一次
反向交替）不减少气泡，但它把激活显存从 O(micro-batch 数) 降到 O(段数)，
并让反向提前开始。官方训练 loss 与单设备一致。

## 本章要回答的问题

1. `PipelineStage` / `Schedule1F1B` / `split_module` 各做什么？
2. 官方 1F1B 的 warmup 公式与手工实现的对应关系？
3. 官方 PP 训练与单设备训练是否数值等价（loss 一致）？

## 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_pp | 2×NCCL（PP=2） | PASS（1F1B 训练 loss 与单设备一致） |
| demo_pp | 2×Gloo（PP=2） | PASS |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
2026-08-15。注意：本版本 API 为 `pipeline()` + `Pipe.build_stage()` +
`Schedule1F1B(stage, n_microbatches, loss_fn)`；loss 经 `step(..., losses=[])`
列表回传，step 返回值是输出拼接。

## 使用手册（本层关键坑）

- `split_module` 自动切图：切分点由 `examples` 传入（每 stage 的输入输出
  样例），无需手写层分配；模型必须可跟踪（module 前向可被 torch.fx 符号
  跟踪）。
- 每个 rank 只跑自己的 `PipelineStage`；`Schedule1F1B.step()` 传入全部
  microbatches 的参数（各 stage 内部按调度收发）。
- loss 只在最后一个 stage 计算（`schedule.step(loss_fn=...)`）；其他 stage
  的 loss 为 None。
- PP 需要 gradient accumulation 语义：`num_microbatches > 1` 时每 step 内部
  自动累积，等价一次大 batch。

## 源码地图

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

## 两种实现层次

| 维度 | 官方 pipelining | 手工 1F1B（最小复刻） |
| --- | --- | --- |
| 切分 | `split_spec` 自动（fx 图） | 手工按层数分配 |
| 通信 | `_batch_p2p`（send/recv 批量融合） | 逐 micro-batch send/recv |
| loss | 仅最后 stage，`losses` 列表回传 | 最后 stage 计算后广播 |
| 调度 | 集中式（stage 外驱动） | 每 stage 内循环 |

warmup 公式两者一致：`num_stages - stage_index`（官方）与
`pp_size - pp_rank - 1`（手工）本质相同。想理解 1F1B 调度，手写一版
（chapter 00 的 p2p 原语即可）；生产中用官方版。
