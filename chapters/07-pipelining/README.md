# 07-pipelining — 官方流水线并行（torch.distributed.pipelining）

目标：读透官方 pipelining——`PipelineStage`（阶段）+ `Schedule1F1B`（调度）。
与 mini-megatron 手写的串行/1F1B 流水线对照。官方基于**自动图切分**
（`_IR.py` 的 `split_module` 自动找切分点），mini-megatron 手工按层数切。

## 本章要回答的问题

1. `PipelineStage` / `Schedule1F1B` / `split_module` 各做什么？
2. 官方 1F1B 的 warmup 公式与 mini-megatron 的对应关系？
3. 官方 PP 训练与单设备训练是否数值等价（loss 一致）？

## 目录

```text
chapters/07-pipelining/
├── README.md      # 本章入口（本文件）
├── source-map.md  # 源码地图
└── demos/
    └── demo_pp.py # 官方 1F1B 训练 vs 单设备 loss 对照
```

## L20 验证记录

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
