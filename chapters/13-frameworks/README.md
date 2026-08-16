# 13-frameworks — 三框架源码级对比（Megatron / PyTorch / DeepSpeed）

读了两边原版源码之后，回答"它们的实现到底差在哪"。三个对比：

- [Megatron TP vs PyTorch TP](notes/01-megatron-tp-vs-torch-tp.md)
- [Megatron PP vs PyTorch pipelining](notes/02-megatron-pp-vs-torch-pp.md)
- [DeepSpeed ZeRO-3 vs PyTorch FSDP](notes/03-zero3-vs-fsdp.md)

## TL;DR

| 对比 | 一句话结论 |
| --- | --- |
| TP | Megatron 把通信**硬编码在自定义 autograd.Function 的 backward 里**并手动与计算重叠；PyTorch 是"贴 DTensor 标签，通信自动插入" |
| PP | Megatron 的调度器是自研 20 年的成熟品（interleaved 1F1B、CUDA graph 兼容、MoE 感知）；PyTorch 是新晋者，覆盖基本 1F1B，深度优化还在追 |
| ZeRO-3 vs FSDP | 同数学；ZeRO-3 是 **fp16 参数 + fp32 分区主副本**双副本结构、模块级 hooks 按需物化、NVMe offload；FSDP 是**单副本参数 + FlatParameter** 生命周期、DTensor 表达 |

## 走读版本

- Megatron-LM：megatron-core 0.18.0+be2b2cd
- DeepSpeed：0.19.3（容器隔离 venv）
- PyTorch：`2.10.0a0+a36e1d39eb.nv26.01.42222806`

## 实测记录（2026-08-16，4×L20，torch 2.10 / DeepSpeed 0.19.3 / megatron-core 0.18.0）

### FSDP vs ZeRO-3（同一 202M 模型、同数据、同 AdamW，2 卡，3 步）

| | 峰值显存 | 吞吐 | loss 序列 |
| --- | --- | --- | --- |
| PyTorch FSDP | **3.94 GB** | **6088 tok/s** | mean loss 下与 ZeRO-3 一致 |
| DeepSpeed ZeRO-3 | 4.87 GB | 4937 tok/s | 同上 |

- **数值等价**：mean loss 下 3 步后参数向量 `max_abs_diff = 0.0`（逐元素
  完全一致）——"同数学"的实测证明。
- **ZeRO-3 显存高 19%**：双副本结构的常驻代价（fp16 分片 + fp32 分区主
  副本）实测出来了，与 notes/03 源码分析吻合。
- 注意：sum loss（量级 364→-8773）下两者出现 3.9e-4 的参数差——FP32
  归约顺序的 1e-7 级差异被超大梯度放大，不是实现错误。

### Megatron TP vs PyTorch TP（同一 2 层 MLP、同权重、同数据，TP=2，3 步）

| | loss 序列 | 吞吐 |
| --- | --- | --- |
| PyTorch DTensor TP | -0.0008, -0.0968, -0.1929 | 27553 tok/s |
| Megatron TP | -0.0009, -0.0967, -0.1926 | 92163 tok/s |

- loss 趋势一致（第 1/2 步相对差 ~1e-3，属 FP32 归约顺序差异）。
- 吞吐差异**不具普适性**：模型仅 4M 参数，PyTorch 侧 DTensor dispatch
  开销主导；大规模模型需另测（本章不覆盖）。

### 运行方式

- `demos/bench_fsdp.py`：系统 python + torchrun（2 卡）
- `demos/bench_zero3.py`：DeepSpeed venv 的 python（`-m torch.distributed.run`）
- `demos/bench_megatron_tp.py`：系统 python + torchrun（2 卡）
