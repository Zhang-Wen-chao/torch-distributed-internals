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

- Megatron-LM：`/mnt/storage01/zhangwenchao02/tools/Megatron-LM`（megatron-core 0.18.0）
- DeepSpeed：0.19.3（容器隔离 venv）
- PyTorch：`2.10.0a0+a36e1d39eb.nv26.01.42222806`
