# torch-distributed-internals

教学型、源码剖析仓库：把 PyTorch 原生分布式（`torch.distributed`）从底层原语到
上层 wrapper 逐层读透——底层怎么实现、怎么使用。它不是生产替代品，不承诺与任何
版本的官方行为完全一致；所有走读笔记标注对应 torch 版本与源码行号。

## 章节

```text
00-primitives   c10d: ProcessGroup、NCCL/Gloo、collective 语义、异步与 stream 重叠
01-device-mesh  DeviceMesh + sharding spec（DTensor 的地基）
02-ddp          bucketing / 梯度 hook / static graph / 与 optimizer 的契约
03-zeroredundant 官方 ZeroRedundancyOptimizer
04-fsdp         FSDP1 分片与生命周期 → FSDP2 / DTensor 统一
05-hsdp         分片组 × 复制组
06-tp           官方 torch.distributed.tensor.parallel
07-pipelining   官方 torch.distributed.pipelining
08-rpc          参数服务器 / 分布式推理
```

每章三件套：

1. **源码走读**：注释版关键路径笔记（官方文件 + 行号 + torch 版本）
2. **机制演示**：最小可运行脚本，单独复现该内部机制（不依赖官方 wrapper）
3. **使用手册**：正确用法 + 坑

## 快速开始

```bash
git clone <project-root>
python -m pip install -e '.[dev]'

# 单进程辅助函数测试
python -m pytest -q

# 两 rank CPU 演示（示例）
torchrun --standalone --nproc_per_node=2 \
  chapters/00-primitives/demos/demo_allreduce.py --device cpu
```

## 与相邻教学项目的关系

| 项目 | 分工 |
| --- | --- |
| [mini-megatron](https://github.com/Zhang-Wen-chao/mini-megatron) | 从零复刻 TP/PP/DP 语义 |
| [mini-deepspeed](https://github.com/Zhang-Wen-chao/mini-deepspeed) | 从零复刻 ZeRO 0/1/2/3 |
| 本项目 | 剖析 PyTorch 官方实现（c10d / DDP / FSDP / DTensor / HSDP / TP / PP / RPC） |

三者独立演进，互不 import。

## 状态

全部章节源码走读 + 演示完成，均已在 4×L20（torch 2.10.0a0 nightly）验证：

| 章节 | 核心验证（L20） |
| --- | --- |
| 00-primitives | all_reduce 语义/子组/异步+stream 重叠、coalescing（2/4×NCCL + Gloo） |
| 01-device-mesh | 2D mesh 组结构/切片/通信域隔离（4×NCCL + Gloo） |
| 02-ddp | 手写 DDP 与官方 DDP 3 步参数逐元素一致（2/4×NCCL + Gloo） |
| 03-zeroredundant | 官方 ZeRO-1 与全量 AdamW 逐元素一致（2/4×NCCL + Gloo） |
| 04-fsdp | 手写 FSDP 与官方 FSDP FULL_SHARD 逐元素一致（2/4×NCCL） |
| 05-hsdp | 手写 HSDP 与官方 HYBRID_SHARD 逐元素一致（4×NCCL） |
| 06-tp | 官方 TP 输出与单设备一致（2×NCCL + Gloo） |
| 07-pipelining | 官方 1F1B loss 与单设备一致（2×NCCL + Gloo） |
| 08-rpc | 参数服务器模式（rpc_sync/rpc_async/RRef） |
| 04b-fsdp2 | FSDP2(fully_shard)/DTensor：Shard/Replicate/redistribute + 数值等价 |
| 04c-state-dict | FSDP FULL_STATE_DICT 保存/加载续训一致 |
| 09-combined | TP=2 × DP=2 组合训练 loss 与单设备一致 |
| 10-memory | 显存实测：DDP 4.88GB > FSDP1 3.67GB > 4卡 3.06GB；手写版未 reshard 最高 |
| 11-nccl | all-reduce 算法实测：~1MB 以下 Tree 优、以上 Ring 优（socket 限 ~1.6GB/s） |
| 12-checkpoint | FSDP+AC 省显存 70.7%（B=16 S=1024 实测） |

每个 wrapper 的"手写 mini 版"（DDP/FSDP/HSDP）与官方实现数值等价，是
本仓库的验证主线；版本差异与踩坑记录在对应章节 notes 末尾。

**已知边界（标注不可行/未覆盖）**：完整 3D（TP×PP×DP）需 ≥8 卡（单机
4 卡不可行）；多节点跨机 NCCL 无第二台机器；FSDP SHARDED_STATE_DICT
（DTensor 表达）未单测（FULL 路径已覆盖）；SequenceParallel 只提了名字
（在 06 的 style.py 地图中）。

## License

MIT
