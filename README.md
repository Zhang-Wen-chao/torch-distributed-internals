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

- [x] 仓库骨架 + 章节地图
- [ ] 00-primitives 源码走读
- [ ] 01-device-mesh
- [ ] 02-ddp
- [ ] 03-zeroredundant
- [ ] 04-fsdp
- [ ] 05-hsdp
- [ ] 06-tp
- [ ] 07-pipelining
- [ ] 08-rpc

## License

MIT
