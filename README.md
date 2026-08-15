# torch-distributed-internals

把 PyTorch 原生分布式（`torch.distributed`）从底层通信原语到上层 wrapper
逐层讲透的教学仓库：**每一个机制都配有源码走读、手写复现、和与官方实现的
数值等价验证**。

## 从哪里开始

👉 **第一次来？读 [TUTORIAL.md](TUTORIAL.md)** —— 一条线性叙事从头讲到尾：
"单卡不够 → 多卡怎么对话 → DDP → FSDP → TP → PP → 组合"，每节配最小可运行
代码，约 20 分钟读完，建立整个领域的全局图。

`chapters/` 是**深挖参考手册**：教程里哪个主题感兴趣，再进对应章节读源码
走读、跑演示。

## 仓库结构

```
chapters/<编号>-<主题>/
├── README.md   # 章入口：白话定位 + TL;DR + 源码地图 + 常见坑
├── demos/      # 最小可运行脚本（可独立验证内部机制）
└── notes/      # 逐段源码走读（仅核心章节 00-04 有）
```

核心章节（02-ddp、04-fsdp、05-hsdp）各附一个**手写 mini 版实现**（不依赖
官方 wrapper），并在 L20 上验证了**手写版与官方实现训练参数逐元素一致**。

## 章节索引

| 编号 | 主题 | 编号 | 主题 |
| --- | --- | --- | --- |
| [00](chapters/00-primitives/) | 通信原语 | [07](chapters/07-pipelining/) | 流水线并行 |
| [01](chapters/01-device-mesh/) | DeviceMesh | [08](chapters/08-rpc/) | RPC |
| [02](chapters/02-ddp/) | DDP | [09](chapters/09-combined/) | 组合并行 |
| [03](chapters/03-zeroredundant/) | ZeRO-1 | [10](chapters/10-memory/) | 显存实测 |
| [04](chapters/04-fsdp/) | FSDP | [11](chapters/11-nccl-internals/) | NCCL 算法实测 |
| [05](chapters/05-hsdp/) | HSDP | [12](chapters/12-activation-checkpoint/) | 激活检查点 |
| [06](chapters/06-tp/) | 张量并行 | | |

## 快速开始

```bash
git clone <project-url>
cd torch-distributed-internals

# 先跑一个最简演示（单机 CPU，两个进程，不需要 GPU）
torchrun --standalone --nproc_per_node=2 \
  chapters/00-primitives/demos/demo_allreduce.py --device cpu
# 预期输出：PASS: all collective checks
```

需要 `torch>=2.1`（带 `torch.distributed`）。GPU 演示需要 2~4 张卡。

## 版本说明（重要，先读）

| 概念 | 版本 | 说明 |
| --- | --- | --- |
| **运行门槛** | `torch>=2.1,<3` | 跑任何 demo/测试的最低要求（pyproject.toml） |
| **走读/验证版本** | `2.10.0a0+a36e1d39eb.nv26.01.42222806`（NGC PyTorch 26.01 nightly） | 所有笔记的源码行号、所有实测数据、所有踩坑结论都绑定这个版本 |

**笔记里的行号只对这个版本有效**。PyTorch 分布式部分快速演进（例如
pipelining 的 API 在 2.10 有破坏性改动），跨版本阅读时请先核对。

所有演示在 4×L20（PCIe，无 NVLink）上实测通过；每章 README 的"验证记录"
段落有配置 × 结果表。**结果绑定特定硬件/版本**，不同环境结论可能不同
（尤其是性能数字，见 [10-memory](chapters/10-memory/) 与
[11-nccl-internals](chapters/11-nccl-internals/)）。

## 已知边界

- 完整 3D 并行（TP×PP×DP）需要 ≥8 卡，本仓库在 4 卡上验证了 TP×DP 组合
- 多节点跨机 NCCL 未验证（单机环境）
- FSDP `SHARDED_STATE_DICT`（DTensor 表达）未单独测试（FULL 路径已覆盖）
- `SequenceParallel` 只写了源码地图，未展开

## License

MIT
