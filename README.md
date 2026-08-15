# torch-distributed-internals

把 PyTorch 原生分布式（`torch.distributed`）从底层通信原语到上层 wrapper
逐层讲透的教学仓库：**每一个机制都配有源码走读、手写复现、和与官方实现的
数值等价验证**。

如果你想知道"`DistributedDataParallel` 底层到底怎么工作的"、"FSDP 为什么
能省显存"、"一次 `all_reduce` 从 Python 到 NCCL 经历了什么"——这个仓库
就是为这些问题写的。

## 你能从这份仓库里学到什么

| 问题 | 对应章节 |
| --- | --- |
| 一次 `dist.all_reduce(t)` 从 Python 到 NCCL 的完整路径？ | [00-primitives](chapters/00-primitives/) |
| 进程组怎么组织？`mesh["dp"]` / `mesh["tp"]` 是什么？ | [01-device-mesh](chapters/01-device-mesh/) |
| DDP 的梯度分桶、hook、all-reduce 机制？ | [02-ddp](chapters/02-ddp/) |
| 官方 ZeRO-1 怎么分片优化器状态？ | [03-zeroredundant](chapters/03-zeroredundant/) |
| FSDP 的 unshard/reshard 生命周期？FSDP2/DTensor 是什么？ | [04-fsdp](chapters/04-fsdp/) |
| HSDP 的分片组 × 复制组？ | [05-hsdp](chapters/05-hsdp/) |
| 官方张量并行（Colwise/Rowwise）？ | [06-tp](chapters/06-tp/) |
| 官方流水线并行（1F1B 调度）？ | [07-pipelining](chapters/07-pipelining/) |
| RPC 参数服务器模式？ | [08-rpc](chapters/08-rpc/) |
| TP+DP 组合并行？ | [09-combined](chapters/09-combined/) |
| DDP vs FSDP 到底省多少显存？（实测） | [10-memory](chapters/10-memory/) |
| NCCL 的 ring/tree 算法怎么选？（实测） | [11-nccl-internals](chapters/11-nccl-internals/) |
| Activation Checkpoint 怎么省显存？（实测） | [12-activation-checkpoint](chapters/12-activation-checkpoint/) |

## 每章的结构

```
chapters/<编号>-<主题>/
├── README.md       # 章入口：白话讲清本章要解决什么问题 + 源码地图
├── demos/          # 最小可运行脚本（可独立验证内部机制）
└── notes/          # 逐段源码走读（仅核心章节 00-04 有）
```

三个核心章节（02-ddp、04-fsdp、05-hsdp）除了读官方源码，还各自附了一个
**手写 mini 版实现**（不依赖官方 wrapper），并在 L20 上验证了
**手写版与官方实现训练参数逐元素一致**——这是本仓库验证主线：读懂之后，
你亲手实现一遍，再和官方对答案。

## 快速开始

```bash
git clone <project-url>
cd torch-distributed-internals

# 1. 先跑一个最简演示（单机 CPU，两个进程）
torchrun --standalone --nproc_per_node=2 \
  chapters/00-primitives/demos/demo_allreduce.py --device cpu
# 预期输出：PASS: all collective checks

# 2. 再跑手写 DDP 对照（有 GPU 时）
torchrun --standalone --nproc_per_node=2 \
  chapters/02-ddp/demos/demo_ddp_mechanism.py --device cpu

# 3. 运行测试
python -m pytest -q
```

> 需要 `torch>=2.1`（带 `torch.distributed`）。GPU 演示需要多卡（2~4 张）。

## 怎么读（两条路线）

**入门路线（想理解"大概怎么回事"）**：按章节顺序读每章的 `README.md`，
跑一遍对应 `demos/` 脚本，然后读 `notes/`。约 1~2 天。

**深入路线（想真正搞懂实现）**：核心是 02（DDP）和 04（FSDP）两章，先读
笔记源码走读，再看手写 mini 版代码，最后看对照脚本怎么验证数值等价。

## 验证环境与可信度

所有演示在 4×L20（PCIe，无 NVLink）上实测通过，torch `2.10.0a0`
（NGC PyTorch 26.01 nightly）。每个章节 README 末尾有验证记录表
（配置 × 结果）。**注意**：结果绑定特定硬件/版本；不同环境结论可能不同
（尤其是性能数字，见 [10-memory](chapters/10-memory/) 和
[11-nccl-internals](chapters/11-nccl-internals/) 的边界说明）。

## 已知边界

- 完整 3D 并行（TP×PP×DP）需要 ≥8 卡，本仓库在 4 卡上验证了 TP×DP 组合
- 多节点跨机 NCCL 未验证（单机环境）
- FSDP `SHARDED_STATE_DICT`（DTensor 表达）未单独测试（FULL 路径已覆盖）
- `SequenceParallel` 只写了源码地图，未展开

## License

MIT
