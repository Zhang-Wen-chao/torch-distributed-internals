# 00-primitives — 通信原语：多 GPU 怎么"互相商量"

这是整个仓库的第一章，也是所有后续章节的地基。如果你读完觉得"不接地气"，
先看下面的白话解释，再决定从哪里入手。

## 这一章到底在讲什么

分布式训练 = 多个 GPU 一起训练同一个模型。多个 GPU 之间需要**通信**，
这些最基础的通信动作就叫**原语（primitives）**：

| 原语 | 动作 | 你会遇到的场景 |
| --- | --- | --- |
| `all_reduce` | 所有人把自己的数拿出来加总，**每个人都拿到总和** | DDP 每步同步梯度 |
| `broadcast` | 一个人的数据复制给**所有人** | 训练开始时的参数初始化 |
| `all_gather` | 每个人的数据拼在一起，**所有人都拿到完整版** | FSDP 前向还原全量参数 |
| `reduce_scatter` | 完整数据切碎、按人归约，**每人只拿自己那块** | FSDP 反向切分梯度 |
| p2p（send/recv） | 两个人之间单向传数据 | 流水线并行的 stage 间 |

打个比方：4 个人各拼一块大拼图，为了保证四份拼图完全一致，每隔一会儿
每个人把自己拼的那块**复制四份分给大家**（all_gather）——"分给大家"
这个动作就是原语。

## 一个问题贯穿全章

**你写下一行 `dist.all_reduce(t)`，到 GPU 真正把数据传完，中间发生了什么？**

```
你的代码                            ← 这里开始
  └─ torch.distributed.all_reduce   （Python 侧，notes/01）
       └─ ProcessGroupNCCL::allreduce（C++ 侧，notes/02）
            └─ ncclAllReduce(...)    （NCCL 库，真正干活）
                 └─ GPU 之间传数据
```

全章 4 篇笔记就是沿着这条链读下来的：

| 笔记 | 回答的问题 |
| --- | --- |
| [01](notes/01-init-and-allreduce-path.md) | `init_process_group`（让 N 个进程互相认识）+ `all_reduce` 的 Python 调用路径 |
| [02](notes/02-processgroupnccl-allreduce.md) | C++ 侧：stream 选择、异步语义、tensor 生命周期 |
| [03](notes/03-nccl-comm-init.md) | NCCL 通信器（连接）怎么创建，子组怎么零成本派生 |
| [04](notes/04-coalescing.md) | 多个小消息合并成一次通信（DDP 梯度分桶的底层机制） |

## 建议阅读顺序

1. **先跑演示**，看输出，建立直觉：

   ```bash
   torchrun --standalone --nproc_per_node=2 \
     chapters/00-primitives/demos/demo_allreduce.py --device cpu
   # 输出：PASS: all collective checks
   ```

2. 再读 `notes/01`（Python 侧路径）和 `notes/02`（C++/NCCL 侧）。
   记不住细节没关系，记住"一次通信从 Python 到 NCCL 分了三层"就够。
3. 回到本页的"常见坑"，对照你跑 demo 时可能遇到的报错。

## 演示脚本

| 脚本 | 验证什么 | 运行 |
| --- | --- | --- |
| `demo_allreduce.py` | all-reduce 四种 op 语义、子进程组、异步 work | CPU(Gloo)/GPU(NCCL)，2~4 进程 |
| `demo_async_stream.py` | 异步 + 多 stream：通信与计算重叠 | 仅 GPU(NCCL) |
| `demo_coalescing.py` | 批量 collective（DDP 梯度桶的底层） | CPU/GPU，2~4 进程 |

## 常见坑（新手最容易踩）

- **collective 是集体操作**：组内所有进程必须用**相同顺序**调用相同操作，
  否则集体挂死。超时只会把"死等"变成报错。
- **backend 选错**：CPU tensor 用 `gloo`，CUDA tensor 用 `nccl`。NCCL
  不支持 CPU tensor。
- **`barrier()` 不是调试工具**：它也是 collective，所有进程都要调用。
- **忘了销毁**：程序结束前调用 `destroy_process_group()`，否则警告/资源泄漏。
- **多卡必须绑设备**：每个进程绑定自己的 GPU
  （`torch.cuda.set_device(LOCAL_RANK)`），否则报
  `Duplicate GPU detected`。

## 验证记录（2026-08-14，4×L20，torch 2.10.0a0 nightly）

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_allreduce | 2/4×NCCL + 2/4×Gloo cpu | PASS |
| demo_async_stream | 2×NCCL | PASS（async 排队期间计算 29.5 ms，未等 NCCL） |
| demo_coalescing | 2/4×NCCL + 2×Gloo cpu | PASS |
| tests/test_single_process.py | CPU | 5/5 PASS |
