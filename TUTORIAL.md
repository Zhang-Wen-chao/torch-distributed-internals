# TUTORIAL — 从零搞懂 PyTorch 分布式训练

> 主线教程：顺着读，读完你就懂了整个领域在解决什么问题、每个东西是干什么的。
> 每节都有可运行的最小代码。想深挖某节，跳转到对应的 `chapters/` 章节。

---

## 0. 怎么读这个教程（30 秒）

整个分布式训练领域只解决**两个问题**：

1. **算得不够快** → 多卡一起算（数据并行、张量并行、流水线并行）
2. **放得不够大** → 把模型/状态切开放（FSDP、ZeRO）

教程的每一节都先讲"问题是什么"，再给最小代码。代码都来自
`chapters/*/demos/`，可以直接跑（需要 `torch>=2.1`）。

---

## 1. 起点：单卡为什么不够

一个模型训练 = 前向（算输出）+ 反向（算梯度）+ 优化器更新（改参数）。
GPU 上有三样东西占显存：

```
显存占用 = 模型参数 + 梯度 + 优化器状态（Adam 还要 ×2）+ 中间激活
```

以 1B 参数模型、AdamW、fp32 为例算一下：

```
参数：      1B × 4B = 4 GB
梯度：      1B × 4B = 4 GB
Adam 状态： 1B × 4B × 2 = 8 GB
────────────────────────────
固定开销：  16 GB  ← 模型还没算就已经没了
```

单卡显存 24~80 GB，放不下大模型；就算放得下，算一个 epoch 也太慢。
所以需要**多张卡**。多卡的第一件事不是"怎么分模型"，而是——

**多张卡之间怎么"互相说话"。** 这就是第 2 节。

---

## 2. 两张卡的第一次对话：进程组 + all_reduce

多卡训练时，每张卡跑一个**进程**（叫 rank）。它们要通信，第一件事是
**互相认识**：

```python
import torch.distributed as dist

# 每个进程执行这行：通过一个共享地址找到彼此，组成"进程组"
dist.init_process_group(backend="nccl")   # GPU 用 nccl；纯 CPU 调试用 gloo
print(f"我是第 {dist.get_rank()} 号进程，我们一共 {dist.get_world_size()} 个")
```

认识之后，最常用的一个通信动作是 **all_reduce**：每个人拿出一个数，
加总之后**每个人都拿到总和**：

```python
t = torch.tensor([dist.get_rank() + 1.0], device="cuda")  # rank0 是 1，rank1 是 2
dist.all_reduce(t, op=dist.ReduceOp.SUM)                   # 大家都变 3.0
```

**这就是整个分布式训练的原子动作。** DDP 同步梯度用 all_reduce，FSDP
还原参数用 all_gather（all_reduce 的表亲），TP 合并结果用 reduce_scatter。
所有上层花样，全是这几个原语拼的。

> 跑一遍：`torchrun --standalone --nproc_per_node=2 chapters/00-primitives/demos/demo_allreduce.py --device cpu`
>
> 深挖：`chapters/00-primitives/`（一次 all_reduce 从 Python 到 NCCL 的完整路径）

---

## 3. 数据并行（DP）：最朴素的"一起算"

有了通信，最直觉的多卡用法是**数据并行**：

- 同一个模型，复制到每张卡上
- 每张卡吃**不同的数据**
- 各算各的梯度，**all_reduce 平均**后各自更新

核心代码只有 3 行（对比单卡训练，只多了这些）：

```python
# 单卡训练循环
for data in dataloader:
    loss = model(data).mean()
    loss.backward()
    optimizer.step()

# 2 卡数据并行：只多 3 行
for data in my_rank_dataloader:          # 每张卡吃自己的数据（划重点）
    loss = model(data).mean()
    loss.backward()
    for p in model.parameters():         # ┐
        dist.all_reduce(p.grad)          # │ 1. 所有卡的梯度求和
        p.grad /= dist.get_world_size()  # │ 2. 除以卡数 = 平均
    optimizer.step()                     # ┘ 3. 各自用平均梯度更新
```

结果：每张卡上模型的参数永远一样，但吞吐约等于 N 张卡叠加。

> 跑一遍：`chapters/02-ddp/demos/demo_ddp_mechanism.py` 的 ManualDDP 就是这个的最小实现
>
> 深挖：`chapters/02-ddp/`

---

## 4. DDP：把第 3 节自动化 + 两个关键优化

手写的 3 行能工作，但 PyTorch 官方的 `DistributedDataParallel`（DDP）
做了两件你手写时想不到的事：

**优化 1：分桶（bucket）**。100 个参数就 all_reduce 100 次太浪费，DDP
把参数按 ~25MB 打包成"桶"，一个桶满才发一次通信——通信次数少、包大，
速度快。

**优化 2：桶满即发（与计算重叠）**。反向传播是按层倒着算的，第 10 层的
梯度算完时第 9 层还没开始。DDP 不等整轮反向结束——**一个桶的梯度齐了就
立刻 all_reduce**，通信在第 9 层计算的同时进行，藏掉了通信时间。

所以 DDP 的本质 = **梯度分桶 + 梯度就绪 hook + 桶满即 all_reduce**。
就这些，没有别的魔法。

```python
# 官方用法就一行包装
from torch.nn.parallel import DistributedDataParallel
model = DistributedDataParallel(model)   # 其余训练循环和单卡一模一样
```

> 深挖：`chapters/02-ddp/notes/`（C++ 侧 Reducer 的完整机制）+ 仓库里的
> `ManualDDP`（150 行手写版，与官方训练参数逐元素一致）

---

## 5. 显存还是不够：FSDP（切参数）

数据并行把模型复制了 N 份——**显存没有任何节省**。模型 16 GB 固定开销
每张卡都要吃。

FSDP 的思路：**参数、梯度、优化器状态全都切碎了分给每张卡**，训练时
哪层要用，临时拼起来：

```
静止状态：每张卡只持有 1/N 的参数 + 1/N 的梯度 + 1/N 的 Adam 状态
前向第 i 层：all_gather 把这一层的完整参数拼出来 → 算 → 用完立刻释放
反向第 i 层：同样拼出完整参数 → 算梯度 → reduce_scatter 把梯度切碎分掉
```

生命周期一句话：**用之前 gather，用完就释放（reshard）**。

关键认知（仓库 10 章实测验证过）：**省显存靠的不是"切"，是"用完就放"**。
只切不释放（仓库手写版就是这样）反而更占显存。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, sharding_strategy="FULL_SHARD")  # 一行包装，显存降到 ~1/N
```

> 深挖：`chapters/04-fsdp/`（生命周期走读 + 手写版 + FSDP2/DTensor）
> 实测数字：`chapters/10-memory/`（DDP 4.88GB → FSDP 3.67GB；省显存的关键
> 是 reshard 的实测证明）

---

## 6. 模型单卡根本放不下：张量并行（TP）

FSDP 解决"状态显存"，但**激活**和前向计算本身还在单卡上跑。模型大到
单卡算不动（比如一层 Linear 权重 10 GB），就要把**计算本身**切开：

```
普通 Linear:        y = x @ W            (W 太大，放不下)
切成 2 份:          W = [W1 | W2]        (每张卡持有一半列)
                    卡 0 算 x @ W1，卡 1 算 x @ W2
                    各自输出一半 → 拼起来（或直接进入下一层的另一半）
```

对应的就是官方 TP 的 ColwiseParallel（按列切）和 RowwiseParallel
（按行切、把分片输出加总）：

```python
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
parallelize_module(model, tp_mesh, {"layer1": ColwiseParallel(), "layer2": RowwiseParallel()})
```

> 深挖：`chapters/06-tp/`（前向输出与单设备完全一致的验证）

---

## 7. 层太多：流水线并行（PP）

TP 切开一层，PP 切开**一整串层**：GPU 0 管第 1~6 层，GPU 1 管第 7~12 层。
数据像流水线一样流过：

```
时刻:  1    2    3    4
GPU0:  F(1) F(2) B(1) F(3) ...
GPU1:       F(1) F(2) B(1) ...
```

这样中间有"气泡"（GPU 空转等上游），所以大家用 **1F1B** 调度
（一次前向一次反向交替）填气泡。官方：

```python
from torch.distributed.pipelining import pipeline, Schedule1F1B
pipe = pipeline(model, mb_args=(example,), split_spec={"layer6": SplitPoint.END})
schedule = Schedule1F1B(stage, n_microbatches=8, loss_fn=loss_fn)
```

> 深挖：`chapters/07-pipelining/`（1F1B 训练 loss 与单设备一致的验证）

---

## 8. 组合起来 + 实测数字

真实大模型训练是**组合**的：TP 切层内 + PP 切层间 + DP 复制整条管线。
进程组怎么组织？靠 DeviceMesh（一张"谁和谁通信"的网格）+ DTensor
（"知道自己在网格上怎么分布"的张量）：

```
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
# 2 行（dp）× 2 列（tp）——每列一个 TP 组，每行一个 DP 组
```

仓库 4×L20 实测的结论速览（完整数字在对应章节）：

| 实测 | 结论 |
| --- | --- |
| 显存（202M 模型） | DDP 4.88GB > FSDP 3.67GB；**手写版没做 reshard 反而 5.69GB** |
| checkpoint 叠加 FSDP | 17.0GB → 4.98GB（省 70.7%） |
| NCCL 算法（4 卡） | ~1MB 以下 Tree 快，以上 Ring 快 |
| TP×DP 组合 | 训练 loss 与单设备完全一致 |

> 深挖：`chapters/09-combined/`、`chapters/10-memory/`、`chapters/11-nccl-internals/`

---

## 9. 下一步读什么

1. 想验证自己真懂了：跑 `chapters/02-ddp/demos/compare_ddp_manual_vs_official.py`
   （150 行手写 DDP 与官方数值等价）
2. 想搞懂细节：对应章节的 `README.md`（白话 + 源码地图）→ `notes/`（走读）
3. 想踩坑时对照：每章 README 的"常见坑"段落

全仓库 13 个主题的完整目录见 [README.md](README.md)。
