# 01 — Megatron TP vs PyTorch TP：通信写在哪

> 走读版本：megatron-core 0.18.0 `tensor_parallel/layers.py`（1394 行）vs
> torch 2.10 `tensor/parallel/style.py`
> 走读日期：2026-08-15

## 两边的 TP 数学相同

都是"权重切列（Colwise）/切行（Rowwise）+ 反向归约"，但**通信的放置位置
完全不同**。

## Megatron：通信硬编码在自定义 autograd.Function 里

`layers.py` 的 `LinearWithGradAccumulationAndAsyncCommunication`
（`:456`）是核心——它不是普通 nn.Module，而是一个 **torch.autograd.Function**：

```
forward（:461-504）:
  ├─ sequence_parallel 时 all_gather 输入（:495-496）
  └─ torch.matmul(total_input, weight.t()) —— 纯计算，无通信

backward（:506-598）:
  ├─ grad_input = grad_output.matmul(weight)          :544
  ├─ if allreduce_dgrad:
  │    torch.distributed.all_reduce(                  :557
  │        grad_input, group=tp_group, async_op=True) ← 通信在这里！
  │    # 依赖 CUDA_DEVICE_MAX_CONNECTIONS=1 保证 all_reduce
  │    # 排在权重梯度计算之前（:558-559）
  └─ 然后才算 weight 梯度（wgrad，:574-598）
      —— 通信与 wgrad 计算重叠（异步 all_reduce + 手写调度顺序）
```

关键设计：

1. **通信与计算重叠靠"手写顺序 + 环境变量"**：异步 all_reduce 先排队、
   wgrad 后算，靠 `CUDA_DEVICE_MAX_CONNECTIONS=1` 保证 NCCL 和 matmul
   在同一 stream 上按序执行（`layers.py:539-540` 的注释明说）。
2. **大量 Megatron 特有优化**：`gradient_accumulation_fusion`（main_grad
   累积，`:518-519`）、wgrad 延迟（`grad_output_buffer`，`:522-525`）、
   全局显存 buffer 复用（`get_global_memory_buffer`，`:495`）、FSDP 感知
   （`hasattr(weight, "__fsdp_param__")`，`:577`）、TE kernel 分支（`:580`）。
3. 每种场景一个专用类：`LinearWithFrozenWeight`（`:350`）是另一个
   autograd.Function——冻结权重、只 all-reduce dgrad。

## PyTorch：通信不在模块里，由 DTensor 自动插入

`style.py` 的 `ColwiseParallel._partition_linear_fn`（`:118-128`）只做一件事：

```python
dist_param = nn.Parameter(
    distribute_tensor(param, device_mesh, [Shard(0)], ...)
)   # 只贴标签，没有任何通信代码
```

通信发生在**算子 dispatch 时**：DTensor 的 `__torch_dispatch__` 看到
"Shard(0) 权重 × Replicate 输入"需要什么布局，自动插入 all_gather /
reduce_scatter（`torch/distributed/tensor/_dispatch.py` 的 redistribute）。
模块代码（nn.Linear）完全不变。

## 源码级差异总结

| 维度 | Megatron | PyTorch DTensor |
| --- | --- | --- |
| 通信代码位置 | autograd.Function 的 backward 里**手写** | 算子系统 dispatch 时**自动插入** |
| 模块形态 | 专用类（LinearWithGradAccumulationAndAsyncCommunication 等 4+ 个变体） | 普通 nn.Linear 不变 |
| 重叠方式 | 手写：异步 all_reduce 先排队 + 依赖 CUDA_DEVICE_MAX_CONNECTIONS=1 | DTensor 调度 + 多 stream（较新版本才完善） |
| 深度优化 | main_grad 融合、wgrad 延迟、全局 buffer、TE 分支——为大规模训练打磨多年 | 较少（靠 torch.compile 融合） |
| 适用性 | 绑定 Megatron 的模型写法 | 任意模型贴标签即可 |

**本质**：Megatron 把"通信 + 计算重叠"写死在每个算子里（每个优化场景一个
专用类）；PyTorch 把"布局"抽象出来，通信交给统一的算子系统。前者极致
优化但绑定框架，后者通用但深度优化还在路上。
