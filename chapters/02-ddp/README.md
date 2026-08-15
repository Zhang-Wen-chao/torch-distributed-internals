# 02-ddp — DistributedDataParallel

目标：读透 DDP——它是用得最多的并行 wrapper，也是理解 FSDP/HSDP 的跳板。
DDP 的核心不在 Python 而在于 C++ 的 `Reducer`：参数分桶、autograd 梯度 hook、
桶就绪即 all-reduce、与 optimizer.step() 的契约。

## 本章要回答的问题

1. `DistributedDataParallel(module)` 初始化时做了什么？（参数排序、跨 rank
   校验、Reducer 创建、bucket 组装）
2. 反向时"每个参数梯度就绪 → 桶满 → all-reduce"的完整机制？
3. bucket 是怎么分的？`bucket_cap_mb` 怎么影响通信粒度？
4. `find_unused_parameters` / `static_graph` / `gradient_as_bucket_view` 各自
   解决了什么问题？
5. 手写一个最小 DDP（autograd hook + 分桶 + all-reduce）能否与官方数值一致？

## 目录

```text
chapters/02-ddp/
├── README.md             # 本章入口（本文件）
├── source-map.md         # 源码地图
├── notes/
│   ├── 01-python-side.md # Python 侧：初始化与 forward 编排
│   └── 02-reducer-cpp.md # C++ Reducer：分桶 / hook / all-reduce
└── demos/
    ├── demo_ddp_mechanism.py     # 手写 DDP（不用官方 wrapper）
    └── compare_ddp_manual_vs_official.py  # 与官方 DDP 数值对照（待写）
```

## L20 验证记录（2026-08-15）

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
4×L20。环境变量同 chapter 00。

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_ddp_mechanism | 2×NCCL | PASS（3 步训练跨 rank 参数一致） |
| demo_ddp_mechanism | 4×NCCL | PASS |
| demo_ddp_mechanism | 2×Gloo cpu | PASS |
| compare_ddp_manual_vs_official | 2×NCCL | PASS（3 步参数逐元素一致，rtol=1e-5） |
| compare_ddp_manual_vs_official | 4×NCCL | PASS |
| compare_ddp_manual_vs_official | 2×Gloo cpu | PASS |

实测发现（详见 notes/02 末尾）：本版本 `register_post_accumulate_grad_hook`
的 grad 参数数值不可靠（~12.5x），需从 `p.grad` 取梯度；桶 pending 计数必须
每步重置。

## 使用手册（本层关键坑）

- DDP 要求所有 rank 的模型**参数顺序一致**、初始值一致（初始化时从 rank 0
  broadcast，`_sync_params_and_buffers`）；否则静默训练错模型。
- 反向必须从 DDP 包装后的模块输出开始（forward 里做了 `prepare_for_backward`）。
- `find_unused_parameters=True` 有性能代价（每步扫描计算图 + local_used_map
  all-reduce）；模型结构稳定时用 `static_graph=True` 或干脆不用该参数。
- `gradient_as_bucket_view=True` 让 `param.grad` 直接指向 bucket 内存，省一次
  拷贝，但要求"每个参数只注册一个 hook"（`:1685-1692` 的检查）。
- DDP 不做梯度累积的除法修正以外的任何魔法：all-reduce 求和后除以 world_size
  （`set_divide_factor`，`reducer.cpp`）。梯度裁剪、AMP 都在 DDP 外部。
