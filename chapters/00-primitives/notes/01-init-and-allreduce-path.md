# 01 — init_process_group 与一次 all_reduce 的完整路径

> 走读版本：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`（NGC PyTorch 26.01 容器）
> 走读日期：2026-08-14
> 文件：`torch/distributed/distributed_c10d.py`（6275 行）

## 全景：一个进程的分布式状态

c10d 的 Python 侧全部状态放在单例 `_world = _World()` 里（`:720`）。`_World` 持有：

| 字段 | 作用 |
| --- | --- |
| `_default_pg` | 默认进程组（含全体 rank），所有不带 `group=` 的 API 都用它 |
| `pg_map` | `ProcessGroup -> (backend 名, Store)` |
| `pg_group_ranks` | `ProcessGroup -> {全局rank: 组内rank}`，子组的 rank 映射 |
| `pg_names` | `ProcessGroup -> 组名` |
| `pg_coalesce_state` | coalescing 上下文里暂存的集体操作列表 |

`GroupMember.WORLD` 是默认进程组的"指针"，通过元类 `_WorldMeta` 的属性 `WORLD`
直接读写 `_world.default_pg`（`:724-748`）。未初始化时是 `None`——这是
`dist.get_rank()` 在未调用 `init_process_group` 时行为异常（返回 -1）的根源。

## `init_process_group` 做了什么（`:1577`）

```
init_process_group(backend=None, init_method=None, ...)
  ├─ 校验：store 与 init_method 互斥；都不给则 init_method = "env://" (:1706)
  ├─ backend 推断：显式 backend > device_id 推断 > "undefined"（懒加载）(:1745-1754)
  ├─ timeout 默认值：NCCL 10 分钟、其他 30 分钟（_get_default_timeout, :751）
  ├─ 找 store（rendezvous）:
  │    init_method 是 URL（env:// 从 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
  │    取数）→ rendezvous() 迭代器产出 (store, rank, world_size) (:1800-1803)
  │    store 再包一层 PrefixStore("default_pg", store)（防多系统 key 冲突，:1808）
  ├─ _new_process_group_helper(...) (:1810) → 创建默认进程组
  ├─ _update_default_pg(default_pg) (:1823)
  │    → _world.default_pg = pg + torch._C._distributed_c10d._set_global_rank(rank)
  └─ 装分布式 excepthook（异常时打 [rankN] 前缀，:1832-1849）
```

## backend 与进程组绑定：`_new_process_group_helper`（`:1896`）

它按 `backend_config.get_device_backend_map()` 逐 device 创建 backend 对象：

- Gloo：`ProcessGroupGloo(prefix_store, group_rank, group_size, timeout)`（`:2040`）
- NCCL：先构造 `ProcessGroupNCCL.Options()`，再
  `ProcessGroupNCCL(prefix_store, group_rank, group_size, options)`（`:2078`）
- `device_id` 存在时（`init_process_group(device_id=...)`）：NCCL 立即建通信器
  （eager init），后续子组可用 `ncclCommSplit` 从默认组派生（`_get_split_source`,
  `:1873`）——"首组快、子组零开销"的机制。

默认组创建的注意点：`_new_process_group_helper` 要求**所有**全局 rank 都调用
（包括不在新组里的 rank，它们返回 `GroupMember.NON_GROUP_MEMBER = -100`，
`:1912-1914, :1961-1970`）。这就是"new_group 必须全体 rank 同步调用"的由来。

## 一次 `dist.all_reduce(t)` 的路径（`:2915`）

```
all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)
  ├─ has_torch_function → 交给 torch function mode（Dynamo/export 兼容）(:2968)
  ├─ _check_single_tensor 校验 (:2979)
  ├─ _rank_not_in_group(group) → 不在组内则警告并返回（不参与）(:2980)
  ├─ 复数 tensor：view_as_real 后按实数处理 (:2984-2987)
  ├─ opts = AllreduceOptions(); opts.reduceOp=op; opts.asyncOp=async_op (:2989)
  ├─ group 为 None 时取 _get_default_group() (:2992-2993)
  ├─ coalescing 上下文（pg_coalesce_state 里）→ 只把操作记下来，不真正发 (:2995-3002)
  └─ work = group.allreduce([tensor], opts)   ← Python → C++ 的边界（pybind）
       └─ async_op 为真返回 work；否则 work.wait() 同步等待 (:3004-3011)
```

关键点：

1. **`group.allreduce([tensor], opts)` 是 Python 到 C++ 的边界**：`ProcessGroup`
   是 `torch._C._distributed_c10d.ProcessGroup` 的绑定对象，`allreduce` 落到
   `ProcessGroupNCCL::allreduce`（下一步走读目标）。tensor 包成列表是因为 C++
   接口支持批量/coalesced 操作。
2. **同步语义在 Python 侧完成**：`async_op=False` 时调用 `work.wait()`；
   NCCL 后端历史上有过"CPP 层不同步"的兼容逻辑（`:3008-3011`），即等待动作
   全部由 Python 侧兜底，保证行为一致。
3. **未初始化的行为**：`_get_default_group()` 抛
   `"Default process group has not been initialized"`（`:1335-1339`）——演示脚本
   `demo_allreduce.py` 在调用 `init_process_group` 前先查 `RANK` 环境变量，
   就是为了在 torchrun 之外运行时报出清晰错误。

## 组内 rank 映射

- `get_rank(group=None)` 返回当前进程在组内的 rank（`:2400`）；
- `_pg_group_ranks` 保存全局 rank → 组内 rank 的映射，子组 collective 时用
  `get_global_rank(group, group_rank)`（`:1054`）换算回全局 rank。

## 下一步

1. C++ 侧：`ProcessGroupNCCL::allreduce` → `ncclGroupStart`/`ncclAllReduce`、
   `NCCLComm` 池化、stream 语义（`torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`）。
2. Gloo vs NCCL 的差异实测（CPU/GPU demo）。
3. 异步与多 stream 重叠：`demo_async_stream.py`。
