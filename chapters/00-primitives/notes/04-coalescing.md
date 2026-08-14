# 04 — Coalescing：DDP 批量梯度同步的底层

> 走读版本：Python `2.10.0a0+a36e1d39eb.nv26.01.42222806`，C++ `pytorch/pytorch@a36e1d39eb`
> 走读日期：2026-08-14
> 文件：`torch/distributed/distributed_c10d.py`（`:2662-2726`）、
> `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`（`:3471-3545`）

DDP 每轮反向会把梯度**分桶**后一次性 all-reduce（而不是逐参数）。批量通信的
底层就是 coalescing：

## Python 侧：`_coalescing_manager`（内部 API）

```
with dist._coalescing_manager(group, device, async_ops) as cm:   :2668
    dist.all_reduce(t1, group)        # 只记入 op_list，不真正发
    dist.all_reduce(t2, group)        #   (all_reduce 里检查 pg_coalesce_state,
    ...                               #    :2995-3002)
cm.wait()                             # 退出时统一发

退出逻辑（:2679-2717）：
  op_list 从 pg_coalesce_state 弹出
  快路径（fast path）：
    全 all_reduce → 一次 group.allreduce_coalesced(tensors, opts)   :2688-2693
    全 all_gather → allgather_into_tensor_coalesced
    全 reduce_scatter → reduce_scatter_tensor_coalesced
  否则抛错（混合 reduce op 不允许，:2664-2666）
```

要点：

- **op_list 的暂存发生在每个 collective 函数里**（如 all_reduce 的
  `:2995-3002`），用户代码完全无感。
- 三种"可 coalesce"的 op 才能走快路径；`dist.all_reduce_coalesced` 已被标记
  deprecated（`:3015`），新代码一律用 context manager。
- 在 `device` 参数存在时还会走 C++ 侧 `group._start_coalescing(device)`
  （`groupStart()` = `ncclGroupStart`）→ `_end_coalescing`（`groupEnd()`）。

## C++ 侧：ncclGroupStart/End

```
startCoalescing()                                             :3471
  ├─ 重置 coalesced 状态（device/comm/tensor 暂存）            :3481-3483
  ├─ coalescing_state_ |= CoalActive                          :3484
  └─ groupStart()  = ncclGroupStart()                         :3485

collective() 里（coalescing 激活时）                           :3645-3663
  ├─ 只登记 coalescedDevice_/coalescedComm_（要求同设备同 comm）
  ├─ 不创建 Work、不立即 enqueue NCCL 调用
  └─ tensor 暂存进 coalescedTensors_

endCoalescing(optype)                                         :3490
  ├─ coalescedComm_ 为空 → 直接 groupEnd() 返回（空操作）      :3491-3496
  ├─ 创建合并的 Work（一个 work 代表整批操作）                 :3518-3526
  ├─ start event 记录在 ncclStream 上                         :3536-3538
  └─ groupEnd() = ncclGroupEnd()：一次启动整批 NCCL 调用       :3540-3544
```

## 为什么快？

1. **减少 kernel launch / 同步开销**：`ncclGroupStart/End` 之间排队的 NCCL 调用
   由 NCCL 内部合并调度，通信量不变但启动开销被摊薄。
2. **一个 Work、一次 wait**：整批操作一个 work 对象统一等待，Python 侧轮询
   开销从 N 次降到 1 次。
3. DDP 的梯度 bucket 正是按这个机制设计的：bucket 满 → `_coalescing_manager`
   批量 all-reduce（第 02 章会看到完整拼图）。

## 下一步

1. 演示脚本 `demo_coalescing.py`（一次 context 批量 all-reduce == 逐 tensor
   all-reduce）。
2. 进入 01-device-mesh。
