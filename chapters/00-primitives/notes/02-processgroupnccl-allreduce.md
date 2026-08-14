# 02 — ProcessGroupNCCL：一次 all_reduce 在 C++/NCCL 侧

> 走读版本：`pytorch/pytorch @ a36e1d39eb`（NGC 26.01 nightly 对应 commit）
> 走读日期：2026-08-14
> 文件：`torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`（5987 行）

Python 侧 `group.allreduce([tensor], opts)` 经 pybind 落到
`ProcessGroupNCCL::allreduce`。完整路径：

```
ProcessGroupNCCL::allreduce (tensors, opts)                :4454
  ├─ 校验：单 tensor；复数 view_as_real (:4457-4466)
  ├─ check_gpu_single_tensor（NCCL 只接受 CUDA tensor）   :4467
  ├─ intra-node 快速路径：intraNodeComm_（NVLink/SHM 优化） :4469-4476
  └─ allreduce_impl(tensor, ...)                           :4499
       └─ collective(input, output, lambda, ...)           :4430

collective 核心模板 (vectors 版本)                          :3611
  ├─ 设备定位 + OptionalCUDAGuard（防止换设备）            :3624-3627
  ├─ CUDA graph 捕获检查（graph 内不允许 NCCL）            :3629-3631
  ├─ seqCollective_++（collective 计数，调试用）            :3634-3636
  ├─ 取/建 NCCL 通信器：
  │    ncclComm = getNCCLComm(key)                         :3640
  │    没有 → initNCCLComm(key, device, opType)（懒加载）  :3642
  ├─ 选 stream（关键）:
  │    asyncOp 为假 → 用「当前 CUDA stream」               :3667-3668
  │    asyncOp 为真 → 用本进程组专属 ncclStreams_[key]
  │                   并先 syncStream（等输入就绪）        :3670-3672
  ├─ initWork：构造 WorkNCCL（含 start/end CUDA event）     :3676
  ├─ recordStream：把 tensor storage 登记到 ncclStream
  │    （防止 tensor 在集体操作完成前被 allocator 回收）    :3714/3740
  ├─ fn(inputs[0], outputs[0], comm, ncclStream)            :3751
  │    = ncclAllReduce(data_ptr, data_ptr, numel,
  │       ncclDataType, ncclReduceOp, comm, stream)         :4440-4447
  └─ post + end event + future 完成 + workEnqueue（挂到
      NCCL watchdog 可取消的队列）                          :3760+
```

## 三个关键机制

### 1. NCCL 通信器池（懒初始化）

- `getNCCLComm(key)` 按 **device key** 查 `devNCCLCommMap_`，没有就
  `initNCCLComm`（`:3228, :2929`）→ `ncclCommInitRank` 或 `ncclCommSplit`。
- 一个进程组对**每个设备**持有一个 `NCCLComm`；同一进程组不同 device 的
  collective 互不干扰（这也是每进程绑定单 GPU 的原因之一）。
- `init_process_group(device_id=...)` 会立即建通信器（eager），并把子组创建
  转成 `ncclCommSplit`（见笔记 01），省掉子组重新建立连接的开销。

### 2. stream 语义：同步 vs 异步

- **默认同步（async_op=False）**：NCCL kernel 直接排到**当前 CUDA stream**，
  Python 侧再 `work.wait()`。等待发生在 `WorkNCCL::wait`（`:788`）：
  `synchronize()` 让当前 stream 等 ncclStream → `isCompleted()`（cudaEventQuery）
  轮询 → 超时则 abort 通信器并抛异常（`:808-842`）。
- **异步（async_op=True）**：NCCL 排到进程组专属 `ncclStreams_[key]`，调用前
  用 `syncStream`（cudaStreamWaitEvent）保证 ncclStream 等当前 stream 上的输入
  就绪；之后用户代码可继续在默认 stream 上计算，最后 `work.wait()` 同步。
- 这就是 DDP/FSDP 能"通信与计算重叠"的底层依据：**一个 NCCL 通信器 + 一组
  专属 stream + event 编排**。第 04 章 FSDP 会再次看到同一模式。

### 3. tensor 生命周期（allocator 安全）

`CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), ncclStream)`
（`:3714, :4214`）：把 tensor 的存储登记到 ncclStream，保证即使 Python 侧
tensor 已销毁，allocator 也不会在 NCCL kernel 读完前回收这块显存。异步场景下
还会把 tensor 暂存到 `work->stashed_for_allocator_safety_`（`:3713`）。

## 与 Gloo 的差异（对照点）

| 维度 | NCCL | Gloo |
| --- | --- | --- |
| 设备 | CUDA tensor（CPU tensor 直接报错） | CPU tensor |
| 调用方式 | ncclAllReduce 异步入队 + event 同步 | 同步阻塞（线程池） |
| 通信器 | ncclComm 池（按 device key） | 无通信器概念，每次建上下文 |
| stream | 当前 stream 或专属 ncclStream | 无 CUDA stream 语义 |
| 多卡拓扑 | 依赖 NCCL 网络检测（PCIe/NVLink） | 走 TCP/共享内存 |

## 下一步

1. `demo_async_stream.py`：用 `async_op=True` + 单独 stream 实测通信与计算重叠。
2. NCCL 初始化与 `ncclCommSplit` 的机制（`initNCCLComm`, `:2929`）。
3. Coalescing：`startCoalescing/endCoalescing`（`:3471`）——DDP 的
   `all_reduce_coalesced` 底层。
