# 03 — NCCL 通信器的创建（initNCCLComm）

> 走读版本：`pytorch/pytorch @ a36e1d39eb`（NGC 26.01 nightly 对应 commit）
> 走读日期：2026-08-14
> 文件：`torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`（`:2929-3128`）

在笔记 02 的 `collective` 里，第一次对某设备做 collective 时
`getNCCLComm(key)` 返回空，触发 `initNCCLComm(key, device, opType)`
（`:3640-3642`）。它是 NCCL 通信器（`ncclComm`）的懒创建路径：

## 流程

```
initNCCLComm(deviceKey, device, opType, ...)                    :2929
  ├─ 校验：deviceKey 非空；若进程组绑定了 device_id（bound_device_id_），
  │    tensor 必须在该设备上（否则报错）                        :2936-2950
  ├─ 确定 comm 大小与 rank：
  │    collective: numRanks=getSize(), rank=getRank()           :3002-3003
  │    单 P2P:     numRanks=2, rank=p2pRank（0 或 1）           :3009-3012
  ├─ 优先用 ncclCommSplit 派生（见下）                          :3034-3055
  ├─ 否则创建新通信器：
  │    rank 0 生成 ncclUniqueId（ncclGetUniqueId）              :3108-3110
  │    通过 store 广播 ncclID（broadcastUniqueNCCLID）          :3114
  │    NCCLComm::create(numRanks, rank, ncclID, deviceIndex, config)
  │        → ncclCommInitRank → 本 rank 加入通信域                :3126-3127
  └─ 存入 devNCCLCommMap_[deviceKey]（此后 getNCCLComm 命中）   :3640
```

## 三个关键机制

### 1. 通信器是"按设备懒创建"的

- 每个进程组对每个 GPU 持有一个 `NCCLComm`，存 `devNCCLCommMap_`（key =
  device index）。
- 第一次用某设备做 collective 时才建；`init_process_group(device_id=...)`
  会提前 eager 建（笔记 01 提到）。
- 这样一次 collective 的正常路径只是查 map：`getNCCLComm` 命中后直接
  `ncclComm` 复用。

### 2. 谁生成 ncclUniqueId？

NCCL 的通信域需要全局唯一的 ID。PyTorch 的做法：**rank 0 生成**（rank 0 或 P2P
的 0 号），然后**通过之前 rendezvous 建立的 store 广播**给其他 rank
（`broadcastUniqueNCCLID`，`:3114`）。store 在这里再次出现——它是分布式初始化的
"控制面"，NCCL ID、rank 号、地址全是走它交换的。

### 3. ncclCommSplit：从已有通信器派生新组

当进程组绑定了 device_id 且是子组时（`:3040-3053`）：

```
NCCLComm::split(parentComm, split_color, rank, config)  → ncclCommSplit()
```

- `split_color` 由 `_process_group_color(global_ranks_in_group)` 生成
  （Python 侧 `distributed_c10d.py:2073`），同一新组的所有 rank 颜色相同。
- 好处：**不需要再走一遍"生成 + 广播 ncclUniqueId + ncclCommInitRank"**，
  直接从父通信器分叉，子组创建几乎零开销——这是 DDP 里
  `find_unused_parameters` 或 FSDP 里反复建子组时性能的关键。
- 注意它是**集体操作**：父组里不在新组的 rank 也要调
  `perform_nocolor_split` 参与（Python 侧 `:1968-1969`）。

### 4. 大规模可扩展初始化（>128 ranks/root）

`useScalableInit`（`:3062-3064`，`TORCH_NCCL_RANKS_PER_ROOT=128`）：
- 多个 root rank 各自生成 ncclID，用 `ncclCommInitRankScalable` 建通信器，
  避免所有 rank 都从 rank 0 拉 ID 的瓶颈。

## 与"Duplicate GPU detected"错误的对应

笔记 02 的 L20 实测中，两个 rank 都用 `cuda:0` 时 NCCL 报
`Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 0`
（NCCLUtils.cpp:94）。NCCL 要求每个进程独占一个 GPU；torchrun 下必须按
`LOCAL_RANK` 绑定设备（`torch.cuda.set_device(local_rank)`），这正是
各并行框架的 `assign_device` 在做的事。

## 下一步

1. Coalescing：`startCoalescing/endCoalescing`（`:3471`）——DDP 批量
   all-reduce 的底层。
2. rendezvous/store：`env://` 之外的 store 类型与 TCPStore 实现。
3. 进入 01-device-mesh。
