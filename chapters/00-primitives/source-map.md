# 00-primitives 源码地图

走读基线：容器内 NGC PyTorch 26.01 = `torch 2.10.0a0`
（`2.10.0a0+a36e1d39eb.nv26.01.42222806`）。行号以该版本为准，走读前先核对。

## Python 侧

| 文件 | 职责 |
| --- | --- |
| `torch/distributed/distributed_c10d.py` | 用户 API 主入口：`init_process_group`、`all_reduce` 等函数、`GroupMember`、`_world`/`_pg` 全局状态 |
| `torch/distributed/c10d_logger.py` | 分布式调用日志 |
| `torch/distributed/_functional_collectives.py` | DTensor 背后的 functional collective（新 API，01/04 章用） |

## C++ 侧（c10d）

源码位置：`pytorch/pytorch` 仓库 `torch/csrc/distributed/c10d/`。

| 文件 | 职责 |
| --- | --- |
| `ProcessGroup.hpp/.cpp` | 抽象基类：`allreduce`/`broadcast`/`scatter`/`gather` 等纯虚接口，`ProcessGroup::Work` |
| `ProcessGroupNCCL.cpp` | NCCL 后端：通信池化、`NCCLComm` 与 device 绑定、async 语义、`ncclGroupStart/End` 批量 |
| `ProcessGroupGloo.cpp` | Gloo 后端：CPU 上的 collective 实现 |
| `Ops.cpp` | Python 绑定（pybind）：把 `dist.all_reduce` 绑定到 `ProcessGroup::allreduce` |
| `Backend.hpp` / `backend_registry.cpp` | 后端注册与初始化（`is_gloo_available` 等） |
| `Utils.hpp` / `Rendezvous.cpp` | store 与 rendezvous：`init_process_group` 的地址交换 |
| `Store.hpp` | 文件/Redis/TCP store：rendezvous 用的键值存储 |
| `NCCLUtils.hpp` | NCCL 版本、错误码、`ncclComm` 生命周期辅助 |

## 推荐阅读顺序

1. `distributed_c10d.py` 的 `init_process_group`（了解 store、backend、全局 `_pg`）
2. `ProcessGroup.hpp` 的虚方法表 + `ProcessGroupNCCL::allreduce`（了解一次 collective
   的实现）
3. `Ops.cpp` 的绑定（了解 Python → C++ 的边界）
4. `NCCLUtils.hpp` 的 `ncclComm` 池化（了解多 rank 的通信句柄管理）

## 取源方式

容器内 Python 包只带编译后的 `_C` 扩展，不带 c10d C++ 源码。需要 C++ 源码时从
GitHub `pytorch/pytorch` 拉取对应 tag（本仓库基线为 NGC 26.01 对应的 nightly
commit，见上），放到容器临时目录并在笔记中标注。

- Python 侧路径：`python -c "import torch.distributed, os; print(os.path.dirname(torch.distributed.__file__))"`
