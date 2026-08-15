# 08-rpc — RPC：点对点远程调用

目标：读透 `torch.distributed.rpc`——与 collective（c10d）不同，RPC 是
**点对点**的远程调用（`rpc_sync` / `rpc_async` / `remote` + RRef），适合
参数服务器、流水线/模型级分布式，以及不是"全体同步"的场景。

## TL;DR

RPC = **点对点**的远程函数调用（不是集体通信），适合参数服务器等非对称
拓扑。三种调用：`rpc_sync`（等结果）/ `rpc_async`（Future）/ `remote`
（远端对象 RRef）。

## 本章要回答的问题

1. `init_rpc` 做了什么？与 `init_process_group` 的关系（RPC 复用其
   rendezvous）？
2. `rpc_sync` / `rpc_async` / `remote` 三种调用的区别？
3. 参数服务器模式怎么搭？（worker 远程调用 server 的函数）

## 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_parameter_server | 2 进程（server + trainer） | PASS（rpc_sync/rpc_async/RRef 三种调用 + 参数更新正确） |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
2026-08-15。踩坑：RPC 远程函数必须是模块级可导入（局部 lambda 不可 pickle）；
容器 hostname 解析不可靠，须显式 loopback rendezvous + `TP_SOCKET_IFNAME=lo`。

## 使用手册（本层关键坑）

- `init_rpc` 会**复用** `MASTER_ADDR/MASTER_PORT` 做 rendezvous（基于已初始化
  的进程组），也可以只初始化 RPC。
- 远程函数必须在**所有 worker 上可导入**（同名模块），且不能是本地闭包。
- `RRef` 是远程对象的引用（owner 持有对象、管理引用计数），`to_here()` 取回
  本地；`remote()` 立即返回 RRef，适合持续持有的服务对象。
- RPC 不保证与 collective 混用的顺序；同一进程里 RPC 和 c10d 的通信要
  小心死锁（避免一边等 RPC 一边等 collective）。
- 优雅退出：`rpc.shutdown()` 需所有 worker 参与。

## 源码地图

走读基线：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`。
目录：`torch/distributed/rpc/`。

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | `init_rpc`（入口）：初始化 RPC agent + 分布式 autograd |
| `api.py`（965 行） | `rpc_sync`/`rpc_async`/`remote`/`RRef`/`get_worker_info`/`shutdown` |
| `backend_registry.py` | 后端注册（TENSORPIPE 默认） |
| `server_process_global_profiler.py` 等 | 服务端 profiling / 分布式 autograd 支持 |

## 关键行号索引（api.py）

| 位置 | 内容 |
| --- | --- |
| `:319` | `shutdown(graceful=True, timeout)` |
| `:420` | `get_worker_info(worker_name)`：按名字取 WorkerInfo |
| `:546` | `remote(to, func, args, kwargs)`：返回 RRef，立即返回 |
| `:759` | `rpc_sync(to, func, args, kwargs)`：阻塞等待结果 |
| `:833` | `rpc_async(to, func, args, kwargs)`：返回 Future |

### __init__.py

| 位置 | 内容 |
| --- | --- |
| `:92` | `init_rpc(name, backend=TENSORPIPE, rank, world_size, rpc_backend_options)`：复用进程组 rendezvous（`env://`），构造 RpcAgent |
| `:89-90` | `__all__` 合并 `api.__all__` + `backend_registry.__all__` |

## 与 c10d collective 的对照

| 维度 | c10d（collective） | RPC |
| --- | --- | --- |
| 通信模式 | 全体/子组同步（SPMD） | 点对点调用（任意拓扑） |
| 数据 | tensor 集体操作 | 任意 Python 对象（序列化） |
| 适用 | DDP/FSDP/TP/PP 的通信 | 参数服务器、模型级协调 |
| 失败模型 | watchdog 统一报错 | 每调用 timeout |
| 底层 | NCCL/Gloo | TensorPipe（gRPC 风格） |

**两者互补**：大模型训练通信用 collective（低开销高吞吐），控制面/异构
协调用 RPC（灵活、对象级）。
