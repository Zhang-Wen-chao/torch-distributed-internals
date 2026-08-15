# 08-rpc 源码地图

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
