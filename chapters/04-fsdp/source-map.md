# 04-fsdp 源码地图

走读基线：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`。

## Python 侧主文件

| 文件 | 职责 |
| --- | --- |
| `torch/distributed/fsdp/fully_sharded_data_parallel.py`（2199 行） | FSDP1 主类：初始化、forward 编排、状态 |
| `torch/distributed/fsdp/_runtime_utils.py`（1654 行） | 运行时钩子：unshard/reshard/pre/post backward |
| `torch/distributed/fsdp/_flat_param.py` | `FlatParameter`：参数展平与分片核心 |
| `torch/distributed/fsdp/_init_utils.py` | 初始化：handle 创建、参数广播 |
| `torch/distributed/fsdp/_state_dict_utils.py` | state_dict（FULL/SHARDED） |
| `torch/distributed/fsdp/_shard_utils.py` | 分片切分工具 |

## 关键行号索引

### fully_sharded_data_parallel.py

| 位置 | 内容 |
| --- | --- |
| `:117` | `class FullyShardedDataParallel(nn.Module, _FSDPState)` |
| `:836` | `forward`：`_fsdp_root_pre_forward` → 子模块 `_pre_forward` |

### _runtime_utils.py

| 位置 | 内容 |
| --- | --- |
| `:348` | `_pre_forward`：unshard（all_gather 全量）+ 注册 post-backward hook |
| `:411` | `_pre_forward_unshard`：`_unshard()` + stream 等待 + forward prefetch |
| `:438` | `_post_forward`：`_post_forward_reshard`（释放全量） |
| `:630` | `_pre_backward_hook`：再次 unshard + 注册 reduce-scatter |
| `:701` | `_post_backward_hook`：`_post_backward_reshard` + `_reduce_grad` |
| `:793` | `_post_backward_reshard`：释放全量参数 + backward prefetch |
| `:831` | `_reduce_grad`：**reduce-scatter 梯度**（默认路径） |
| `:890` | `_get_reduce_scatter_tensors`：**padding**（`numel_to_pad`，`:897`） |
| `:1084` | `_post_backward_final_callback`：写回分片梯度（root 收尾） |

## 推荐阅读顺序

1. `_flat_param.py` 的 `FlatParameter`（展平 + 分片 + padding）
2. `_runtime_utils.py`：`_pre_forward → _post_forward → _pre_backward →
   _post_backward → _reduce_grad` 的生命周期链
3. `_get_reduce_scatter_tensors`（padding 细节）
4. `_post_backward_final_callback`（root 收尾、分片梯度写回）
