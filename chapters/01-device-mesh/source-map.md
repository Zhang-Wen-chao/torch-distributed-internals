# 01-device-mesh 源码地图

走读基线：容器内 NGC PyTorch 26.01 = `torch 2.10.0a0`
（`2.10.0a0+a36e1d39eb.nv26.01.42222806`）。行号以该版本为准。

## Python 侧

| 文件 | 职责 |
| --- | --- |
| `torch/distributed/device_mesh.py` | DeviceMesh 类（mesh→进程组）、`init_device_mesh`、子 mesh 切片 |
| `torch/distributed/_mesh_layout.py` | `_MeshLayout`：mesh 布局（shape/strides）与 rank 映射 |
| `torch/distributed/_mesh_resources.py` | mesh 全局注册表（`get_current_mesh` 等，DTensor 上下文用） |
| `torch/distributed/distributed_c10d.py` | `new_group` / `split_group` / `_resolve_process_group`（底层） |
| `torch/distributed/tensor/_api.py` | DTensor API（下一章接上） |

## 关键行号索引（device_mesh.py）

| 位置 | 内容 |
| --- | --- |
| `:128` | `class DeviceMesh` |
| `:184` | `__init__`：mesh tensor → `_MeshLayout` + `_rank_map` |
| `:274-280` | `_setup_world_group_and_device()` + `_init_process_groups()` |
| `:324` | `_setup_world_group_and_device`：auto init_process_group + 设设备 |
| `:375` | `_init_one_process_group`：一维 mesh → 一个进程组 |
| `:436` | `split_group`（ncclCommSplit 路径，有 bound_device_id 时） |
| `:452` | `new_group`（常规路径，逐组创建） |
| `:470` | `_init_process_groups`：逐维创建 |
| `:546` | `__getitem__`：`mesh["tp"]` 切片子 mesh |
| `:614` | `get_group(mesh_dim)` |
| `:660` | `get_all_groups()` |
| `:704` | `_create_flatten_mesh`：展平 mesh（FSDP 用） |
| `:1274` | `init_device_mesh`：入口，校验 + 构造 |

## 推荐阅读顺序

1. `init_device_mesh`（`:1274`）→ `DeviceMesh.__init__`（`:184`）
2. `_init_process_groups` → `_init_one_process_group`（`:470/:375`）：
   这是"mesh 的每一维 = 一个进程组"的核心
3. `split_group` vs `new_group` 两条路径的取舍（`:436/:452`）
4. `__getitem__` / `get_group`（`:546/:614`）
5. `_mesh_layout.py`（理解 rank_map 与坐标）
