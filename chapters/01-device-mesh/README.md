# 01-device-mesh — DeviceMesh：谁和谁通信的组织图

目标：读透 `torch.distributed.device_mesh`——它是 FSDP2 / DTensor / TP 的地基。
DeviceMesh 把"一维的 rank 列表"组织成 n 维网格，为每一维自动创建进程组，
通信在不同维度上互不干扰。

## TL;DR

DeviceMesh = 把 GPU 排成一张网格（行是 DP、列是 TP），**每一维自动建一个
通信组**；`mesh["tp"]` 切片零成本。它是 DTensor / FSDP2 / TP 共同的地基。

## 本章要回答的问题

1. `init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))` 到底创建了
   哪些进程组？组内 rank 如何排列？
2. 为什么 DeviceMesh 能"零成本"创建子组？（`split_group` / `ncclCommSplit`）
3. `mesh["tp"]` 切片、`get_group("tp")`、`get_group_rank` 的实现？
4. DeviceMesh 与 DTensor 的关系（`sharding` 如何落在 mesh 上）？

## 验证记录

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`。

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_device_mesh | 4×NCCL | PASS（dp=[0,2] tp=[0,1]，切片复用 PG，通信域隔离正确） |
| demo_device_mesh | 4×Gloo cpu | PASS（同上） |

实测发现的问题：

1. **CPU 路径 backend 解析**：容器有 CUDA 时，DeviceMesh 懒初始化默认组
   （backend=None）会被加速器检测解析成 `{cuda:nccl}`，子组继承后 CPU tensor
   集体操作报 `No backend type associated with device type cpu`。修法：cpu 路径
   先显式 `init_process_group(backend="gloo")`。
2. **`os.environ.get(key, default)` 提前求值**：默认参数 `dist.get_rank()` 会在
   未初始化时抛错，即使 key 存在。必须直接 `os.environ["LOCAL_RANK"]`。

## 使用手册（本层关键坑）

- mesh 数组（布局）必须所有 rank 一致，否则静默挂死（SPMD 模型）。
- `device_type` 不能带 index（`"cuda:0"` 不行，`init_device_mesh:1348`）。
- `mesh_dim_names` 长度必须等于 mesh 维数、名字不能重复（`:1326-1336`）。
- `get_group()` 在多维 mesh 上必须给 `mesh_dim`，否则报错（`:629`）。
- 未初始化进程组时 DeviceMesh 会自己 `init_process_group()`（`:329`），
  并自动按 `LOCAL_RANK` 设设备（`:343-349`）。

## 源码地图

走读基线：NGC PyTorch 26.01 镜像 = `torch 2.10.0a0`
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
