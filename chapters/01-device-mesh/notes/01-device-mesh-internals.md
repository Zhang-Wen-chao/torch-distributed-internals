# 01 — DeviceMesh 内部：从 mesh 到进程组

> 走读版本：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`
> 走读日期：2026-08-14
> 文件：`torch/distributed/device_mesh.py`（1367 行）

## 一句话总结

**DeviceMesh = 一个 n 维 rank 网格 + 每维一个（组内成员为该维切片的）进程组。**

## 构造路径

```
init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))     :1274
  ├─ 校验：名字唯一、维度数匹配、device_type 纯字母            :1325-1352
  ├─ layout = _MeshLayout((2,4), suffix_product((2,4))=(4,1))
  │     └─ shape 是 (2,4)，stride 是 (4,1)（行主序 rank 映射）  :1354
  ├─ rank_map = arange(8)（恒等映射：mesh[i,j] == 8 里的 i*4+j）:1357-1358
  └─ DeviceMesh(device_type, _layout, _rank_map, mesh_dim_names) :1359

DeviceMesh.__init__                                             :184
  ├─ mesh tensor → _layout + _rank_map（校验非重叠、连续）      :201-229
  ├─ _setup_world_group_and_device()                            :274
  │    └─ 未初始化则 init_process_group()（懒初始化默认组）     :328-329
  │       未设设备则按 LOCAL_RANK（无则 rank%num_devices）设设备 :339-371
  ├─ _dim_group_names = _init_process_groups(layout, rank_map,
  │     mesh_dim_names, backend_override)                       :275-280
  └─ 计算本 rank 在 mesh 上的坐标：
       rank_coords = (mesh == _rank).nonzero() → _coordinate_on_dim :290-297
```

## 核心：`_init_process_groups`（`:470`）——每维一个进程组

```
for dim in range(len(layout)):                       # 每维
  dim_name = mesh_dim_names[dim] or f"dim_{dim}"
  _init_one_process_group(layout[dim], rank_map, dim_name, backend_override[dim])

_init_one_process_group(sub_layout, rank_map, ...)              :375
  ├─ pg_ranks_by_dim = sub_layout.nest().remap_to_tensor(rank_map)
  │    → 该维的所有切片（如 (2,4) 的 dim0 → [[0,4],[1,5],[2,6],[3,7]]）
  │      dim1 → [[0,1,2,3],[4,5,6,7]]                           :383
  ├─ 特例：子布局覆盖全体 rank 且 backend 默认 → 复用默认组    :400-417
  ├─ 快路径：默认组有 bound_device_id（eager NCCL init）        :427-443
  │    → split_group(parent_pg, split_ranks=...) 一次调用        :436
  │      内部 ncclCommSplit：子组零成本（见 chapter 00 笔记 03）
  └─ 常规路径：逐切片 new_group(ranks=subgroup_ranks, ...)       :450-458
       → 每个切片一次进程组创建（慢，但无需 eager init）
```

以 `(2, 4)` mesh 为例（8 个全局 rank）：

| 维 | 切片（组） | rank 成员 |
| --- | --- | --- |
| dim 0（dp） | `[0,4]` `[1,5]` `[2,6]` `[3,7]` | 跨主机 |
| dim 1（tp） | `[0,1,2,3]` `[4,5,6,7]` | 主机内 |

## `get_group` / 切片

```
get_group(mesh_dim)                                             :614
  ├─ 1D mesh 且未给 dim → 直接返回该组                          :638-639
  ├─ 多维必须给 dim（否则报错）                                 :629-635
  ├─ flatten mesh 的维度 → 查根 mesh 的 _flatten_mapping        :641-647
  └─ 常规：_resolve_process_group(_dim_group_names[mesh_dim])   :649-658
       ——进程组按名字从全局注册表解析（new_group 返回名字）

mesh["tp"]（__getitem__）                                       :546
  ├─ 名字等于全部维 → 返回自身                                  :597-598
  ├─ 否则 _get_slice_mesh_layout(...) 计算子布局                :600
  └─ _create_sub_mesh：新 DeviceMesh（_init_backend=False，
       不新建进程组！复用根的 _dim_group_names 切片）           :669-702
```

关键点：**子 mesh 不创建新进程组**，只是复用根 mesh 对应维的进程组
（`:699-701`，`_init_backend=False`）。所以 `mesh["dp"]` 和 `mesh["tp"]`
零开销。

## 与底层 c10d 的衔接

- `new_group` → `distributed_c10d.py:5286`，最终走笔记 00 的
  `_new_process_group_helper`。
- `split_group` → `distributed_c10d.py` 里对 `ncclCommSplit` 的封装
  （只有 `device_id` eager init 过才可用，`:427-429` 的 `bound_device_id` 判断）。
- `_resolve_process_group` 按组名从全局查进程组（内部 `_world.pg_names`）。

## 下一步

1. 演示：4 rank 上建 `(2,2)` mesh，验证 dp/tp 两组各自成员、切片、分组通信。
2. 进入 DTensor：`sharding` 布局如何落到 mesh 维度上（chapter 04 前置）。
