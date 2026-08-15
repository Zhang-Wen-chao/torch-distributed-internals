# 03 — FSDP state_dict：FULL vs SHARDED

> 走读版本：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`
> 走读日期：2026-08-15
> 文件：`torch/distributed/fsdp/_state_dict_utils.py`（932 行）

## 机制：state_dict 的三条后处理路径

FSDP 注册了 `_pre_state_dict_hook` / `_post_state_dict_hook`（
`fully_sharded_data_parallel.py` 注册到 `state_dict()`/`load_state_dict()`），
按 `fsdp_state._state_dict_type` 分派（`_state_dict_utils.py:719-723`）：

| 类型 | post hook | 语义 |
| --- | --- | --- |
| FULL_STATE_DICT | `_full_post_state_dict_hook` | **all-gather 全量**，与普通模型 state_dict 一致（保存/分享用） |
| LOCAL_STATE_DICT | `_local_post_state_dict_hook` | 本地分片（每 rank 自己的） |
| SHARDED_STATE_DICT | `_sharded_post_state_dict_hook` | ShardedTensor/DTensor 表达（大模型加载用，rank 间不重复） |

**FULL_STATE_DICT 的实现（`_full_post_state_dict_hook`，`:195` 附近）**：
```
state_dict() → 普通 nn.Module 语义拿到各 rank 的分片值
  → pre-hook 里 unshard 参数（`:680` 的 _enter_unshard_params_ctx，
    writeback=True）
  → post hook 把分片拼成全量（每个 rank 得到完整权重）
```
代价：每个 rank 都会物化**全量**权重（内存峰值回到全量）。

**SHARDED_STATE_DICT**：state_dict 值是 `ShardedTensor`（或 FSDP2 的
DTensor），不物化全量；加载时每 rank 只取自己的分片。适合 checkpoint
落盘 + 续训。

## 使用注意

- 默认 `state_dict_type=StateDictType.FULL_STATE_DICT`（兼容普通流程）；
- 大模型（全量权重放不下）必须用 SHARDED_STATE_DICT；
- `load_state_dict` 同样分派：FULL 时每 rank 从全量里取自己分片；
- 与 DDP 的互操作：`_data_parallel_utils` 处理 DDP 侧转换。

## 下一步

演示：FULL_STATE_DICT 保存/加载 == 单设备权重；加载后续训与基线一致。
