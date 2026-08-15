# 12-activation-checkpoint 源码地图

走读基线：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`。
文件：`torch/utils/checkpoint.py`（1666 行）。

| 位置 | 内容 |
| --- | --- |
| `:228` | `CheckpointFunction`：autograd.Function，反向重算 |
| `:349` | `checkpoint(function, *args, use_reentrant=None)` 入口 |
| `:517` | `checkpoint_sequential`：顺序模块按段包装 |
| `:1484` | `_checkpoint_without_reentrant_generator`：非 reentrant 实现 |
| `:393-403` | reentrant vs 非 reentrant 差异（early stop / 图记录） |

## 与 FSDP 的组合

- FSDP 分片参数 + checkpoint 省激活 → 两者正交叠加（大模型标配）。
- 注意事项：checkpoint 包装**必须包含整个 FSDP 单元的 forward**
  （在 FSDP 模块内部按层包装即可），且 `use_reentrant=False` 与 FSDP2 的
  `torch.compile` 兼容性更好。
