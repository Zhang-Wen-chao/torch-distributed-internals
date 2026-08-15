# 04-fsdp — FullyShardedDataParallel

目标：读透 FSDP1 的核心生命周期——这是显存优化的关键 wrapper，也是理解
FSDP2/DTensor 和 HSDP 的基础。核心机制：**参数展平成 FlatParameter 分片；
forward 前 all-gather 还原全量；backward 时 reduce-scatter 梯度；反向结束
释放全量参数**。

## 本章要回答的问题

1. FSDP1 的参数分片结构？（FlatParameter、shard、padding）
2. forward / backward 的完整生命周期？（pre-forward unshard → post-forward
   reshard → pre-backward unshard → post-backward reduce_scatter + reshard）
3. reduce-scatter 的 padding 与掩码处理？
4. 手写一个最小 FSDP 能否与官方数值一致？
5. FSDP1 vs FSDP2（DTensor）的差异？

## 目录

```text
chapters/04-fsdp/
├── README.md          # 本章入口（本文件）
├── source-map.md      # 源码地图
├── notes/
│   └── 01-fsdp1-lifecycle.md  # FlatParameter + forward/backward 生命周期
└── demos/
    ├── demo_fsdp_mechanism.py         # 手写 FSDP（不用官方 wrapper）
    └── compare_fsdp_manual_vs_official.py  # 与官方 FSDP 数值对照
```

## L20 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_fsdp_mechanism | 2×NCCL | PASS（跨 rank 一致 + 与全量 AdamW 逐元素一致） |
| demo_fsdp_mechanism | 4×NCCL | PASS（同上） |
| demo_fsdp_mechanism | 2×Gloo cpu | PASS（同上） |
| compare_fsdp_manual_vs_official | 2×NCCL | PASS（与官方 FSDP 3 步逐元素一致） |
| compare_fsdp_manual_vs_official | 4×NCCL | PASS |
| compare_fsdp_manual_vs_official | 2×Gloo cpu | SKIP（官方 FSDP 强制 CUDA） |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
2026-08-15。踩坑记录见 notes/01 末尾。

## 使用手册（本层关键坑）

- **优化器必须建立在 `fsdp.parameters()` 上**（它返回的是**分片**后的参数），
  不是原始 module 的参数；否则优化器状态全量复制，分片失效。
- `use_orig_params=True` 时保存/加载 `state_dict` 的语义不同（`state_dict_type`
  的 FULL_STATE_DICT / SHARDED_STATE_DICT）。
- 混合精度在 FSDP 内配置（`MixedPrecision`），不要在外部套 `autocast` 时
  自己 cast 参数——FSDP 会管理参数/梯度/通信的 dtype。
- FSDP 包装粒度决定显存/通信权衡：包装越细，显存越低、通信次数越多。
- 梯度累积时 FSDP 默认每步都同步梯度（`sync_gradients=False` 可关），
  与 DDP 的 `no_sync()` 对应。
