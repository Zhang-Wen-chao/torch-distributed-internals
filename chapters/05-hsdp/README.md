# 05-hsdp — Hybrid Sharded Data Parallel

目标：HSDP 是 FSDP1 的 HYBRID_SHARD 策略（也是 FSDP2 的 `fully_shard` +
复制组的组合）：**组内分片（reduce-scatter）+ 组间复制（all-reduce）**。
它解决了纯 FSDP 跨节点通信量大的问题：通信被限制在分片组内，复制组之间
只同步分片梯度。

## 本章要回答的问题

1. HSDP 的进程组结构？（分片组 shard_group × 复制组 replicate_group）
2. 梯度路径：组内 reduce-scatter + 组间 all-reduce 的顺序与划分？
3. 与 FSDP 的差异：参数只 gather 一次还是两次？
4. 手写 HSDP 与官方 HYBRID_SHARD 数值对照。

## 目录

```text
chapters/05-hsdp/
├── README.md      # 本章入口（本文件）
├── source-map.md  # 源码地图
└── demos/
    └── demo_hsdp.py  # 手写 HSDP + 与官方 HYBRID_SHARD 对照
```

## L20 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_hsdp | 4×NCCL（分片2×复制2） | PASS（与官方 HYBRID_SHARD 3 步逐元素一致，复制组内一致） |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
2026-08-15。踩坑：`broadcast` 的 `src` 是全局 rank 语义（子组需
`get_global_rank`）；all_gather 输出列表须与输入同 device（NCCL 不校验，
CPU 输出会写坏内存）。

## 使用手册（本层关键坑）

- `device_mesh` 形状决定分组：`init_device_mesh("cuda", (replicate, shard))`，
  第一个维度是复制组、第二个是分片组（FSDP 里 `_sharding_strategy` 配合
  mesh 的 `get_group("replicate")` / `get_group("shard")`）。
- HSDP 的显存与通信折中：分片组越小（组越多）→ 显存越省但复制组间的
  all-reduce 越多。
- `ShardingStrategy.HYBRID_SHARD`（zero-1 风格，反向后不重新 gather）vs
  `_HYBRID_SHARD_ZERO2`（backward 时缓存 gather，省一次）。
