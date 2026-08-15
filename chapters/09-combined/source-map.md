# 09-combined 组合方法

## 4 卡上的可行组合

| 组合 | 卡数 | 说明 |
| --- | --- | --- |
| TP=2 × PP=2 | 4 | 每 stage 内部 2 卡 TP；stage 间 p2p |
| TP=2 + DP=2 | 4 | DDP 包 TP（DDP 官方支持 device_mesh） |
| TP=2 × PP=1 + DP=2 | 4 | DDP 复制 TP 模型 |
| 完整 3D（TP×PP×DP） | ≥8 | 本仓库环境不可行，标注 |

## demo_3d 设计（TP=2 × PP=2）

```
模型（6 个 Linear）：切 2 个 stage（每 stage 3 个 Linear）
每个 stage 内部：parallelize_module（TP=2）：第一个 Linear Colwise、
最后一个 Rowwise、中间层的接续（全量/分片注意布局）
PP 通信：stage 间 p2p（官方 pipelining 的 Schedule1F1B 支持
TP 组：需要 stage 的 forward/backward 由 TP 组内两个 rank 并行执行
```

实现（官方组件）：
1. `init_device_mesh("cuda", (2, 2), mesh_dim_names=("pp", "tp"))`
2. 模型按层切 2 个 stage：stage0 = [Linear0, ReLU, Linear1, ReLU]，stage1 = [Linear2]
   （用 `pipe_split()` 或 `split_spec` 标注）
3. **每个 stage 内部先 TP 化**（`parallelize_module(stage_mod, mesh["tp"], plan)`）
4. `pipeline()` + `Schedule1F1B`，PP 组 = mesh["pp"]
5. 训练与单设备 loss 对比

注意：官方 pipelining + TP 组合需要 stage 用 TP mesh 建（官方有
`pipeline(..., group=mesh["pp"])` + stage 内 TP 的示例模式）。

## 验证标准

- PP=2 训练 loss == 单设备 loss（同 seed 同数据同 micro-batch 累积）。
- 记录 bubble 与吞吐（可选）。
