# 01 — FSDP1 生命周期：FlatParameter 与 unshard/reshard

> 走读版本：`torch 2.10.0a0+a36e1d39eb.nv26.01.42222806`
> 走读日期：2026-08-15
> 文件：`torch/distributed/fsdp/_runtime_utils.py`、`fully_sharded_data_parallel.py`

## 数据结构：FlatParameter

FSDP 把一组参数**展平成一个叶子张量**（FlatParameter），然后按 world_size
均匀分片：

```
params: [w0(512), b0(32), w1(16*32), ...]  →  flat (numel N)
shard_i = flat[ceil(N/W)*i : ceil(N/W)*(i+1)]    # 均匀分片，末尾 padding
```

- 模型的 `nn.Parameter` 被替换为 **flat 的视图**（`_flat_param_to_param`）。
  视图不是叶子 → autograd 梯度最终累积到 **flat.grad**（全量大小）。
- 每个 rank 只持有自己的 shard（`_local_shard`），optimizer 建立在 shard 上。

## 一次 forward + backward 的生命周期

```
forward（每个被包装的子模块前后）:
  _pre_forward(state, handle, ...)                          runtime:348
    ├─ _pre_forward_unshard → _unshard()                    :411/:421
    │     all_gather(全量, shards) → flat.data 还原全量     （unshard_stream）
    ├─ 当前 stream 等 unshard_stream（event 同步）          :426-430
    ├─ 注册 post-backward hook（每轮必须重注册）            :390
    └─ forward prefetch（预取下一层）

  module.forward(...)

  _post_forward → _post_forward_reshard                     :438/:486
    └─ 释放全量参数（free flat 的全量存储，只留 shard）
       —— 这是 FSDP 显存省下来的关键点

backward:
  _pre_backward_hook                                        :630
    ├─ 再次 _unshard()（backward 需要全量参数算梯度）       :677
    └─ handle.prepare_gradient_for_backward()               :694
       —— 只保留 flat 的 shard 部分梯度，全量 grad 设为 None

  _post_backward_hook（flat 的梯度累积完后触发）            :701
    ├─ _post_backward_reshard：再次释放全量参数             :742/:793
    └─ _reduce_grad()                                       :744/:831
          ├─ unsharded_grad = flat.grad（全量）→ flat.grad = None :846-847
          ├─ padding 到 world_size 整数倍                   :848/:890-902
          ├─ dist.reduce_scatter_tensor(shard_grad, 全量)   :858-862
          │    每 rank 拿到自己 shard 的梯度之和
          ├─ 除 world_size（均值，pre/post divide 因子）    :852/:879
          └─ 累积进 shard 梯度（_accumulate_sharded_grad）  :885

  _post_backward_final_callback（root 收尾）                 :1084
    └─ 把 reduce-scatter 后的 shard 梯度写回 flat 的 shard 视图，
       供 sharded optimizer 使用；并处理低精度梯度
```

## 关键机制细节

### 1. unshard 是"通信 + 复制"

`_unshard` 在**独立的 unshard_stream** 上 all_gather，结果复制进 flat 的全量
buffer；当前 stream 用 event 等它（`:426-430`）。这是 FSDP 通信与计算重叠的
落点（同 chapter 00 的 stream 语义）。

### 2. reshard 是"释放显存"

`_reshard` 释放 flat 全量 buffer 的显存（free 掉 gather 产物），只保留
`_local_shard`。**一次 forward/backward 中全量参数被 materialize 两次**：
forward 一次、backward 一次，用完即释放。

### 3. reduce-scatter 的 padding

`_get_reduce_scatter_tensors`（`:890`）：全量梯度 `chunk(world_size)` 后
`F.pad([0, numel_to_pad])` 到整数倍，reduce_scatter 输出每个 shard 大小
的梯度。末尾 shard 的 padding 部分由 sharded grad view 掩码处理
（`_use_sharded_grad_views` 不暴露给优化器）。

### 4. 为什么这是"数据并行"的显存优化版

| | DDP | FSDP1 |
| --- | --- | --- |
| 参数 | 每 rank 全量 | 每 rank 1/W（静态） |
| forward | 直接用 | all_gather 还原（每层） |
| 梯度 | 全量 all-reduce | 全量梯度 reduce-scatter → 每 rank shard |
| optimizer | 全量 | 只在 shard 上 |
| 通信量 | 每步 2×P（fwd 前无） | 每层 2×P（gather + reduce-scatter） |
| 峰值显存 | P + 2P(grad+moments) | P/W + ...（约 1/W） |

通信量没省（甚至略多），省的是**显存**——这就是 FSDP 的定位。

## 下一步

1. 演示：手写 FlatParameter + all_gather/reduce_scatter 生命周期，与官方
   FSDP 数值对照。
2. FSDP2/DTensor：sharding 布局由 DTensor 表达（chapter 01 的 mesh 落点）。
3. HSDP：分片组 × 复制组（chapter 05）。

## 实测补充（2026-08-15，L20，torch 2.10.0a0 nightly）

手写 FSDP（demo_fsdp_mechanism）时踩到三个坑，值得记录：

1. **参数视图替换必须按路径递归**：`module._parameters['0.weight'] = view` 只
   会在顶层字典加无效键，替换不生效、模型仍用原始参数（训练看起来正常但
   flat 从未参与计算，跨 rank "一致"是假阳性）。必须
   `module.get_submodule('0')` 后写 `parent._parameters['weight']`。
2. **优化器参数必须是叶子**：分片参数不能是 flat 的视图（`can't optimize a
   non-leaf Tensor`），要独立存储 + 前向时 all_gather 还原。
3. **step 后快照必须重新 gather**：optimizer 更新独立 shard 存储，flat 只在
   forward 时重建；step 后直接读 flat 拿到的是旧值。`full_params()` 需重新
   all_gather（对应官方 `summon_full_params`）。
4. **官方 FSDP 强制 CUDA**：纯 CPU（Gloo）上构造官方 FSDP 会报
   `_check_flat_params_on_expected_device`；手写版不受限。
