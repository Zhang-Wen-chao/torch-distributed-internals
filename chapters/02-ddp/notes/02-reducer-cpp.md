# 02 — C++ Reducer：分桶、梯度 hook、all-reduce

> 走读版本：`pytorch/pytorch@a36e1d39eb`
> 走读日期：2026-08-14
> 文件：`torch/csrc/distributed/c10d/reducer.cpp`（2501 行）、`reducer.hpp`

DDP 的全部通信逻辑在 `Reducer` 里。Python 侧只是编排（注册 hook、forward 前后
调用），真正的"梯度就绪 → 桶 → all-reduce"在这里。

## 一次反向的完整时序

```
loss.backward()
  └─ autograd 引擎按拓扑序计算梯度，每个参数梯度算完立即调用
       Reducer::autograd_hook(index)                     reducer.cpp:649
         └─ mark_variable_ready(index)                        :876
              ├─ 记录就绪、backward 计时                         :883-885
              ├─ 按 variable_locators_ 找到参数所在桶            :893-894
              ├─ 把梯度拷入桶的扁平缓冲（bucket_views_in）        :896-902
              └─ if (--bucket.pending == 0):
                     mark_bucket_ready(bucket_index)             :911-913
                       └─ all_reduce_bucket(bucket)              :955
                            ├─ 构造 GradBucket（扁平梯度 + 偏移元数据）:966-974
                            └─ run_comm_hook → run_allreduce_hook :939-952
                                 = _AllReduceBySumCommHook → process_group_->allreduce
                                    （async，future_work 挂到桶上）
              最后一批桶就绪 → 注册 engine 回调：
                    finalize_backward()                          :922-935

finalize_backward()                                              :1705
  ├─ 逐个桶 work.wait()（等 NCCL 完成）                          :1718-1730
  ├─ finalize_bucket_dense：结果写回 param.grad                  :1611-1703
  └─ 复位状态（expect_autograd_hooks_=false），等下一次 forward  :1707-1714
```

## 数据结构：Bucket

```
Bucket {
  variables:        桶内参数（按初始化顺序）
  offsets/lengths:  每个参数在扁平桶中的偏移/长度
  bucket_views_in:  扁平 buffer 的视图（backward 拷入梯度）
  bucket_views_out: 归约结果视图
  pending:          尚未就绪的参数数（--到 0 触发 all-reduce）
  future_work:      桶的 all-reduce future
}
```

`initialize_buckets`（`:1064`）：按 Python 侧算好的 `bucket_indices` 组装，
把每个参数 `flatten` 进连续内存；`variable_locators_` 记录
`参数索引 → (桶号, 桶内序号)`，让 hook 在 O(1) 内定位。

## 两个设计要点

### 1. 桶满即通信（overlap）

`pending == 0` 就立刻 all-reduce，不等整轮 backward 结束——桶与后续参数的反向
计算重叠（NCCL 在专属 stream 上，见 chapter 00 笔记 02）。这是 DDP 性能的
核心。`bucket_cap_mb` 越小，overlap 越细、通信次数越多。

### 2. local_used_map：find_unused_parameters 的成本

`find_unused_parameters=True` 时（`dynamic_graph_find_unused()`）：
- `prepare_for_backward` 里 `search_unused_parameters` 扫计算图（`reducer.cpp:1556-1559`）
- 反向最后 `all_reduce_local_used_map`（`:736`）一次额外 all-reduce
- `finalize_bucket_dense` 里对"本 rank 未用"的参数**懒等待** local_used_map
  归约，确认它是否全局未用（`:1615-1650`）；全局未用的梯度不写回、不 all-reduce。

代价：每步多一次整 map 的 all-reduce + 计算图扫描。`static_graph=True`
只扫一次并把结论缓存（`:1550-1558`）。

## 与 chapter 00/01 的衔接

- `process_group_->allreduce(...)` 走 `ProcessGroupNCCL::allreduce`（笔记 00-02），
  异步返回 Work，挂在 `bucket.future_work`。
- 桶的 all-reduce 用 `async_op=true` 语义：计算继续，`finalize_backward` 统一
  `wait()`——这就是"通信与计算重叠"在 DDP 里的落点。
- DDP 的 device_mesh 参数（`:698-703`）取 mesh 的 1D 组——与 chapter 01 的
  DeviceMesh 对接；TP+DP 时 DDP 跑在 dp 维上。

## 下一步

手写最小 DDP（demo_ddp_mechanism.py）：autograd hook + 分桶 + all-reduce，
与官方 DDP 数值对照。

## 实测补充（2026-08-15，L20，torch 2.10.0a0 nightly）

手写 DDP（demo_ddp_mechanism）时踩到两个坑，值得记录：

1. **`register_post_accumulate_grad_hook` 的 grad 参数在本版本数值不可靠**：
   实测 hook 收到的 grad 是真实局部梯度的 ~12.5 倍，且 `grad is p.grad` 为
   False；而 hook 触发时 `p.grad` 已就绪且与 `torch.autograd.grad` 独立计算
   一致（误差 < 1e-7）。结论：**从 `p.grad` 取梯度，不要用 hook 参数**。
   这也解释了官方为什么在 C++ 层（Reducer）处理梯度而不依赖 Python hook
   参数。*此结论仅绑定本版本，跨版本需重新验证。*
2. **桶计数必须每步重置**：`bucket.pending` 只减不增，第二步起永不触发
   all-reduce（对应官方 `prepare_for_backward` 的 `reset_bucket_counting`）。
   修复位置：`sync_and_step` 末尾重置。
3. **手写 DDP 的使用契约**：归约梯度写回发生在 `sync_and_step`（对应官方
   `finalize_bucket_dense` + `optimizer.step()` 的顺序），直接用裸
   `opt.step()` 会拿本地未归约梯度训练。
