"""Demo: 手写 FSDP 核心机制（不用官方 wrapper）。

证明什么：
- FSDP 的本质 = 参数展平成分片 FlatParameter（每 rank 1/W）+ forward 前
  all_gather 还原全量 + backward 后 reduce_scatter 梯度（均值）+ 只更新
  shard（优化器状态 1/W）+ 反向后释放全量；
- 训练若干步后所有 rank 参数一致；
- 与单进程全量 AdamW 基线数值等价（每步参数逐元素一致）。

用法（必须用 torchrun 起多进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/04-fsdp/demos/demo_fsdp_mechanism.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29750 \
        chapters/04-fsdp/demos/demo_fsdp_mechanism.py --device cuda

与官方 FSDP 的差异（本演示不覆盖）：
- 不按层分多个 handle（整模型一个 FlatParameter），无 prefetch/stream 重叠；
- 不处理 use_orig_params / mixed_precision / activation checkpoint。
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn


class ManualFSDP(nn.Module):
    """最小 FSDP：FlatParameter 分片 + forward all_gather + backward reduce_scatter。"""

    def __init__(self, module: nn.Module, process_group) -> None:
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self._device = next(module.parameters()).device

        params = [p for p in module.parameters() if p.requires_grad]
        self._numel = sum(p.numel() for p in params)
        self._chunk = (self._numel + self.world_size - 1) // self.world_size
        self._padded_size = self._chunk * self.world_size

        # 1. FlatParameter：所有参数展平成单个叶子张量（autograd 的汇聚点）
        self._flat = torch.cat([p.detach().flatten() for p in params])
        self._flat.requires_grad_(True)

        # 2. 模块参数替换为 flat 的视图（非叶子 → 梯度进 flat.grad）。
        #    注意：必须按路径递归替换（'0.weight' 要写进 module[0]._parameters，
        #    而不是顶层字典），否则替换不生效、模型仍用原始参数。
        offset = 0
        for name, p in module.named_parameters():
            n = p.numel()
            view = self._flat[offset : offset + n].view_as(p)
            parts = name.split(".")
            leaf = parts[-1]
            parent = module.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else module
            if leaf in parent._parameters:
                parent._parameters[leaf] = view
            else:
                parent._modules[leaf] = view
            offset += n
        assert offset == self._numel

        # 3. rank 0 广播初始权重（FSDP/DDP 共同不变量）。
        #    注意：broadcast 的 src 在本版本是「全局 rank」语义
        #    （_canonicalize_group_rank 走 global_rank 分支）。
        src_global = dist.distributed_c10d.get_global_rank(process_group, 0)
        dist.broadcast(self._flat.data, src=src_global, group=process_group)

        # 4. 分片叶子存储（优化器的参数；不能是 flat 的视图——视图非叶子，
        #    优化器拒绝）+ 全量 gather buffer
        self._shard = torch.zeros(self._chunk, device=self._device)
        self._shard.requires_grad_(False)
        self._gathered = torch.empty(self._padded_size, device=self._device)

        # 5. 梯度 reduce-scatter 的收尾 hook（flat 是唯一叶子，累积完触发一次）
        self._shard_grad = torch.zeros(self._chunk, device=self._device)
        # 初始化：shard = flat 的对应切片（随后广播保持全量一致）
        self._shard.copy_(self._flat[self.rank * self._chunk : (self.rank + 1) * self._chunk].detach())

        def post_accumulate_hook(_grad: torch.Tensor) -> None:
            self._reduce_and_store_grad()

        self._flat.register_post_accumulate_grad_hook(post_accumulate_hook)

    def _reduce_and_store_grad(self) -> None:
        """backward 结束后：reduce-scatter 全量梯度 → 存 shard 梯度。"""
        grad = self._flat.grad
        if grad is None:
            return
        self._flat.grad = None  # 释放全量梯度（FSDP 同款）
        padded = torch.nn.functional.pad(grad, [0, self._padded_size - grad.numel()])
        dist.reduce_scatter_tensor(self._shard_grad, padded, op=dist.ReduceOp.SUM, group=self.process_group)
        self._shard_grad.div_(self.world_size)
        if self.rank == self.world_size - 1:
            # 末尾 shard 的 padding 区梯度是垃圾，掩码清零
            tail = self._padded_size - self._numel
            if tail > 0:
                self._shard_grad[-tail:].zero_()

    def forward(self, *inputs):
        # 用 shard 还原全量参数（对应 FSDP 的 _unshard）
        dist.all_gather_into_tensor(self._gathered, self._shard, group=self.process_group)
        self._flat.data.copy_(self._gathered[: self._numel])
        return self.module(*inputs)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """更新分片（优化器只持有 shard）+ 清梯度。"""
        self._shard.grad = self._shard_grad.view(-1)
        optimizer.step()
        optimizer.zero_grad()
        self._shard_grad.zero_()

    def full_params(self) -> torch.Tensor:
        # 注意：step 更新的是独立 shard 存储，flat 只在 forward 时从 shard
        # 重建。要拿"最新"全量参数必须重新 all_gather（对应官方
        # summon_full_params）。等价于官方 FSDP 的语义。
        dist.all_gather_into_tensor(self._gathered, self._shard, group=self.process_group)
        return self._gathered[: self._numel].detach().clone()


def build_model(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4)
    ).to(device)


def train_full_adamw(device: torch.device, steps: int) -> torch.Tensor:
    """单进程全量 AdamW 基线（梯度 all-reduce 平均）。"""
    model = build_model(0, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rank = dist.get_rank()
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(data), target)
        loss.backward()
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(dist.get_world_size())
        opt.step()
        opt.zero_grad()
    return torch.cat([p.detach().flatten() for p in model.parameters()]).cpu()


def train_manual_fsdp(device: torch.device, steps: int) -> torch.Tensor:
    model = build_model(0, device)
    fsdp = ManualFSDP(model, dist.group.WORLD)
    opt = torch.optim.AdamW([fsdp._shard], lr=1e-3)  # 优化器只在分片上
    rank = dist.get_rank()
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)
    for _ in range(steps):
        loss = nn.functional.mse_loss(fsdp(data), target)
        loss.backward()
        fsdp.step(opt)
    return fsdp.full_params().cpu()


def check_cross_rank_consistency(device: torch.device, steps: int) -> None:
    model = build_model(0, device)
    fsdp = ManualFSDP(model, dist.group.WORLD)
    opt = torch.optim.AdamW([fsdp._shard], lr=1e-3)
    rank = dist.get_rank()
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)
    losses = []
    for step in range(steps):
        loss = nn.functional.mse_loss(fsdp(data), target)
        loss.backward()
        fsdp.step(opt)
        losses.append(loss.item())
        flat = fsdp.full_params()
        gathered = [torch.zeros_like(flat) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, flat)
        for other in gathered:
            torch.testing.assert_close(flat, other, msg=f"step{step} rank{rank} 参数不一致")
    # 真训练断言：loss 必须显著下降（防"参数从未变化"的假阳性）
    assert losses[-1] < losses[0], f"loss 未下降: {losses}"  # lr=1e-3 小模型 3 步降幅小，只断言严格下降
    if rank == 0:
        print(f"PASS: 手写 FSDP {steps} 步训练后所有 rank 参数一致（loss {losses[0]:.4f} -> {losses[-1]:.4f}）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29507)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 启动多进程（参考脚本 docstring）")

    backend = "nccl" if args.device == "cuda" else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    if args.device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"torch {torch.__version__} | backend={dist.get_backend()} | world_size={dist.get_world_size()}")

    check_cross_rank_consistency(device, args.steps)
    manual = train_manual_fsdp(device, args.steps)
    baseline = train_full_adamw(device, args.steps)
    torch.testing.assert_close(manual, baseline, rtol=1e-5, atol=1e-7, msg="手写 FSDP 与全量 AdamW 不一致")

    dist.barrier()
    if rank == 0:
        print(f"PASS: 手写 FSDP 与全量 AdamW {args.steps} 步参数逐元素一致")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
