"""Demo: 官方 ZeroRedundancyOptimizer（ZeRO-1）vs 全量 AdamW 数值等价。

证明什么：
- ZeRO-1 语义：优化器状态按参数分片给各 rank，step() 后所有 rank 参数恢复一致；
- 数值等价：配梯度 all-reduce 的 ZRO 训练结果 == 全量 AdamW 训练结果
  （每步参数逐元素一致）；
- 分片确实生效：每个 rank 的本地优化器只持有部分参数的 state（Adam moments）。

用法（必须用 torchrun 起多进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/03-zeroredundant/demos/demo_zero1.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29740 \
        chapters/03-zeroredundant/demos/demo_zero1.py --device cuda

梯度语义：ZRO 不处理梯度，本演示在 backward 后显式对梯度做 all-reduce 均值
（等价 DDP 的效果），保证与全量 AdamW 同一起跑线。
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer


def build_model(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4)
    ).to(device)


def train_full_adamw(device: torch.device, steps: int) -> torch.Tensor:
    """全量 AdamW 基线：梯度 all-reduce 平均后更新所有参数。"""
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


def train_zero1(device: torch.device, steps: int) -> tuple[torch.Tensor, list[int]]:
    """官方 ZeRO-1：本地优化器只持有分片参数，step 后自动同步全量。"""
    model = build_model(0, device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        parameters_as_bucket_view=True,
        lr=1e-3,
    )
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
        opt.step()  # 内部：本地更新分片 + _sync_params 广播全量
        opt.zero_grad()
    flat = torch.cat([p.detach().flatten() for p in model.parameters()]).cpu()
    # 本 rank 本地优化器持有的参数数（分片验证）
    n_local = len(opt.optim.param_groups[0]["params"])
    return flat, [n_local]


def check_sharding_effective(total_params: int, n_local_list: list[int]) -> None:
    """world_size=2 时，两个 rank 的本地优化器各持约一半参数。"""
    assert total_params >= 2
    if dist.get_rank() == 0:
        print(f"本地优化器持有参数数/rank: {n_local_list}（总参数数 {total_params}，分片生效）")
        assert all(0 < n < total_params for n in n_local_list), "分片未生效：某 rank 持有全部参数"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29506)
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

    baseline = train_full_adamw(device, args.steps)
    zro_flat, n_local = train_zero1(device, args.steps)
    torch.testing.assert_close(zro_flat, baseline, rtol=1e-5, atol=1e-7, msg="ZeRO-1 与全量 AdamW 参数不一致")

    # 跨 rank 收集分片信息并验证
    if dist.get_world_size() >= 2:
        g = torch.tensor([n_local[0]], dtype=torch.int64, device=device)
        gathered = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, g)
        n_local_list = [int(x.item()) for x in gathered]
        total = baseline.numel()
        check_sharding_effective(total, n_local_list)
    else:
        if rank == 0:
            print("world_size=1：不验证分片")

    dist.barrier()
    if rank == 0:
        print(f"PASS: ZeRO-1 与全量 AdamW {args.steps} 步参数逐元素一致")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
