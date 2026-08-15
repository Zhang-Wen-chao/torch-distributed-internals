"""对照: 手写 DDP（demo_ddp_mechanism）与官方 DistributedDataParallel 数值等价。

证明什么：
- 同一模型、同一初始权重、同一数据下，ManualDDP 与官方 DDP 训练同样步数，
  每步之后的参数向量逐元素一致（rtol=1e-5, atol=1e-7）。

用法（必须用 torchrun 起多进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/02-ddp/demos/compare_ddp_manual_vs_official.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29731 \
        chapters/02-ddp/demos/compare_ddp_manual_vs_official.py --device cuda

本脚本 import 官方 DistributedDataParallel 只用于对照，ManualDDP 不依赖它。
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from demo_ddp_mechanism import ManualDDP, build_model


def train(model, wrapper_factory, device: torch.device, steps: int) -> list[torch.Tensor]:
    wrapped = wrapper_factory(model)
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.05)
    rank = dist.get_rank()
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)

    snapshots: list[torch.Tensor] = []
    for _ in range(steps):
        loss = nn.functional.mse_loss(wrapped(data), target)
        loss.backward()
        if isinstance(wrapped, ManualDDP):
            wrapped.sync_and_step(optimizer)  # 写回归约梯度 + step + 重置桶计数
        else:
            optimizer.step()
            optimizer.zero_grad()
        snapshots.append(
            torch.cat([p.detach().flatten() for p in wrapped.parameters()]).cpu()
        )
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29505)
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

    manual = train(
        build_model(0, device).to(device),  # 所有 rank 同种子，避免初始权重差异
        lambda m: ManualDDP(m, dist.group.WORLD),
        device,
        args.steps,
    )
    official = train(
        build_model(0, device).to(device),
        lambda m: DistributedDataParallel(m, process_group=dist.group.WORLD),
        device,
        args.steps,
    )

    for step, (m, o) in enumerate(zip(manual, official)):
        torch.testing.assert_close(m, o, rtol=1e-5, atol=1e-7, msg=f"step {step} 参数不一致")
    if rank == 0:
        print(f"PASS: ManualDDP 与官方 DDP {args.steps} 步参数逐元素一致")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
