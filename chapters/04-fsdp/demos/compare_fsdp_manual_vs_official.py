"""对照: 手写 FSDP（demo_fsdp_mechanism）与官方 FullyShardedDataParallel 数值等价。

证明什么：
- 同一模型、同一初始权重、同一数据下，ManualFSDP 与官方 FSDP
  （FULL_SHARD）训练同样步数，每步之后的参数向量逐元素一致。

用法（必须用 torchrun 起多进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/04-fsdp/demos/compare_fsdp_manual_vs_official.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29751 \
        chapters/04-fsdp/demos/compare_fsdp_manual_vs_official.py --device cuda

本脚本 import 官方 FullyShardedDataParallel 只用于对照；ManualFSDP 不依赖它。
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_fsdp_mechanism import ManualFSDP, build_model


def snapshot(wrapped, device: torch.device) -> torch.Tensor:
    """取全量参数快照（ManualFSDP 直接拿 flat；官方 FSDP 用 summon_full_params）。"""
    if isinstance(wrapped, ManualFSDP):
        return wrapped.full_params().cpu()
    with FSDP.summon_full_params(wrapped):
        return torch.cat([p.detach().flatten().cpu() for p in wrapped.parameters()])


def train(model, wrap_factory, device: torch.device, steps: int) -> list[torch.Tensor]:
    wrapped = wrap_factory(model)
    # ManualFSDP 的优化器必须在分片（_shard）上；官方 FSDP 在 parameters() 上
    if isinstance(wrapped, ManualFSDP):
        opt = torch.optim.AdamW([wrapped._shard], lr=1e-3)
    else:
        opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    rank = dist.get_rank()
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)

    snapshots: list[torch.Tensor] = []
    for _ in range(steps):
        loss = nn.functional.mse_loss(wrapped(data), target)
        loss.backward()
        if isinstance(wrapped, ManualFSDP):
            wrapped.step(opt)  # 更新分片 + 清梯度
        else:
            opt.step()
            opt.zero_grad()
        snapshots.append(snapshot(wrapped, device))
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29508)
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

    if args.device == "cpu":
        # 官方 FSDP 强制要求 CUDA（_check_flat_params_on_expected_device），
        # 纯 CPU 只能验证手写部分（demo_fsdp_mechanism 已覆盖）。
        if rank == 0:
            print("SKIP: 官方 FSDP 不支持纯 CPU（需 CUDA），跳过对照")
        dist.barrier()
        dist.destroy_process_group()
        return

    manual = train(
        build_model(0, device),
        lambda m: ManualFSDP(m, dist.group.WORLD),
        device,
        args.steps,
    )
    official = train(
        build_model(0, device),
        lambda m: FSDP(
            m,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=False,
            device_id=torch.cuda.current_device() if args.device == "cuda" else None,
        ),
        device,
        args.steps,
    )

    for step, (m, o) in enumerate(zip(manual, official)):
        torch.testing.assert_close(m, o, rtol=1e-5, atol=1e-7, msg=f"step {step} 参数不一致")
    if rank == 0:
        print(f"PASS: ManualFSDP 与官方 FSDP {args.steps} 步参数逐元素一致")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
