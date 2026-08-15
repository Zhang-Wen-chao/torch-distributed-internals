"""Demo: DTensor 布局 + FSDP2（fully_shard）与 FSDP1/手写版数值等价。

证明什么：
- DTensor：Shard(0)/Replicate 布局、本地切片、redistribute（Shard→Replicate
  触发 all-gather）语义；
- FSDP2：fully_shard 后参数是 Shard(0) DTensor；训练结果与官方 FSDP1
  逐元素一致（同一套 ZeRO 语义的另一种表达）。

用法（必须用 torchrun 起 2 进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/04-fsdp/demos/demo_fsdp2.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29752 \
        chapters/04-fsdp/demos/demo_fsdp2.py --device cuda
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1, ShardingStrategy
from torch.distributed.fsdp import fully_shard as fsdp2_fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_fsdp_mechanism import ManualFSDP, build_model  # noqa: E402


def check_dtensor_basics(device: torch.device, rank: int) -> None:
    mesh = init_device_mesh(device.type, (2,))
    full = torch.arange(8, dtype=torch.float32, device=device)

    # Shard(0)：每 rank 持有 4 个元素（偶数分片）
    dt = distribute_tensor(full.clone(), mesh, [Shard(0)])
    assert isinstance(dt, DTensor)
    local = dt.to_local()
    expected = full[rank * 4 : (rank + 1) * 4]
    torch.testing.assert_close(local, expected, msg="Shard(0) 本地切片错误")

    # Replicate：每 rank 持有全量
    dr = distribute_tensor(full.clone(), mesh, [Replicate()])
    torch.testing.assert_close(dr.to_local(), full, msg="Replicate 错误")

    # redistribute：Shard(0) → Replicate（触发 all-gather）
    gathered = dt.redistribute(placements=[Replicate()])
    torch.testing.assert_close(gathered.to_local(), full, msg="redistribute(Shard→Replicate) 错误")

    # redistribute：Replicate → Shard(0)（触发 scatter）
    resharded = gathered.redistribute(placements=[Shard(0)])
    torch.testing.assert_close(resharded.to_local(), local, msg="redistribute(Replicate→Shard) 错误")

    if rank == 0:
        print("PASS: DTensor Shard/Replicate/redistribute 语义")


def train(wrapped, device: torch.device, steps: int, rank: int) -> torch.Tensor:
    if isinstance(wrapped, ManualFSDP):
        opt = torch.optim.AdamW([wrapped._shard], lr=1e-3)
    else:
        opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)
    for _ in range(steps):
        loss = nn.functional.mse_loss(wrapped(data), target)
        loss.backward()
        if isinstance(wrapped, ManualFSDP):
            wrapped.step(opt)
        else:
            opt.step()
            opt.zero_grad()
    if isinstance(wrapped, ManualFSDP):
        return wrapped.full_params().cpu()
    if isinstance(wrapped, nn.Module) and hasattr(wrapped, "fully_shard_state"):
        return wrapped.fully_shard_state().fully_shard_state_dict()  # placeholder
    with FSDP1.summon_full_params(wrapped):
        return torch.cat([p.detach().flatten().cpu() for p in wrapped.parameters()])


def fsdp2_train(device: torch.device, steps: int, rank: int) -> torch.Tensor:
    """FSDP2（fully_shard）训练，逐层分片（自底向上）。"""
    model = build_model(0, device)
    mesh = init_device_mesh(device.type, (2,))  # 1D mesh：FSDP（Shard(0)）
    # 自底向上逐层 fully_shard（每个调用一个通信组），显式传 mesh
    for layer in reversed(list(model.children())):
        fsdp2_fully_shard(layer, mesh=mesh)
    fsdp2_fully_shard(model, mesh=mesh)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(data), target)
        loss.backward()
        opt.step()
        opt.zero_grad()
    # 取全量参数：DTensor 先 redistribute 到 Replicate（触发 all-gather），
    # to_local() 即完整张量。summon_full_params 对本版本 FSDP2 有兼容问题，
    # 手动 gather 更稳。
    with torch.no_grad():
        parts = []
        for p in model.parameters():
            if isinstance(p, DTensor):
                rep = p.redistribute(placements=[Replicate()])
                parts.append(rep.to_local().flatten().cpu())
            else:
                parts.append(p.flatten().cpu())
        return torch.cat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29512)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--only-dtensor", action="store_true", help="只跑 DTensor 语义检查")
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 起 2 进程（参考脚本 docstring）")

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

    check_dtensor_basics(device, rank)
    if args.only_dtensor:
        return

    # FSDP2 vs 官方 FSDP1 vs 手写 FSDP（3 步参数对照）
    if args.device == "cuda":
        manual = train(ManualFSDP(build_model(0, device), dist.group.WORLD), device, args.steps, rank)
        official = train(
            FSDP1(
                build_model(0, device),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=False,
                device_id=torch.cuda.current_device(),
            ),
            device,
            args.steps,
            rank,
        )
        fsdp2 = fsdp2_train(device, args.steps, rank)
        torch.testing.assert_close(fsdp2, manual, rtol=1e-5, atol=1e-7, msg="FSDP2 与手写 FSDP 不一致")
        torch.testing.assert_close(fsdp2, official, rtol=1e-5, atol=1e-7, msg="FSDP2 与官方 FSDP1 不一致")
        if rank == 0:
            print(f"PASS: FSDP2(fully_shard) 与 FSDP1/手写版 {args.steps} 步参数逐元素一致")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
