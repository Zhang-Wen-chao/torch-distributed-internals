"""Demo: 峰值显存实测——单卡 / DDP / 官方 FSDP1 / 手写 FSDP。

测量内容（固定模型 355M、固定 batch）：
- 各配置的峰值显存（torch.cuda.max_memory_allocated）
- 吞吐（tok/s）
- 手写 FSDP vs 官方 FSDP 的显存差距

用法：
    # 单卡（355M 可能 OOM，OOM 也记录）
    python chapters/10-memory/demos/demo_memory.py --mode single --device cuda
    # 2 卡 DDP / FSDP1 / 手写 FSDP
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29761 \
        chapters/10-memory/demos/demo_memory.py --mode ddp|fsdp|manual --device cuda
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../04-fsdp/demos"))
from demo_fsdp_mechanism import ManualFSDP  # noqa: E402


def build_model(seed: int, device: torch.device) -> nn.Module:
    """~300M 参数 MLP 风格模型（hidden=1024, layers=24, 全 Linear——
    避免 FSDP1 与 embedding 的视图 inplace 限制）。"""
    torch.manual_seed(seed)
    model = nn.Sequential()
    model.add_module("proj", nn.Linear(512, 1024))
    for i in range(24):
        model.add_module(f"layer{i}", nn.Sequential(
            nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024),
        ))
    model.add_module("head", nn.Linear(1024, 512))
    return model.to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_loop(model, opt, device: torch.device, steps: int, rank: int, mode: str) -> tuple[int, float]:
    """返回 (峰值显存 bytes, tok/s)。"""
    torch.manual_seed(42)
    seq = 512
    data = torch.randn(2, seq, 512, device=device)
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(steps):
        loss = model(data).sum()  # 全模型 forward（FSDP 生命周期要求）
        loss.backward()
        opt.step()
        opt.zero_grad()
    elapsed = time.perf_counter() - start

    peak = torch.cuda.max_memory_allocated()
    toks = 2 * seq * steps
    return peak, toks / elapsed


def run(mode: str, device: torch.device, steps: int, rank: int, world: int) -> None:
    model = build_model(0, device)
    n_params = count_params(model)

    if mode == "single":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        peak, tps = train_loop(model, opt, device, steps, rank, mode)
        print(f"[single] params={n_params/1e6:.0f}M peak={peak/1e9:.2f}GB tok/s={tps:.0f}")
        return

    if mode == "manual":
        wrapped = ManualFSDP(model, dist.group.WORLD)
        opt = torch.optim.AdamW([wrapped._shard], lr=1e-4)
        torch.manual_seed(42)
        data = torch.randn(2, 512, 512, device=device)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(steps):
            loss = wrapped(data).sum()
            loss.backward()
            wrapped.step(opt)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        tps = 2 * 512 * steps / elapsed
        print(f"[manual-fsdp] params={n_params/1e6:.0f}M peak={peak/1e9:.2f}GB tok/s={tps:.0f}")
        return

    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, process_group=dist.group.WORLD)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        peak, tps = train_loop(model, opt, device, steps, rank, mode)
        print(f"[ddp] params={n_params/1e6:.0f}M peak={peak/1e9:.2f}GB tok/s={tps:.0f}")
        return

    if mode == "fsdp":
        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                     use_orig_params=False, device_id=torch.cuda.current_device())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        peak, tps = train_loop(model, opt, device, steps, rank, mode)
        print(f"[fsdp] params={n_params/1e6:.0f}M peak={peak/1e9:.2f}GB tok/s={tps:.0f}")
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "ddp", "fsdp", "manual"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--master-port", type=int, default=29515)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    if args.mode != "single":
        dist.init_process_group(backend="nccl", init_method="env://")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    try:
        run(args.mode, device, args.steps, rank, world)
    except torch.cuda.OutOfMemoryError:
        print(f"[{args.mode}] OOM（单卡放不下 355M 全量训练）")
    finally:
        if args.mode != "single":
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
