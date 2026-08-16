"""Benchmark: PyTorch FSDP（与 ZeRO-3 同模型同数据对比）。

训练 3 步，记录峰值显存/吞吐/loss，dump rank0 全量参数到 /tmp/fsdp_params.pt。
配合 bench_zero3.py 使用（各自 dump 后对比数值等价）。
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../10-memory/demos"))
from demo_memory import build_model  # noqa: E402


def main() -> None:
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    model = build_model(0, dev)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=False,
        device_id=torch.cuda.current_device(),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.manual_seed(42)
    data = torch.randn(4, 512, 512, device=dev)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = []
    for _ in range(3):
        loss = model(data).sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()
        opt.zero_grad()
    elapsed = time.perf_counter() - start

    peak = torch.cuda.max_memory_allocated()
    tps = 4 * 512 * 3 / elapsed
    with FSDP.summon_full_params(model):
        full = torch.cat([p.detach().flatten() for p in model.parameters()]).cpu()
    if rank == 0:
        print(f"[fsdp] peak={peak/1e9:.2f}GB tok/s={tps:.0f} loss={[round(l,2) for l in losses]}")
        dump = os.environ.get("BENCH_DUMP", "/tmp/fsdp_params.pt")
        torch.save(full, dump)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
