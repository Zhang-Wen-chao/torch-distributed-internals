"""Benchmark: DeepSpeed ZeRO-3（与 FSDP 同模型同数据对比）。

训练 3 步，记录峰值显存/吞吐/loss，dump rank0 全量参数到 /tmp/zero3_params.pt。
必须在 DeepSpeed venv 的 python 下跑（不能用系统 torchrun）：
    /tmp/mini-deepspeed-ds-venv/bin/python -m torch.distributed.run ...
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../10-memory/demos"))
from demo_memory import build_model  # noqa: E402

import deepspeed  # noqa: E402
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus  # noqa: E402


def main() -> None:
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    model = build_model(0, dev)
    engine, opt, _, _ = deepspeed.initialize(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        model_parameters=model.parameters(),
        config={
            "train_batch_size": 4,
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": False,
                "contiguous_gradients": False,
            },
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
        },
    )
    torch.manual_seed(42)
    data = torch.randn(4, 512, 512, device=dev)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = []
    for _ in range(3):
        loss = engine(data).sum()
        losses.append(loss.item())
        engine.backward(loss)
        engine.step()
    elapsed = time.perf_counter() - start

    peak = torch.cuda.max_memory_allocated()
    tps = 4 * 512 * 3 / elapsed

    # 取全量参数：ZeRO-3 静止期参数是分片的，用 GatheredParameters
    params = list(engine.module.parameters())
    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
        full = torch.cat([p.detach().flatten().cpu() for p in params])
    if rank == 0:
        print(f"[zero3] peak={peak/1e9:.2f}GB tok/s={tps:.0f} loss={[round(l,2) for l in losses]}")
        torch.save(full, "/tmp/zero3_params.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
