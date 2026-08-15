"""Benchmark: all-reduce 大小 × 算法（NCCL_ALGO）耗时对比。

测量：
- 4 种消息大小（1KB / 1MB / 16MB / 256MB）
- 3 种算法候选（Ring / Tree / PatRing，通过环境变量 NCCL_ALGO 传入）
- 每组合 warmup 后测 20 次取中位数耗时，打印带宽（GB/s）

用法（4 进程）：
    torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29765 \
        chapters/11-nccl-internals/demos/bench_allreduce.py --device cuda --algo Ring
    # --algo 支持 Ring / Tree / PatRing / Auto（不设 NCCL_ALGO）

注意：算法是 NCCL 启发式候选；实际选用看 NCCL_DEBUG=INFO 日志。
"""

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist


def bench(size_bytes: int, device: torch.device, iters: int = 20) -> float:
    """返回中位耗时（秒）。"""
    n = size_bytes // 4
    t = torch.full((n,), 1.0, device=device)
    for _ in range(5):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    times = []
    for _ in range(iters):
        dist.barrier()
        start = time.perf_counter()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--algo", choices=["Ring", "Tree", "PatRing", "Auto"], default="Auto")
    parser.add_argument("--master-port", type=int, default=29516)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 起 4 进程（参考脚本 docstring）")

    if args.algo != "Auto":
        os.environ["NCCL_ALGO"] = args.algo
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    if rank == 0:
        print(f"torch {torch.__version__} | NCCL_ALGO={os.environ.get('NCCL_ALGO', 'Auto')} | 4xL20")

    sizes = [1024, 1024 * 1024, 16 * 1024 * 1024, 256 * 1024 * 1024]
    for size in sizes:
        dt = bench(size, device)
        bw = size / dt / 1e9
        if rank == 0:
            print(f"size={size/1024:>8.0f}KB 耗时={dt*1e3:8.2f}ms  带宽={bw:6.2f}GB/s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
