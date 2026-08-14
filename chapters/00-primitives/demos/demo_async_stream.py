"""Demo: 异步 collective 与多 stream 重叠。

证明什么：
- async_op=True 时 NCCL 排到进程组专属 ncclStream，调用立即返回；
- 用单独 CUDA stream 上的计算与通信重叠：通信期间默认 stream 继续做计算，
  最终 work.wait() 后结果与同步版一致；
- 同步版（async_op=False）则阻塞当前 stream。

用法（必须用 torchrun 起多进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/00-primitives/demos/demo_async_stream.py --device cuda

本脚本只用 torch.distributed 原语，不依赖任何官方 wrapper。CPU (gloo) 后端
没有 CUDA stream 语义，重叠部分仅在 cuda 设备上断言。
"""

import argparse
import os
import time

import torch
import torch.distributed as dist


def _barrier_then_time() -> float:
    dist.barrier()
    return time.perf_counter()


def check_async_overlap(device: torch.device) -> None:
    """异步模式下，all_reduce 排队期间默认 stream 上的计算不等 NCCL。"""
    size = 8 * 1024 * 1024  # 64 MB float32，足够让 NCCL 可见耗时
    payload = torch.full((size,), float(dist.get_rank() + 1), device=device)

    work = dist.all_reduce(payload, op=dist.ReduceOp.SUM, async_op=True)

    start = _barrier_then_time()
    # 通信进行中，默认 stream 立即执行计算（不等待 NCCL）
    x = torch.arange(size, device=device)
    y = x * 2.0 + 1.0
    busy_wait = y.sum().item()
    elapsed = time.perf_counter() - start

    work.wait()
    expected = float(dist.get_world_size() * (dist.get_world_size() + 1) / 2)
    torch.testing.assert_close(payload, torch.full((size,), expected, device=device))
    assert busy_wait > 0
    if dist.get_rank() == 0:
        print(f"async: all_reduce 排队期间计算耗时 {elapsed * 1e3:.1f} ms（未等 NCCL）")


def check_sync_blocks(device: torch.device) -> None:
    """同步模式下，all_reduce 返回前默认 stream 上的后续计算被阻塞。"""
    size = 8 * 1024 * 1024
    payload = torch.full((size,), float(dist.get_rank() + 1), device=device)

    dist.barrier()
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)  # async_op=False

    # 走到这里说明 NCCL 已排入当前 stream；后续计算在 NCCL 之后排队
    x = torch.arange(size, device=device)
    torch.testing.assert_close((x * 2.0).sum(), torch.tensor(float(size * (size - 1)), device=device))
    if dist.get_rank() == 0:
        print("sync: all_reduce 返回后当前 stream 上的计算排在 NCCL 之后")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--master-port", type=int, default=29501)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 启动多进程（参考脚本 docstring）")

    backend = "nccl" if args.device == "cuda" else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    if args.device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"torch {torch.__version__} | backend={backend} | world_size={dist.get_world_size()}")

    if args.device == "cuda":
        check_async_overlap(device)
        check_sync_blocks(device)
    else:
        tensor = torch.tensor(float(dist.get_rank() + 1), device=device)
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        torch.testing.assert_close(
            tensor,
            torch.tensor(float(dist.get_world_size() * (dist.get_world_size() + 1) / 2), device=device),
        )
        if rank == 0:
            print("cpu: 异步 work 语义正确（无 stream 重叠）")

    dist.barrier()
    if rank == 0:
        print("PASS: async/stream checks")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
