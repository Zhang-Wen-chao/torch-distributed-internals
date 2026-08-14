"""Demo: coalescing（批量 collective）。

证明什么：
- `dist._coalescing_manager` 上下文里多次 all_reduce 在退出时合并为一次
  `allreduce_coalesced`（DDP 梯度桶同步的底层机制）；
- 合并结果与逐 tensor 分别 all_reduce 完全一致。

注意（非契约行为）：
- `_coalescing_manager` 是带下划线的内部 API（无向后兼容保证），本演示
  只为展示 DDP 的底层机制。
- 混合 reduce op（如 SUM + PRODUCT）官方声明不支持（未定义行为），且当前
  版本不保证抛错，所以本演示不对此做断言。

用法（必须用 torchrun 起多进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/00-primitives/demos/demo_coalescing.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29715 \
        chapters/00-primitives/demos/demo_coalescing.py --device cuda
"""

import argparse
import os

import torch
import torch.distributed as dist


def check_coalesced_equals_manual(device: torch.device) -> None:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    sizes = [4, 128, 1024, 7]  # 混合大小，验证每个 tensor 独立正确

    manual = [torch.full((s,), float(rank), device=device) for s in sizes]
    for t in manual:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = [torch.full((s,), float(world_size * (world_size - 1) / 2), device=device) for s in sizes]

    coalesced = [torch.full((s,), float(rank), device=device) for s in sizes]
    # gloo 没有 C++ startCoalescing：cpu 不传 device 走 Python fast path
    # （group.allreduce_coalesced，Gloo 已实现）；cuda 传 device 走
    # ncclGroupStart/End 路径（ProcessGroupNCCL::startCoalescing）。
    mgr_device = device if device.type == "cuda" else None
    with dist._coalescing_manager(device=mgr_device, async_ops=False) as cm:
        for t in coalesced:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    cm.wait()

    for m, c, e in zip(manual, coalesced, expected):
        torch.testing.assert_close(m, e)
        torch.testing.assert_close(c, e)
    if dist.get_rank() == 0:
        print("coalesced all_reduce == 逐 tensor all_reduce")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29502)
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

    check_coalesced_equals_manual(device)

    dist.barrier()
    if rank == 0:
        print("PASS: coalescing checks")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
