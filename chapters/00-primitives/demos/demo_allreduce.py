"""Demo: collective 语义（all-reduce 的 SUM/PRODUCT/MIN/MAX）、子组、异步 work。

证明什么：
- all-reduce 的四种 reduce_op 语义正确（组内所有 rank 结果一致且等于全局归约值）；
- 子进程组（rank 子集）上 collective 只在组内生效；
- async_op=True 返回的 Work 对象可查询/等待，行为与同步调用等价。

用法（必须用 torchrun 起多进程）：
    torchrun --standalone --nproc_per_node=2 chapters/00-primitives/demos/demo_allreduce.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29710 \
        chapters/00-primitives/demos/demo_allreduce.py --device cuda

本脚本只用 torch.distributed 原语，不依赖任何官方 wrapper。
"""

import argparse
import os

import torch
import torch.distributed as dist


def sum_ranks(world_size: int) -> float:
    return float(world_size * (world_size + 1) / 2)


def check_reduce_ops(world_size: int, device: torch.device) -> None:
    base = torch.tensor(float(dist.get_rank() + 1), device=device)

    sum_out = base.clone()
    dist.all_reduce(sum_out, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(
        sum_out, torch.tensor(sum_ranks(world_size), device=device),
        msg="all-reduce SUM 不等于各 rank 之和",
    )

    prod_out = base.clone()
    dist.all_reduce(prod_out, op=dist.ReduceOp.PRODUCT)
    torch.testing.assert_close(
        prod_out, torch.tensor(float(torch.arange(1, world_size + 1).prod()), device=device),
        msg="all-reduce PRODUCT 不等于各 rank 之积",
    )

    min_out = base.clone()
    dist.all_reduce(min_out, op=dist.ReduceOp.MIN)
    torch.testing.assert_close(min_out, torch.tensor(1.0, device=device), msg="all-reduce MIN 不等于 1")

    max_out = base.clone()
    dist.all_reduce(max_out, op=dist.ReduceOp.MAX)
    torch.testing.assert_close(
        max_out, torch.tensor(float(world_size), device=device), msg="all-reduce MAX 不等于 world_size"
    )


def check_subgroup(rank: int, device: torch.device) -> None:
    group = dist.new_group(ranks=[0, 1])
    if rank in (0, 1):
        tensor = torch.tensor(float(rank), device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        torch.testing.assert_close(
            tensor, torch.tensor(1.0, device=device),
            msg="子组 all-reduce 结果错误（期望 rank0+rank1 = 1）",
        )
    dist.destroy_process_group(group)


def check_async_work(device: torch.device) -> None:
    tensor = torch.tensor(float(dist.get_rank() + 1), device=device)
    work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    assert isinstance(work, dist.Work), "async_op=True 应返回 Work 对象"
    work.wait()
    torch.testing.assert_close(
        tensor, torch.tensor(sum_ranks(dist.get_world_size()), device=device),
        msg="异步 all-reduce 结果错误",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 启动多进程（参考脚本 docstring）")

    backend = "nccl" if args.device == "cuda" else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(args.device)

    if rank == 0:
        print(f"torch {torch.__version__} | backend={backend} | world_size={world_size}")

    check_reduce_ops(world_size, device)
    check_async_work(device)
    if world_size >= 2:
        check_subgroup(rank, device)

    dist.barrier()
    if rank == 0:
        print("PASS: all collective checks")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
