"""Demo: DeviceMesh 的组结构与切片。

证明什么：
- `init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))` 在 4 个全局
  rank 上创建两个进程组：dp 组沿 dim0 切（[0,2]、[1,3]），tp 组沿 dim1 切
  （[0,1]、[2,3]）；
- `get_group` / `get_all_groups` 返回的进程组成员正确；
- `mesh["tp"]` / `mesh["dp"]` 切片返回子 mesh，且与整维组共享同一个进程组
  （不新建）；
- 分组通信：同一 tensor 在 tp 组上 all_reduce 与 dp 组上 all_reduce 的
  归约范围不同（验证通信域隔离）。

用法（必须用 torchrun 起 4 进程）：
    torchrun --standalone --nproc_per_node=4 \
        chapters/01-device-mesh/demos/demo_device_mesh.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29720 \
        chapters/01-device-mesh/demos/demo_device_mesh.py --device cuda

本脚本只用 torch.distributed 原语 + DeviceMesh，不依赖 DTensor。
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def group_members(pg) -> list[int]:
    return sorted(dist.get_process_group_ranks(pg))


def check_group_structure(mesh, rank: int, device: torch.device) -> None:
    dp_group = mesh.get_group("dp")
    tp_group = mesh.get_group("tp")

    # (2,2) mesh: dim0 = [0,2],[1,3]; dim1 = [0,1],[2,3]
    dp_members = group_members(dp_group)
    tp_members = group_members(tp_group)

    expected_dp = [0, 2] if rank in (0, 2) else [1, 3]
    expected_tp = [0, 1] if rank in (0, 1) else [2, 3]

    assert dp_members == expected_dp, f"rank{rank} dp 组成员错误: {dp_members}"
    assert tp_members == expected_tp, f"rank{rank} tp 组成员错误: {tp_members}"

    all_groups = mesh.get_all_groups()
    assert len(all_groups) == 2
    if rank == 0:
        print(f"rank0: dp={dp_members} tp={tp_members}")


def check_slicing(mesh) -> None:
    tp_sub = mesh["tp"]
    dp_sub = mesh["dp"]
    # 切片子 mesh 与整维组共享进程组（不新建）
    assert tp_sub.get_group() is mesh.get_group("tp")
    assert dp_sub.get_group() is mesh.get_group("dp")
    if dist.get_rank() == 0:
        print("切片子 mesh 复用原进程组")


def check_communication_domains(mesh, device: torch.device) -> None:
    """同一 tensor 在不同维上 all_reduce，验证通信域隔离。"""
    rank = dist.get_rank()

    # tp 组（dim1，行内）：[0,1] 和 [2,3]，行 = rank//2，组和 = 4*(rank//2)+1
    # dp 组（dim0，列内）：[0,2] 和 [1,3]，列 = rank%2，组和 = 2*(rank%2)+2
    t = torch.tensor(float(rank), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=mesh.get_group("tp"))
    tp_sum = 4.0 * (rank // 2) + 1

    t2 = torch.tensor(float(rank), device=device)
    dist.all_reduce(t2, op=dist.ReduceOp.SUM, group=mesh.get_group("dp"))
    dp_sum = 2.0 * (rank % 2) + 2

    torch.testing.assert_close(t, torch.tensor(tp_sum, device=device))
    torch.testing.assert_close(t2, torch.tensor(dp_sum, device=device))
    if rank == 0:
        print(f"tp 归约={tp_sum}, dp 归约={dp_sum}（通信域隔离正确）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29503)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 启动多进程（参考脚本 docstring）")

    if args.device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 必设；不能用
        # os.environ.get 的默认值（会提前求值 dist.get_rank()，未初始化时抛错）
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        # cpu 路径必须显式初始化 gloo 默认组：否则 DeviceMesh 懒初始化时
        # backend=None 会被容器的 CUDA 加速器检测解析成 {cuda:nccl}，
        # 子组继承后 CPU tensor 集体操作报 "No backend type associated with
        # device type cpu"。
        dist.init_process_group(backend="gloo", init_method="env://")

    mesh = init_device_mesh(
        args.device, (2, 2), mesh_dim_names=("dp", "tp")
    )
    rank = dist.get_rank()
    if rank == 0:
        print(f"torch {torch.__version__} | device={args.device} | world_size={dist.get_world_size()}")

    check_group_structure(mesh, rank, device)
    check_slicing(mesh)
    check_communication_domains(mesh, device)

    dist.barrier()
    if rank == 0:
        print("PASS: device mesh checks")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
