"""Demo: TP=2 × DP=2 组合并行（4 卡）训练 vs 单设备 loss 对照。

证明什么：
- 组合并行可用：TP（模型内张量并行）+ DP（DDP 数据并行）叠加训练，
  loss 与单设备一致；
- 进程组划分：mesh (dp, tp) 两维各管一种并行；DDP 用 1D dp 子 mesh，
  TP 用 tp 子 mesh（chapter 01 的子 mesh 切片是组合的地基）。

用法（必须用 torchrun 起 4 进程）：
    torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29760 \
        chapters/09-combined/demos/demo_3d.py --device cuda --steps 3

说明：
- TP 覆盖前两层（Linear0 Colwise + Linear2 Rowwise 配对），末层保持全量
  （输入已是 Replicate，输出也是全量，模型输出语义不变）；
- DDP 需要 1D mesh（`mesh["dp"]`），DP 组内梯度 all-reduce；
- PP 组合（TP×PP）在本版本需要 pipelining manual frontend 手动拼 stage，
  见 source-map 的说明；本 demo 聚焦 TP×DP。

在 4 卡上完整 3D（TP×PP×DP）需 ≥8 卡，本环境标注不可行。
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.nn.parallel import DistributedDataParallel


def build_model(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4)
    ).to(device)


def train_single(device: torch.device, steps: int, rank: int) -> list[float]:
    model = build_model(0, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(42)  # 所有 rank 同数据：与组合做"零额外误差"对比
    data = torch.randn(32, 16, device=device)
    target = torch.randn(32, 4, device=device)
    losses = []
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(data), target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses


def train_tp_dp(device: torch.device, steps: int, rank: int, mesh) -> list[float]:
    model = build_model(0, device)
    # 1. TP：在 tp 维上并行化前两层（Colwise+Rowwise 配对），末层全量
    parallelize_module(
        model,
        mesh["tp"],
        {"0": ColwiseParallel(), "2": RowwiseParallel()},
    )
    # 2. DP：DDP 套在 dp 维上（需要 1D 子 mesh）
    model = DistributedDataParallel(model, device_mesh=mesh["dp"])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(42)  # 所有 rank 同数据（组合正确性验证）
    data = torch.randn(32, 16, device=device)
    target = torch.randn(32, 4, device=device)
    losses = []
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(data), target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--master-port", type=int, default=29514)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 起 4 进程（参考脚本 docstring）")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = int(os.environ["RANK"])

    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))

    if rank == 0:
        print(f"torch {torch.__version__} | world_size=4 | TP=2 × DP=2")

    combined_losses = train_tp_dp(device, args.steps, rank, mesh)
    single = train_single(device, args.steps, rank)
    print(f"TP+DP loss: {[round(l, 4) for l in combined_losses]}")
    print(f"单设备 loss: {[round(l, 4) for l in single]}")
    for i, (a, b) in enumerate(zip(combined_losses, single)):
        torch.testing.assert_close(
            torch.tensor(a), torch.tensor(b), rtol=1e-4, atol=1e-5,
            msg=f"step {i} TP+DP 与单设备 loss 不一致",
        )
    if rank == 0:
        print(f"PASS: TP=2 × DP=2 组合训练 loss 与单设备一致（{args.steps} 步）")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
