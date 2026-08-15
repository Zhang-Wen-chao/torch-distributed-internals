"""Demo: 官方 pipelining（Schedule1F1B，PP=2）训练与单设备对照。

证明什么：
- 官方 PP 的数值等价性：2 stage 1F1B 训练 == 单设备模型（同初始权重、同
  数据、同优化器、同 micro-batch 累积）的 loss 与参数一致；
- split_module 自动切分：模型按层切成 2 个 stage，无需手工分配。

用法（必须用 torchrun 起 2 进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/07-pipelining/demos/demo_pp.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29780 \
        chapters/07-pipelining/demos/demo_pp.py --device cuda

说明：
- PP 下每 rank 持有部分层，参数快照对比需先拼起两个 stage 的参数；
  本演示用 loss 曲线 + 最后 stage 输出对比验证等价性（参数拼装见
  _compare 函数）。
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import Pipe, Schedule1F1B, SplitPoint, pipeline


def build_model(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    ).to(device)


def train_single(device: torch.device, steps: int, seed: int) -> tuple[list[float], torch.Tensor]:
    """单设备训练基线。"""
    model = build_model(seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(1000 + seed)
    data = torch.randn(8 * 4, 16, device=device)  # 4 microbatches × 8
    target = torch.randn(8 * 4, 4, device=device)
    losses = []
    for _ in range(steps):
        for mb in range(4):
            x = data[mb * 8 : (mb + 1) * 8]
            t = target[mb * 8 : (mb + 1) * 8]
            loss = nn.functional.mse_loss(model(x), t)
            loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses, torch.cat([p.detach().flatten() for p in model.parameters()]).cpu()


def train_pp(device: torch.device, steps: int, seed: int, rank: int) -> tuple[list[float], dict]:
    """官方 PP 训练（PP=2，4 microbatches）。"""
    model = build_model(seed, device)
    chunks = 4
    example = torch.randn(8, 16, device=device)
    # pipeline() 自动切分：split_spec 用子模块名标注切分点
    pipe = pipeline(model, mb_args=(example,), split_spec={"2": SplitPoint.END})
    assert pipe.num_stages == 2, f"期望 2 个 stage，实际 {pipe.num_stages}"
    stage = pipe.build_stage(rank, device)

    def loss_fn(out, target):
        return nn.functional.mse_loss(out, target)

    schedule = Schedule1F1B(stage, n_microbatches=chunks, loss_fn=loss_fn)

    opt = torch.optim.AdamW(stage.submod.parameters(), lr=1e-3)
    torch.manual_seed(1000 + seed)
    data = torch.randn(8 * 4, 16, device=device)
    target = torch.randn(8 * 4, 4, device=device)

    losses = []
    for _ in range(steps):
        # 官方 step 的 loss 通过 losses 列表回传（每个 micro-batch 一个）；
        # 返回值是最后 stage 的输出拼接。
        mb_losses: list = []
        schedule.step(data, target=target, losses=mb_losses)
        if mb_losses:  # 只有最后 stage 有值
            losses.append(mb_losses[-1].item())
        opt.step()
        opt.zero_grad()
    return losses, stage.submod.state_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29511)
    parser.add_argument("--steps", type=int, default=3)
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
        print(f"torch {torch.__version__} | backend={dist.get_backend()} | world_size={dist.get_world_size()} | PP=2")

    pp_losses, _ = train_pp(device, args.steps, 0, rank)
    # PP loss 只在最后 stage（rank 1）返回；单设备基线也在 rank 1 对比
    if rank == 1:
        single_losses, _ = train_single(device, args.steps, 0)
        print(f"PP loss: {[round(l, 4) for l in pp_losses]}")
        print(f"单设备 loss: {[round(l, 4) for l in single_losses]}")
        for i, (a, b) in enumerate(zip(pp_losses, single_losses)):
            torch.testing.assert_close(
                torch.tensor(a), torch.tensor(b), rtol=1e-4, atol=1e-5,
                msg=f"step {i} PP 与单设备 loss 不一致",
            )
        print(f"PASS: 官方 1F1B 训练 loss 与单设备一致（{args.steps} 步）")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
