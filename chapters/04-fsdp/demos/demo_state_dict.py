"""Demo: FSDP state_dict（FULL_STATE_DICT 保存/加载）。

证明什么：
- FULL_STATE_DICT：FSDP 的 state_dict == 单设备模型的 state_dict
  （分片被 all-gather 成全量）；
- 保存 + 加载后续训：与不保存的基线逐元素一致（checkpoint 语义正确）。

用法（必须用 torchrun 起 2 进程）：
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29755 \
        chapters/04-fsdp/demos/demo_state_dict.py --device cuda --steps 3
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_fsdp_mechanism import build_model  # noqa: E402


def train(
    model, device: torch.device, steps: int, rank: int, load_path: str | None
) -> tuple[dict, torch.Tensor, dict]:
    """返回 (FULL_STATE_DICT, 全量参数向量, optimizer state_dict)。

    checkpoint 语义：模型权重 + optimizer 状态都要保存/恢复，否则 Adam
    等有状态优化器的续训轨迹与不中断训练不一致。
    """
    wrapped = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=False,
        device_id=torch.cuda.current_device(),
    )
    opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)

    if load_path is not None:
        ckpt = torch.load(load_path, map_location="cpu")
        with FSDP.state_dict_type(
            wrapped, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True)
        ):
            wrapped.load_state_dict(ckpt["model"])
        opt.load_state_dict(torch.load(f"/tmp/fsdp_opt_rank{rank}.pt", map_location="cpu"))

    for _ in range(steps):
        loss = nn.functional.mse_loss(wrapped(data), target)
        loss.backward()
        opt.step()
        opt.zero_grad()

    with FSDP.state_dict_type(
        wrapped, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True)
    ):
        sd = wrapped.state_dict()
    flat = torch.cat([v.detach().flatten() for v in sd.values()])
    return sd, flat, opt.state_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--master-port", type=int, default=29513)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 起 2 进程（参考脚本 docstring）")

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = int(os.environ["RANK"])
    ckpt = "/tmp/fsdp_ckpt.pt"  # 模型全量（共享一份）
    opt_ckpt = f"/tmp/fsdp_opt_rank{rank}.pt"  # optimizer 分片（每 rank 私有！）

    # 1. 训练 2 步 + 保存 checkpoint。
    #    关键：模型权重是 FULL_STATE_DICT（全量，所有 rank 共享一份）；
    #    optimizer 状态是「分片的」（FSDP 下每 rank 只持有一部分参数的
    #    Adam state），必须每 rank 保存自己的——否则 rank1 会加载 rank0
    #    的分片状态，续训不一致。
    sd2, _, opt_sd2 = train(build_model(0, device), device, 2, rank, None)
    if rank == 0:
        torch.save({"model": sd2}, ckpt)
    torch.save(opt_sd2, opt_ckpt)
    dist.barrier()

    # 2. 加载后继续训练 1 步 vs 不保存直接训练 3 步（基线）
    _, loaded, _ = train(build_model(0, device), device, 1, rank, ckpt)
    _, baseline, _ = train(build_model(0, device), device, 3, rank, None)
    # 基线 = 2 步 + 1 步；loaded = (加载 2 步权重+opt) + 1 步 → 应该一致
    torch.testing.assert_close(loaded, baseline, rtol=1e-5, atol=1e-7, msg="加载后续训与基线不一致")

    if rank == 0:
        print(f"PASS: FULL_STATE_DICT 保存/加载后续训与不保存基线 {args.steps} 步一致")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
