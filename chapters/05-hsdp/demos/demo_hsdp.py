"""Demo: 手写 HSDP（分片组 × 复制组）与官方 HYBRID_SHARD 数值对照。

证明什么：
- HSDP 的梯度路径：分片组内 reduce-scatter → 复制组间 all-reduce → 均值；
- 4 rank（分片组=2 × 复制组=2）下，手写 HSDP 与官方 FSDP HYBRID_SHARD
  训练 3 步参数逐元素一致；
- 跨 rank 一致性：训练后所有 rank 参数相同。

用法（必须用 torchrun 起 4 进程）：
    torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29760 \
        chapters/05-hsdp/demos/demo_hsdp.py --device cuda

结构：mesh (2, 2)：dim0 = replicate（复制组，[0,2]/[1,3]），dim1 = shard
（分片组，[0,1]/[2,3]）。rank 在分片组内的位置决定 shard 切片。
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../04-fsdp/demos"))
from demo_fsdp_mechanism import ManualFSDP, build_model  # noqa: E402


class ManualHSDP(ManualFSDP):
    """在 ManualFSDP 基础上加复制组 all-reduce（HSDP 的关键差异）。"""

    def __init__(self, module: nn.Module, shard_group, replicate_group) -> None:
        self.replicate_group = replicate_group
        super().__init__(module, shard_group)

    def _reduce_and_store_grad(self) -> None:
        grad = self._flat.grad
        if grad is None:
            return
        self._flat.grad = None
        padded = torch.nn.functional.pad(grad, [0, self._padded_size - grad.numel()])
        dist.reduce_scatter_tensor(
            self._shard_grad, padded, op=dist.ReduceOp.SUM, group=self.process_group
        )
        # HSDP：分片梯度跨复制组 all-reduce（复制组内梯度相同）
        dist.all_reduce(self._shard_grad, op=dist.ReduceOp.SUM, group=self.replicate_group)
        self._shard_grad.div_(self.world_size * dist.get_world_size(self.replicate_group))
        if self.rank == self.world_size - 1:
            tail = self._padded_size - self._numel
            if tail > 0:
                self._shard_grad[-tail:].zero_()


def train(wrapped, device: torch.device, steps: int, rank: int) -> torch.Tensor:
    if isinstance(wrapped, ManualHSDP):
        opt = torch.optim.AdamW([wrapped._shard], lr=1e-3)
    else:
        opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)
    for _ in range(steps):
        loss = nn.functional.mse_loss(wrapped(data), target)
        loss.backward()
        if isinstance(wrapped, ManualHSDP):
            wrapped.step(opt)
        else:
            opt.step()
            opt.zero_grad()
    if isinstance(wrapped, ManualHSDP):
        return wrapped.full_params().cpu()
    with FSDP.summon_full_params(wrapped):
        return torch.cat([p.detach().flatten().cpu() for p in wrapped.parameters()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--master-port", type=int, default=29509)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 起 4 进程（参考脚本 docstring）")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = int(os.environ["RANK"])  # init_device_mesh 会懒初始化默认组

    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
    shard_group = mesh.get_group("shard")
    replicate_group = mesh.get_group("replicate")

    if rank == 0:
        print(f"torch {torch.__version__} | world_size={dist.get_world_size()} | HSDP(shard=2, replicate=2)")

    manual = train(
        ManualHSDP(build_model(0, device), shard_group, replicate_group),
        device,
        args.steps,
        rank,
    )
    official = train(
        FSDP(
            build_model(0, device),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=mesh,
            use_orig_params=False,
        ),
        device,
        args.steps,
        rank,
    )

    torch.testing.assert_close(manual, official, rtol=1e-5, atol=1e-7, msg="手写 HSDP 与官方 HYBRID_SHARD 不一致")

    # HSDP 语义：复制组内参数一致（分片组间本来就不同——各自持有不同切片）。
    # 只在复制组内做 all_gather 对比。
    # 注意：all_gather 的输入/输出都必须是 CUDA（NCCL 不校验输出 device，
    # CPU 输出会写坏内存导致假不一致）。
    manual_dev = manual.to(device)
    rep_size = dist.get_world_size(replicate_group)
    gathered = [torch.zeros_like(manual_dev) for _ in range(rep_size)]
    dist.all_gather(gathered, manual_dev, group=replicate_group)
    for other in gathered:
        torch.testing.assert_close(manual, other.cpu(), msg="复制组内参数不一致")

    dist.barrier()
    if rank == 0:
        print(f"PASS: 手写 HSDP 与官方 HYBRID_SHARD {args.steps} 步参数逐元素一致，复制组内一致")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
