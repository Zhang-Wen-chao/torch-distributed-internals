"""Demo: FSDP ± activation checkpoint 峰值显存对比（复用 chapter 10 的模型）。

证明什么：
- checkpoint 用计算换显存：FSDP + checkpoint 的峰值显存低于纯 FSDP；
- 与单设备/无 checkpoint 的完整矩阵对比。

用法（必须用 torchrun 起 2 进程）：
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29766 \
        chapters/12-activation-checkpoint/demos/demo_checkpoint.py --device cuda --steps 3
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../10-memory/demos"))
from demo_memory import build_model  # noqa: E402


def make_checkpointed(model: nn.Module) -> nn.Module:
    """把 body 的 24 层包装成 checkpoint 段。

    注意：原层必须注册为包装器子模块（否则参数从模型丢失），且闭包要
    绑定当次循环的模块（避免 Python 闭包共享循环变量）。
    """
    for name, sub in list(model.named_children()):
        if name.startswith("layer"):
            def make_wrap(orig_module: nn.Module) -> nn.Module:
                class CkptWrap(nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.orig = orig_module  # 注册子模块，保住参数

                    def forward(self, x):
                        return ckpt.checkpoint(self.orig, x, use_reentrant=False)

                return CkptWrap()

            model._modules[name] = make_wrap(sub)
    return model


def run(use_ckpt: bool, device: torch.device, steps: int, rank: int) -> tuple[float, float]:
    model = build_model(0, device)
    if use_ckpt:
        model = make_checkpointed(model)
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 use_orig_params=False, device_id=torch.cuda.current_device())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.manual_seed(42)
    data = torch.randn(16, 1024, 512, device=device)  # 大 batch：激活占主导

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        loss = model(data).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    elapsed = time.perf_counter() - start
    return torch.cuda.max_memory_allocated(), 16 * 1024 * steps / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 起 2 进程（参考脚本 docstring）")

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    if rank == 0:
        print(f"torch {torch.__version__} | 2xL20 | FSDP ± activation checkpoint")

    base_peak, base_tps = run(False, device, args.steps, rank)
    ck_peak, ck_tps = run(True, device, args.steps, rank)
    if rank == 0:
        print(f"FSDP            peak={base_peak/1e9:.2f}GB tok/s={base_tps:.0f}")
        print(f"FSDP+checkpoint peak={ck_peak/1e9:.2f}GB tok/s={ck_tps:.0f}")
        print(f"省显存 {(1 - ck_peak/base_peak)*100:.1f}%，吞吐损失 {(1 - ck_tps/base_tps)*100:.1f}%")
        assert ck_peak < base_peak, "checkpoint 应降低峰值显存"
        print("PASS: activation checkpoint 降低峰值显存")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
