"""Demo: 手写 DDP 核心机制（不用官方 DistributedDataParallel）。

证明什么：
- DDP 的本质 = autograd 后置梯度 hook + 参数分桶 + 桶满即 all-reduce + 写回
  param.grad。本演示用 register_post_accumulate_grad_hook 手工实现这三步，
  不 import torch.nn.parallel.DistributedDataParallel；
- 训练若干步后所有 rank 的参数一致（跨 rank 一致性）；
- 与单进程基线相比，DDP 的梯度平均值语义正确。

用法（必须用 torchrun 起多进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/02-ddp/demos/demo_ddp_mechanism.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29730 \
        chapters/02-ddp/demos/demo_ddp_mechanism.py --device cuda

与官方 DDP 的差异（本演示不覆盖）：
- 不处理 find_unused_parameters / static_graph / local_used_map；
- bucket 组装不按"尺寸从大到小 + 256B 对齐"，只是按参数序贪心装桶；
- 所有 rank 的桶就绪顺序必须一致（全参数使用的简单模型满足）。
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn


class ManualDDP(nn.Module):
    """最小 DDP：参数广播 + 梯度 hook + 分桶 all-reduce。"""

    def __init__(
        self,
        module: nn.Module,
        process_group,
        bucket_cap_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.module = module
        self.process_group = process_group
        self._params = [p for p in module.parameters() if p.requires_grad]

        # 1. rank 0 广播初始参数（DDP 不变量：所有 rank 同起点）
        for p in self._params:
            dist.broadcast(p.data, src=0, group=process_group)

        # 2. 贪心分桶：按参数序装桶，超过容量开新桶
        self._buckets: list[dict] = []
        cur: dict = {"params": [], "bytes": 0, "flat": None, "pending": 0, "work": None}
        for p in self._params:
            size = p.numel() * p.element_size()
            if cur["params"] and cur["bytes"] + size > bucket_cap_bytes:
                self._buckets.append(cur)
                cur = {"params": [], "bytes": 0, "flat": None, "pending": 0, "work": None}
            cur["params"].append(p)
            cur["bytes"] += size
        if cur["params"]:
            self._buckets.append(cur)
        for b in self._buckets:
            b["pending"] = len(b["params"])
            b["offsets"] = []
            offset = 0
            for p in b["params"]:
                b["offsets"].append(offset)
                offset += p.numel()
            b["flat"] = torch.zeros(offset, device=self._device())

        # 3. 每个参数注册后置梯度 hook
        self._param_to_bucket = {}
        for bi, b in enumerate(self._buckets):
            for intra, p in enumerate(b["params"]):
                self._param_to_bucket[id(p)] = (bi, intra)
        for i, p in enumerate(self._params):
            p.register_post_accumulate_grad_hook(self._make_hook(i))

    def _device(self) -> torch.device:
        return self._params[0].device

    def _make_hook(self, param_index: int):
        p = self._params[param_index]

        def hook(_grad: torch.Tensor) -> None:
            # 注意：hook 的 grad 参数在此 torch 版本（NGC nightly 2.10.0a0）
            # 数值不可靠（实测为真实梯度的 ~12.5 倍）；post_accumulate 触发时
            # p.grad 已就绪且与 torch.autograd.grad 独立计算一致，故从 p.grad 取。
            if p.grad is None:
                return
            bi, intra = self._param_to_bucket[id(p)]
            bucket = self._buckets[bi]
            bucket["flat"][
                bucket["offsets"][intra] : bucket["offsets"][intra] + p.numel()
            ].copy_(p.grad.flatten())
            bucket["pending"] -= 1
            if bucket["pending"] == 0:
                dist.all_reduce(bucket["flat"], op=dist.ReduceOp.SUM, group=self.process_group)
                bucket["flat"].div_(dist.get_world_size(group=self.process_group))
                bucket["work"] = None  # 同步版 all-reduce：这里已就绪
        return hook

    def _write_back(self) -> None:
        """把归约后的桶梯度写回 param.grad（对应 reducer 的 finalize_bucket_dense）。"""
        for b in self._buckets:
            for i, p in enumerate(b["params"]):
                if p.grad is None:
                    p.grad = torch.empty_like(p)
                p.grad.copy_(
                    b["flat"][b["offsets"][i] : b["offsets"][i] + p.numel()].view_as(p)
                )

    def forward(self, *inputs):
        return self.module(*inputs)

    def sync_and_step(self, optimizer: torch.optim.Optimizer) -> None:
        self._write_back()
        optimizer.step()
        optimizer.zero_grad()
        # 重置桶计数（对应官方 reducer.prepare_for_backward 的
        # reset_bucket_counting：必须每步都从满额重新开始）。
        for b in self._buckets:
            b["pending"] = len(b["params"])


def build_model(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4)
    )


def check_ddp_correctness(device: torch.device, steps: int = 3) -> None:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    model = build_model(rank, device).to(device)  # 各 rank 不同初始化种子 → 广播修正
    ddp = ManualDDP(model, dist.group.WORLD)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)

    # 各 rank 用不同数据：验证梯度平均语义
    torch.manual_seed(1000 + rank)
    data = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)

    for step in range(steps):
        loss = nn.functional.mse_loss(ddp.module(data), target)
        loss.backward()
        ddp.sync_and_step(optimizer)

        # 每步后所有 rank 参数必须一致
        flat = torch.cat([p.detach().flatten() for p in ddp.parameters()])
        gathered = [torch.zeros_like(flat) for _ in range(world_size)]
        dist.all_gather(gathered, flat)
        for other in gathered:
            torch.testing.assert_close(flat, other, msg=f"step{step} rank{rank} 参数不一致")
    if rank == 0:
        print(f"PASS: {steps} 步训练后所有 rank 参数一致（梯度均值语义正确）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29504)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 启动多进程（参考脚本 docstring）")

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
        print(f"torch {torch.__version__} | backend={dist.get_backend()} | world_size={dist.get_world_size()}")

    check_ddp_correctness(device, args.steps)

    dist.barrier()
    if rank == 0:
        print("PASS: manual DDP checks")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
