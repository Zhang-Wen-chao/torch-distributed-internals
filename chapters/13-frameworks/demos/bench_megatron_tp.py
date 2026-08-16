"""Benchmark: Megatron TP=2 vs PyTorch 官方 TP=2（同模型同权重同数据 loss 对照）。

用法（2 进程，系统 python——megatron-core 0.18.0 装在系统环境）：
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29820 \
        chapters/13-frameworks/demos/bench_megatron_tp.py
"""
import os
import time

import torch
import torch.nn as nn

import megatron.core.parallel_state as ps
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import TransformerConfig

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module


def build_torch_tp(device):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024)).to(device)
    mesh = init_device_mesh("cuda", (2,))
    parallelize_module(m, mesh, {"0": ColwiseParallel(), "2": RowwiseParallel()})
    return m


def main() -> None:
    import torch.distributed as dist
    dist.init_process_group(backend="nccl", init_method="env://")
    ps.initialize_model_parallel(tensor_model_parallel_size=2)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(0)
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    tp_rank = ps.get_tensor_model_parallel_rank()

    # PyTorch 官方 TP 基线（先跑）
    torch_tp = build_torch_tp(dev)
    opt1 = torch.optim.AdamW(torch_tp.parameters(), lr=1e-4)
    torch.manual_seed(42)
    data = torch.randn(4, 512, 1024, device=dev)
    t0 = time.perf_counter()
    torch_losses = []
    for _ in range(3):
        loss = torch_tp(data).mean()
        torch_losses.append(loss.item())
        loss.backward()
        opt1.step()
        opt1.zero_grad()
    torch_tps = 4 * 512 * 3 / (time.perf_counter() - t0)

    # Megatron TP：与 PyTorch 相同的全量权重（手工切分赋值）
    torch.manual_seed(0)
    w1_full = nn.Linear(1024, 4096).weight.detach().to(dev)
    w2_full = nn.Linear(4096, 1024).weight.detach().to(dev)
    cfg = TransformerConfig(
        hidden_size=1024, ffn_hidden_size=4096, num_layers=2,
        num_attention_heads=8, num_query_groups=8,
    )
    meg_col = ColumnParallelLinear(1024, 4096, bias=False, config=cfg,
                                    init_method=lambda x: x)
    meg_row = RowParallelLinear(4096, 1024, bias=False, config=cfg,
                                init_method=lambda x: x, input_is_parallel=True,
                                skip_bias_add=False)
    # ColumnParallelLinear: weight (out/2, in) = w1_full 按 out 维切
    meg_col.weight.data.copy_(w1_full[tp_rank * 2048 : (tp_rank + 1) * 2048])
    # RowParallelLinear: weight (out, in/2) = w2_full 按 in 维切
    meg_row.weight.data.copy_(w2_full[:, tp_rank * 2048 : (tp_rank + 1) * 2048])

    def meg_forward(x):
        # Megatron 的 ParallelLinear forward 恒返回 (output, bias) 元组
        x = meg_col(x)[0]
        x = torch.relu(x)
        x = meg_row(x)
        return x[0] if isinstance(x, tuple) else x

    params = list(meg_col.parameters()) + list(meg_row.parameters())
    opt2 = torch.optim.AdamW(params, lr=1e-4)
    torch.manual_seed(42)
    data2 = torch.randn(4, 512, 1024, device=dev)
    t1 = time.perf_counter()
    meg_losses = []
    for _ in range(3):
        loss = meg_forward(data2).mean()
        meg_losses.append(loss.item())
        loss.backward()
        opt2.step()
        opt2.zero_grad()
    meg_tps = 4 * 512 * 3 / (time.perf_counter() - t1)

    if rank == 0:
        print(f"[torch-tp] loss={[round(l, 6) for l in torch_losses]} tok/s={torch_tps:.0f}")
        print(f"[megatron-tp] loss={[round(l, 6) for l in meg_losses]} tok/s={meg_tps:.0f}")
        for i, (a, b) in enumerate(zip(torch_losses, meg_losses)):
            rel = abs(a - b) / max(abs(a), 1e-12)
            print(f"step{i}: torch={a:.6f} megatron={b:.6f} rel_diff={rel:.2e}")

    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
