"""Demo: 官方 TP（ColwiseParallel + RowwiseParallel）语义验证。

证明什么：
- 官方保证「单设备语义保持」：TP 化后的模型前向输出 == 单设备模型输出
  （同一权重，同一输入）；
- Colwise 权重切分正确：本地权重 == 全局权重的 Shard(0) 切片；
- Rowwise 正确：其输入是 Shard(-1)（上一层的输出），输出全量。

用法（必须用 torchrun 起 2 进程）：
    torchrun --standalone --nproc_per_node=2 \
        chapters/06-tp/demos/demo_tp.py --device cpu
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29770 \
        chapters/06-tp/demos/demo_tp.py --device cuda
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module


def build_mlp(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--master-port", type=int, default=29510)
    args = parser.parse_args()

    if not (dist.is_available() and "RANK" in os.environ and os.environ["RANK"] != "-1"):
        raise SystemExit("必须用 torchrun 起 2 进程（参考脚本 docstring）")

    if args.device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = int(os.environ["RANK"])
    else:
        dist.init_process_group(backend="gloo", init_method="env://")
        device = torch.device("cpu")
        rank = dist.get_rank()

    tp_mesh = init_device_mesh(args.device, (2,))

    # 单设备参考模型 + TP 模型（同一权重：先建参考模型，复制权重）
    ref = build_mlp(0, device)
    tp_model = build_mlp(0, device)
    tp_model.load_state_dict(ref.state_dict())

    parallelize_module(
        tp_model,
        tp_mesh,
        {"0": ColwiseParallel(), "2": RowwiseParallel()},
    )

    torch.manual_seed(999)
    x = torch.randn(4, 16, device=device)

    ref_out = ref(x)
    tp_out = tp_model(x)

    torch.testing.assert_close(tp_out, ref_out, rtol=1e-5, atol=1e-7, msg="TP 输出与单设备不一致")

    # Colwise 权重切分验证：本地权重 == 全局 Shard(0) 切片
    w_local = tp_model[0].weight.to_local()  # DTensor → 本地分片
    w_full = ref[0].weight
    chunk = w_full.shape[0] // 2
    torch.testing.assert_close(
        w_local.detach(),
        w_full.detach()[rank * chunk : (rank + 1) * chunk],
        msg="Colwise 权重切分错误",
    )

    if rank == 0:
        print("PASS: 官方 TP 前向输出与单设备模型一致，Colwise 权重切分正确")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
