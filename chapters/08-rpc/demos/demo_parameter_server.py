"""Demo: RPC 参数服务器模式。

证明什么：
- rpc_sync（阻塞拿到结果）、rpc_async（Future）、remote（RRef 持有远端
  对象）三种调用语义；
- 参数服务器模式：rank 0 持有全局参数/累加器，rank 1（trainer）通过
  RPC 远程更新/读取，不需要 collective。

用法（必须用 torchrun 起 2 进程；不用 --standalone，容器 hostname 解析
不可靠，需显式 loopback rendezvous）：
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29790 \
        chapters/08-rpc/demos/demo_parameter_server.py

注意：远程函数必须是「模块级可导入」的（RPC 序列化要求），不能是局部
lambda/闭包。
"""

import os

import torch
import torch.distributed.rpc as rpc


def _tensor_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


class ParameterServer:
    """远端服务对象：持有参数，支持远程更新与查询。"""

    def __init__(self) -> None:
        torch.manual_seed(42)
        self.param = torch.randn(4, 4)

    def add_grad(self, grad: torch.Tensor) -> None:
        self.param = self.param - 0.1 * grad

    def get_param(self) -> torch.Tensor:
        return self.param.clone()

    def get_step(self) -> int:
        return self._step if hasattr(self, "_step") else 0


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "演示需要 2 个进程"

    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    server_name = "worker0"

    if rank == 0:
        # 参数服务器：注册服务对象，供 trainer 通过 RRef 访问
        _ = rpc.remote(server_name, ParameterServer)
        print(f"rank0(server): 注册 ParameterServer，等待 trainer...")
    else:
        # trainer：拿远端对象的 RRef 并远程操作
        server_rref = rpc.remote(server_name, ParameterServer)
        p0 = server_rref.rpc_sync().get_param()
        print(f"rank1(trainer): 初始 param.sum={p0.sum().item():.4f}")

        # 三种调用方式
        for step in range(3):
            grad = torch.ones(4, 4) * (step + 1)
            server_rref.rpc_sync().add_grad(grad)          # 同步
            fut = server_rref.rpc_async().get_param()      # 异步 Future
            p = fut.wait()
            print(f"rank1: step{step} 后 param.sum={p.sum().item():.4f}")

        # 直接 rpc_sync 到 server 执行模块级函数（不需要先 remote）
        p2 = rpc.rpc_sync(server_name, _tensor_add, args=(p0, p0))
        print(f"rank1: 直接 rpc_sync 计算 p0+p0 的 sum={p2.sum().item():.4f}")
        print("PASS: RPC 参数服务器模式（rpc_sync/rpc_async/RRef）")

    rpc.shutdown()


if __name__ == "__main__":
    main()
