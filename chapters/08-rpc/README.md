# 08-rpc — 分布式 RPC（参数服务器 / 模型级分布式）

目标：读透 `torch.distributed.rpc`——与 collective（c10d）不同，RPC 是
**点对点**的远程调用（`rpc_sync` / `rpc_async` / `remote` + RRef），适合
参数服务器、流水线/模型级分布式，以及不是"全体同步"的场景。

## 本章要回答的问题

1. `init_rpc` 做了什么？与 `init_process_group` 的关系（RPC 复用其
   rendezvous）？
2. `rpc_sync` / `rpc_async` / `remote` 三种调用的区别？
3. 参数服务器模式怎么搭？（worker 远程调用 server 的函数）

## 目录

```text
chapters/08-rpc/
├── README.md      # 本章入口（本文件）
├── source-map.md  # 源码地图
└── demos/
    └── demo_parameter_server.py  # 参数服务器模式
```

## L20 验证记录

| 演示 | 配置 | 结果 |
| --- | --- | --- |
| demo_parameter_server | 2 进程（server + trainer） | PASS（rpc_sync/rpc_async/RRef 三种调用 + 参数更新正确） |

环境：4×L20（PCIe，无 NVLink），torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`，
2026-08-15。踩坑：RPC 远程函数必须是模块级可导入（局部 lambda 不可 pickle）；
容器 hostname 解析不可靠，须显式 loopback rendezvous + `TP_SOCKET_IFNAME=lo`。

## 使用手册（本层关键坑）

- `init_rpc` 会**复用** `MASTER_ADDR/MASTER_PORT` 做 rendezvous（基于已初始化
  的进程组），也可以只初始化 RPC。
- 远程函数必须在**所有 worker 上可导入**（同名模块），且不能是本地闭包。
- `RRef` 是远程对象的引用（owner 持有对象、管理引用计数），`to_here()` 取回
  本地；`remote()` 立即返回 RRef，适合持续持有的服务对象。
- RPC 不保证与 collective 混用的顺序；同一进程里 RPC 和 c10d 的通信要
  小心死锁（避免一边等 RPC 一边等 collective）。
- 优雅退出：`rpc.shutdown()` 需所有 worker 参与。
