# FlagCX PD 稳态 decode 算子 profile：hybrid 与 vendor

2026-09-23，在 `.1` P / `.3` D 的 DeepSeek V4.1 Flash 服务上，比较 D 节点 `SGLANG_FL_DSV41_BACKEND=hybrid` 与 `vendor`。两次只改变 D 的后端开关；P、router、FlagCX、模型、TP8/EP8、D attention TP2×DP4、DSpark block5、decode CUDA Graph、缓存容量均相同。D 使用 8 张 H100 80GB。

负载是相同哈希的代码 prompt，80 路并发，每路恰好 131072 输入 token、固定生成 8192 token（`ignore_eos=True`）。等四个 DP 组各有 20 路运行、无预分配等待/状态传输/retraction 且 `cuda graph: True` 后，在 D 的 8 个 rank 各抓取 10 个 decode forward step；每份 trace 均有 20 次 CUDA Graph launch，W8A8 主 GEMM 和量化均执行 2300 次。Profiler 使用 CPU+GPU activity、stack 与 shape 记录，不合并 rank。P 的 prefill 未进入本次 trace。

| 算子 GPU 时间（8 rank 中位数，ms/step） | hybrid | vendor | hybrid/vendor | 差值 |
| --- | ---: | ---: | ---: | ---: |
| W8A8 激活量化 | 4.435 | 1.067 | 4.16× | +3.369 |
| W8A8 GEMM，含 vendor split-K 归约 | 20.708 | 12.771 | 1.62× | +7.937 |
| **W8A8 合计** | **25.137** | **13.836** | **1.82×** | **+11.301** |
| FP4 六 query 分组打分 | 4.769 | 5.050 | 0.94× | −0.281 |
| FP4 普通打分 | 2.298 | 1.748 | 1.31× | +0.550 |
| **FP4 打分合计** | **7.066** | **6.798** | **1.04×** | **+0.269** |
| mHC 统计投影 | 1.739 | 1.736 | 1.00× | +0.003 |
| mHC Sinkhorn 归约 | 1.215 | 0.836 | 1.45× | +0.379 |
| mHC combine + post | 0.624 | 0.619 | 1.01× | +0.005 |
| **mHC 合计** | **3.577** | **3.190** | **1.12×** | **+0.387** |

W8A8 主 GEMM 每步 230 次：FlagGems 约 90.0 µs/次，vendor 的主 GEMM 加上部分调用所需的 split-K 归约约 55.5 µs/次。量化也各 230 次/步：FlagGems Triton `_quantize` 约 19.3 µs/次，vendor 的 `per_token_group_quant_8bit_v2_kernel` 约 4.6 µs/次。FlagGems 记录了 88 次缺少 block32 调优配置、改用默认配置的告警；vendor 使用 V4.1 的 H100 形状配置与 Hopper static/split-K 路径。这些配置和内核选择与观测到的 GEMM 差值一致，但本次 trace 没有单独隔离配置参数，不能把全部差值归因于某一项。

作为未改动路径的对照，MoE CUTLASS GEMM 是 17.989 对 18.010 ms/step，稀疏注意力是 1.727 对 1.736 ms/step。全 GPU kernel 耗时相加是 75.766 对 63.607 ms/step，差 12.159 ms；其中 W8A8 的差值为 11.301 ms，约占 93%。从首个到末个 GPU kernel 的 trace 跨度是 76.794 对 63.602 ms/step。多 stream 可以重叠，所以各类 kernel 时间之和不是端到端时延的严格分解。

未开启 profiler 的相同负载中，hybrid 两轮每路稳态 decode TPS 中位数为 76.66、75.98；vendor 三轮为 97.69、98.02、97.88。vendor 每路中位数约高 28%，与 trace 的方向一致。aggregate decode TPS 的 hybrid 两轮为 5088、5077，vendor 三轮为 5941、6597、5652；它受 80 路到达与完成时间差影响更明显。本次带 profiler 的轮次分别为 hybrid 74.87 / vendor 91.31 每路中位 TPS，不用作未采样吞吐对照。

原始数据：

- hybrid 8 rank：`/public-nvme/yjwu/sglang-fl-0518-pd/node3/profiles/hybrid-80x128k8192-20260923a/`
- vendor 8 rank：`/public-nvme/yjwu/sglang-fl-0518-pd/node3/profiles/vendor-80x128k8192-20260923a/`
- 按 rank 分类后的 JSON：`/public-nvme/yjwu/sglang-fl-0518-pd/node3/profiles/decode-op-gap-20260923a.json`
- 两轮请求 receipt：`node1/pd-profile-hybrid-80x128k8192.json`、`node1/pd-profile-vendor-80x128k8192.json`；未采样基线见 `FLAGCX_PD.md`。
- `node3/decode-profile-hybrid.log`、`node3/decode-profile-vendor.log` 包含采样窗口及 backend 选择。

用本目录的 `profile_decode_ops.py` 复算表格：

```bash
python profile_decode_ops.py \
  --hybrid /public-nvme/yjwu/sglang-fl-0518-pd/node3/profiles/hybrid-80x128k8192-20260923a \
  --vendor /public-nvme/yjwu/sglang-fl-0518-pd/node3/profiles/vendor-80x128k8192-20260923a \
  --steps 10
```

下一步优先优化 FlagGems 的 H100/block32 W8A8 配置与量化内核，再复测同一负载。FP4 普通打分值得检查，但 FP4 总耗时差只有 0.27 ms/step；mHC 总差只有 0.39 ms/step。profile 窗口的 MTP accept len 有小幅变化（hybrid 约 4.6、vendor 约 4.8），因此按相同 forward step 比较内核时间，稳态 token TPS 则引用未开启 profiler 的完整请求轮次。

为启用 SGLang 的 `/start_profile`，本轮补齐了 OOT `PlatformFL` 的 PyTorch profiler activity 声明；首次抓取前旧实现因 `NotImplementedError` 使 D 调度器退出，修复后针对性测试 2 项通过，两组 trace 均成功导出。该失败轮次没有计入性能数据。
