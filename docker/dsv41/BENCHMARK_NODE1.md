# `.1` 单节点稳态 decode 实验

2026-09-23 在 `aiops-10-8-2-1`（8 × H100 80GB）对已启动的 DeepSeek V4.1 Flash TP8/EP8 服务运行。引擎为官方 SGLang v0.5.18 的回移代码，插件为 `dev/0.5.18` 派生分支；DSpark block5、decode CUDA Graph 开启，PD 关闭。运行入口和原始数据分别见 [benchmark_decode.py](benchmark_decode.py) 与 [结果 JSON](/public-nvme/yjwu/sglang-fl-0518-node1/decode-benchmark-v11-shape.json)。

实验沿用主线 V11 的约 33K 共享前缀、1/4/16 并发、固定 512 输出 token、`ignore_eos=true`、2 轮预热和 10 轮计时形状。原 V11 的私有输入未随报告交付，因此这里用确定性的 weighted-interval-scheduling 参考内容构造共享前缀；服务 tokenizer 计数为 32627，完整请求为 32637。输入 SHA256 和每轮数据在结果 JSON 中。温度为 0，关闭 thinking。每个请求必须以 `length` 完成且报告 512 个输出 token。

主指标与主线的流式生成速率口径一致：`(输出 token 数 - 1) / (最后一个内容块时间 - 第一个内容块时间)`。它排除 TTFT，但流式块可包含多 token，仍不是逐 token GPU 计时。

| 并发 | 测量请求 | 单请求 decode TPS 中位数 | 最低单请求 decode TPS |
| ---: | ---: | ---: | ---: |
| 1 | 10 | 197.5 | 180.3 |
| 4 | 40 | 172.9 | 161.0 |
| 16 | 160 | 111.4 | 97.6 |

全部 210 个测量请求及 42 个预热请求完成 512 token，未发现请求失败。scheduler 日志在本轮记录了 16 个同时运行的 decode 请求、`cuda graph: True` 和 DSpark 接受长度；16 并发时部分预填充阶段仍有短暂排队，所以不能把整批墙钟吞吐直接称为稳态 decode TPS。结果 JSON 保留每轮及每请求的时间、token 数和输出 SHA256。

主线 V11 固定 512-token 实验在 **两台各 8 张 H100、PD attention TP2×DP4** 上，1/4/16 并发的候选版本单请求估计 decode TPS 中位数为 223.2/246.4/201.6。这里是一台机器、非 PD、TP8/EP8，输入文本也不同；仅能对照实验形状与指标定义，不能据此计算迁移性能回归。主线 V12 的 128K 输入、80/128 并发还依赖双机 PD；当前服务限制为 65K context、16 个并发请求，未运行该组实验。
