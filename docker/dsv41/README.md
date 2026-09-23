# DeepSeek V4.1 Flash / SGLang FL 0.5.18

本目录交付代码迁移、NVIDIA 镜像和 H100 整模联调。引擎基于官方 SGLang **v0.5.18**，插件基于 **dev/0.5.18**；具体提交、镜像 digest、模型 revision、补丁及文件 SHA256 见 [manifest.json](manifest.json)。DeepSeek V4.1 Flash 已在 `aiops-10-8-2-1` 的 8 张 H100 上以 TP8/EP8 启动；验证了非 PD 服务、DSpark/MTP 和 decode CUDA Graph。FlagCX 两机 PD 配置与验收见 [FLAGCX_PD.md](FLAGCX_PD.md)。

镜像默认使用 `hybrid`：block FP8 线性层、packed FP4 索引器和 V4.1 predecessor-pre mHC 进入 FlagGems；稀疏注意力、压缩缓存和 TP/EP 专家链保留 V12 厂商实现。单节点已测稳态 decode TPS；两机 PD 使用 FlagCX RoCE 状态传输和 decode Graph，D attention TP2×DP4 配置已通过 80 路同时运行的 128K 输入验收。当前 hybrid 模式的每路 decode TPS 仍低于 V12 历史结果，`vendor` 对照也未达到其每路最低 100 TPS 门槛；详见 [PD 实验报告](FLAGCX_PD.md) 和 [稳态 decode 算子 profile](PROFILE_STEADY_DECODE.md)。整模质量及更长时间稳定性仍需单独验收。

## 代码与补丁

| 补丁 | 内容 |
| --- | --- |
| `patches/0001-dsv41-functional.patch` | V4.1 配置、权重加载、Engram、压缩/索引、缓存、DSpark 和必要正确性修复 |
| `patches/0002-v12-serving.patch` | V12 block32 配置、紧凑候选、六 query 解码复用、mHC、缓存游标、图元数据和 PD 拓扑修复 |
| `patches/0003-0518-compatibility.patch` | 0.5.18 请求字段、配置默认值、TopK 依赖、缓存接口和 CPU 测试适配 |
| `patches/0004-oot-interface.patch` | `sglang.srt.layers.dsv41_ops` ABI v1，显式分发及必需插件守卫 |
| `patches/0005`–`0011` | 修复 0.5.18 的压缩状态、SWA 分配、图元数据、V4.1 processor，以及请求 KV/SWA 字段迁移 |
| `patches/0012`–`0014` | FlagCX PD 的 DSpark 与 D attention TP2×DP4 准入、连续索引器页，以及按请求窗口分配 D 的 SWA 缓存 |
| `patches/flaggems-dsv41.patch` | FlagGems K tile 缩放边界修复、FP8/BF16 线性计算、packed 索引器及 mHC |

功能回移与 V12 优化分别提交。未引入新版请求对象或整套运行时重构。SGLang 版本保持 `0.5.18`，镜像内 `/opt/sglang-fl/manifest.json` 标明补丁后的源码提交和文件哈希。源分支带入的 `benchmark/deepseek_v41_h100` 报告仅作为历史资料，其吞吐、质量和 GPU 测试结果不代表本镜像的结果。

## 实际算子覆盖

| 路径 | hybrid 后端 | 本轮验证 / 仍需验证 |
| --- | --- | --- |
| block32 E4M3 线性层 | FlagGems | 分组、空输入、scale 类型 CPU 契约；SM90 离线编译；GPU 数值通过 |
| 无原生 FP8 的量化兼容线性层 | FlagGems BF16 tile 解码 | 保留 E4M3 舍入、逐组 scale；SM80 离线编译；GPU 数值待测 |
| 显式 BF16 线性层 | FlagGems | 独立模式，取消激活量化；权重按 tile 解码；整模 BF16 未打通 |
| packed FP4 索引器 | FlagGems | E2M1 布局、紧凑候选接口、零长度契约；SM90 GPU 数值通过，修复零乘 NaN |
| 六 query 复用 | FlagGems | 保留逐次 replay 的请求 ID 判断及原形状守卫；整模 decode graph 与 1/4/8 并发通过 |
| mHC predecessor-pre | FlagGems | FP32 混合权重、前一子层 pre 契约；SM90 离线编译；融合数值误差待测 |
| 稀疏注意力 / RoPE / 压缩缓存 | vendor | 保留模型语义；其余芯片的完整算子链待补齐 |
| MoE TP/EP | FlashInfer MXFP4 | 8 卡 TP8/EP8 整模请求通过；未将单 rank 参考 MegaMoE 当成 EP 后端 |
| 通信 | 单节点 TP/EP 使用 NCCL；PD 状态使用 FlagCX | `.1`/`.3` 的 `mlx5_bond_0` RoCE 设备均 ACTIVE；跨机请求通过 |
| V4.1 DSpark PD | FlagCX，P TP8、D 可选 attention TP2×DP4 | C128/SWA-ring 状态与索引器页迁移；跨拓扑单请求、3 路并发及 128K 输入通过；80 路容量见 PD 报告 |

V4.1 的 C128/SWA-ring 状态、C2 stride 和完成状态已接入 FlagCX；D 的 attention TP2×DP4 仅在 `PD_DECODE_DPA=1` 时开放，P 保持 TP8/DP1，static-verify 与缓存布局校验仍生效。取消请求、retraction 和长时间压力仍需验证。不同厂商设备混入同一个 TP/EP 组不在本次范围。

FlagGems 的 block32 修复同时约束默认和调优配置的 K tile，避免一个 tile 跨多个 scale 组；不修改缓存中的配置字典。线性层只接受解码后的浮点 scale，`uint8 UE8M0` 与普通数值 scale 的接口明确区分。

无原生 FP8 的兼容算子用 BF16 表示已经按 E4M3 舍入的激活，并在 tile 内解码权重；显式 `bf16` 模式则改变激活和权重的舍入行为。两种模式不能互称等价。缺少其他算子时，模型准入会直接报错，不会靠 FP32 fallback 启动整模。

## 镜像构建

默认构建产物：`sglang-fl-dsv41:0.5.18`，平台 `linux/amd64`。

保持官方镜像的 Torch `2.13.0+cu130`、Triton `3.7.1`、FlashInfer `0.6.17`、sglang-kernel `0.4.6.post1`、TileLang `0.1.11`、Transformers `5.12.1`。这里不安装通用 CUDA 配方中的 FlagTree。构建依赖安装在临时 venv，不改变最终运行时的 compiler。Rust `_multimodal` 扩展从回移后的源码重编译，确认包含 `dsv41.resize_patchify`；独立 Rust API server/router 二进制不在这个 Python 服务镜像中重建。

准备与 `manifest.json` 匹配的 patched FlagGems checkout，或从上游基点应用补丁：

```bash
git init FlagGems-dsv41
git -C FlagGems-dsv41 remote add origin https://github.com/flagos-ai/FlagGems.git
git -C FlagGems-dsv41 fetch --depth=1 origin 8ea592557659491930ebf24c24392f958b29ac21
git -C FlagGems-dsv41 checkout --detach FETCH_HEAD
git -C FlagGems-dsv41 apply /absolute/path/sglang-plugin-FL/docker/dsv41/patches/flaggems-dsv41.patch
git -C FlagGems-dsv41 add src
git -C FlagGems-dsv41 commit -m 'Apply DSV4.1 migration patch'
```

提前准备官方基础镜像和 wheelhouse。wheel 文件名与 SHA256 已锁定；下载使用运行环境的网络设置，凭据不进入构建目录。

```bash
docker pull --platform=linux/amd64 lmsysorg/sglang@sha256:bde16a8447b19e89056b9eea06c72be6c02801dc89d528c9ea90c53368fd74bf
docker tag lmsysorg/sglang@sha256:bde16a8447b19e89056b9eea06c72be6c02801dc89d528c9ea90c53368fd74bf lmsysorg/sglang:v0.5.18
python -m pip download --only-binary=:all: --no-deps --dest wheelhouse \
  setuptools==76.1.0 setuptools-scm==9.2.2 setuptools-rust==1.12.0 \
  semantic-version==2.10.0 wheel==0.46.2 SQLAlchemy==2.0.48 greenlet==3.3.2
python docker/dsv41/build_image.py \
  --flaggems /absolute/path/FlagGems-dsv41 \
  --wheelhouse /absolute/path/wheelhouse \
  --output /absolute/path/new-build-context
```

构建脚本校验基础镜像 ID、源码和 wheel 哈希，以 `git archive` 打包源码，Docker 构建阶段使用 `--network=none`。输出目录必须是新目录。直接使用 Dockerfile 构建时，默认 `FROM` 也固定为不可变 digest。构建结果记录保存在镜像外，镜像内的离线编译记录由该次构建生成。

## 后端开关与验收

| 环境变量 | 含义 |
| --- | --- |
| `SGLANG_FL_DSV41_BACKEND=hybrid` | 默认迁移路径；允许固定的 vendor 路径，已接入算子优先 FlagGems |
| `...=vendor` | 原始算子对照；配合镜像默认关闭的通用 ATen/fused-op 替换 |
| `...=flagos` | 单算子严格检查；未覆盖路径直接报错，拒绝整模准入 |
| `...=reference` | 只允许有界 CPU 线性层 fixture，拒绝整模准入 |
| `SGLANG_FL_DSV41_QUANTIZATION=quantized` | 默认保留量化语义；兼容计算也显式保留 E4M3 舍入 |
| `...=bf16` | 单算子 BF16 模式；完整模型链未覆盖，整模准入拒绝 |
| `SGLANG_FL_DSV41_REQUIRED=1` | 插件初始化失败时禁止静默执行原生路径 |

镜像默认 `USE_FLAGGEMS=0`、`SGLANG_FL_OOT_ENABLED=0`，只启用本次明确接入的 DSV4.1 数值接口，通信插件照常注册。每个接口首次执行输出 `DSV4.1 op=... backend=... quantization=...`；选择完成后不捕获异常来尝试另一后端。`Dsv41Backend.snapshot()` 可读取本进程选择记录。

构建中执行依赖/源码哈希、插件 entry point、ABI、关键模块导入、Rust 符号、CPU 回归和 Triton 离线编译检查。离线编译生成 SM90/SM80 cubin；GPU 数值和整模 graph replay 另在目标机器验证。

GPU 数值测试入口默认跳过。在 `.1` 机器上显式运行 `RUN_DSV41_GPU_TESTS=1`，14 项通过；随后 TP8/EP8、DSpark 与 decode graph 整模服务通过 14 个已知答案及并发请求。非 PD 的 32K 稳态 decode TPS 已测；FlagCX PD 的首次两机请求与 3 路并发通过，详细结果见 [FLAGCX_PD.md](FLAGCX_PD.md)。H100 源分支的性能数字不作为本次验收结果。

初始非 PD 镜像构建结果见 [build-result.json](build-result.json)：89 项 CPU 测试、63 个子用例通过，2 项跳过；10 个离线编译用例通过。另用明确的 CPU device/driver doubles 检查了 FlagGems 全包导入、插件初始化及 DSV4.1 注册，未执行 GPU 操作。包含 FlagCX PD 的后续镜像构建记录见 [FLAGCX_PD.md](FLAGCX_PD.md)。

## 单节点 Graph / MTP 联调入口

[serve_single_node.sh](serve_single_node.sh) 使用 TP8/EP8、FlashInfer MXFP4 MoE、DSpark block5、static verify 和 decode CUDA Graph；默认上下文 65536、最多 16 个并发请求、静态内存比例 0.92，不启用 PD。SM90 上关闭不兼容该索引分数 stride 的 TopK v2。`MODEL_PATH` 必须指向固定 revision 的完整 checkpoint；`SERVE_PORT` 默认 31818。

先检查容器内 8 张 GPU 均可由 PyTorch 访问，并执行 `RUN_DSV41_GPU_TESTS=1 pytest -q /opt/sglang-fl/tests/dsv41/test_gpu_ops.py`。将本目录只读挂载为 `/work/scripts` 后，可在容器内运行：

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash bash /work/scripts/serve_single_node.sh
```

[smoke_service.py](smoke_service.py) 验证已启动服务的非 PD / DSPARK / Graph 配置、单请求及 4/8 并发的已知答案、连续生成和 draft token 接受情况：

```bash
python /work/scripts/smoke_service.py \
  --url http://127.0.0.1:31818 --output /work/results/service-smoke.json
```

本次 [服务结果](/public-nvme/yjwu/sglang-fl-0518-node1/smoke-service.json) 为 14/14 请求通过，`avg_spec_accept_length=4.13`。同一轮 [服务日志](/public-nvme/yjwu/sglang-fl-0518-node1/server.log) 显示 target/draft 图捕获完成，且实际解码 batch 1/8 的 `cuda graph: True`。prefill 图按该模型的 SGLang 配置关闭；这里的 Graph 验收指 decode replay。

按主线 V11 的 32K 共享前缀、1/4/16 并发、固定 512 输出 token 形状复跑的稳态 decode 结果见 [BENCHMARK_NODE1.md](BENCHMARK_NODE1.md)。单请求中位数分别为 197.5、172.9、111.4 token/s。该服务是单机非 PD，输入内容也不同，因此不将这些数值解释为相对主线双机 PD 的性能变化。

`.1` 的服务地址、运行容器快照、官方基础镜像构建 ID、源码提交及验证证据统一记录在 [deployment-state.json](/public-nvme/yjwu/sglang-fl-0518-node1/deployment-state.json)。
