# Lightning Indexer · Ascend C 开发实践

中国电信星辰杯高校 AI 算子开发挑战赛 C 组简单题的学习与工程复盘。项目记录了一个初学者在 AI 辅助下，从算子公式、Host/Tiling/Kernel 到真实 NPU 自测的开发过程。

**当前定位：正确性基线与未完成项目的技术复盘。** 历史云端 RUN 部署通过 40 个自建 NPU 用例；记录中的一次官方提交 8/8 Runtime Error。SHARED 加载问题未闭环，未完成性能优化，也没有有效的官方性能成绩。

## 算子做什么

对每个 Query token，计算它与各 Key 的相关性，再挑出 Top-k 位置：

```text
QKᵀ → ReLU → 按 Query head 乘权重 → 对 head 求和
    → 长度与 causal 规则 → 稳定 Top-k → 选中索引升序输出
```

输入布局为 BSND，Key head 数为 1，head dimension 为 128。支持的实现范围包括 FP16/BF16 输入、同类型或 FP32 权重、可选有效长度、mode 0/3 和可选 score 输出。详细语义及未确认部分见 [实现说明](docs/design.md)。

## 做了什么

- 在赛事模板上补全 Host 元数据校验、Tiling 和设备端计算。
- 实现单 AIV 核基线：缓存 Q/K/权重，逐项计算 FP32 score，用有界堆筛选 Top-k，再按索引排序写回 GM。
- 修复 BF16 输出舍入问题，补充正负中点回归用例。
- 建立 9 个 NPU smoke 用例和 31 个生成/边界用例；CPU 参考计算仅用于自测，不参与算子输出。
- 定位 SHARED 路径使用系统同名 Tiling、参数数量不匹配等运行问题，并保留失败结论。

## 验证状态

| 验证层级 | 结果 | 边界 |
| --- | --- | --- |
| Python 算法测试 | 2026-10-10 归档时 18/18 通过 | 不是 NPU 执行 |
| 云端 RUN 自测 | 历史记录中 9 + 31 通过 | 自建用例、暂定语义；不能代替官方验证 |
| 官方提交 | 记录中的一次提交 0/8，Runtime Error | GetWorkspaceSize 阶段失败，未获得有效耗时 |
| SHARED 修复实验 | 未闭环 | 未证明自动加载正确 |
| 性能优化 | 未完成 | 无加速比、无 Cube/多核优化结果 |

环境记录：Ascend 910B3 / Linux ARM64 / CANN 8.5.2 / GCC 9.4 / CMake 3.20.5。这里的 NPU 结果来自开发期间回传的终端记录，归档时没有重新连接 NPU。

## 目录

```text
code/
  op_host/       算子注册、Shape/dtype 校验、Tiling
  op_kernel/     NPU 计算、Tiling 数据结构与分派 key
  CMakeLists.txt RUN / SHARED 构建配置
tests/           Python 参考对照、NPU smoke 与扩展测试
run_npu_smoke_run.sh 运行已安装 RUN 包的测试
build_local.sh   历史默认构建入口（SHARED，存在已知问题）
docs/            实现说明、复现步骤、复盘及证据边界
```

建议从 [复现步骤](docs/reproduce.md) 入手，使用显式 `RUN` 配置，不要把默认 SHARED 构建成功理解为能够运行。

## 这段经历的收获

最重要的教训是：编译通过、云端自测通过、官方提交通过，是三个不同的验收点。后期花了较多时间排查加载与注册问题，没有进入有效的性能优化阶段。复盘见 [开发总结](docs/retrospective.md)。

## 来源与致谢

工程起点是赛事提供的空模板，代码实现、测试及文档在 AI 辅助下迭代完成。外部学习资料包括 CANN 官方文档与 cann-learning-hub。未将 SDK、完整学习仓库、赛事网页抓取或个人原始日志打包进本项目。

- [赛事题目入口](https://cannjudge.cn/public/ct_starcup_aiop_g3/lightningindexer/)
- [CANN 学习中心](https://atomgit.com/cann/cann-learning-hub)
- [Ascend C 文档](https://asc.gitcode.com/)

来源与授权边界见 [NOTICE](NOTICE.md)。本项目不代表赛事官方实现。
