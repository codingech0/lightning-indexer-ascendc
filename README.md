# MhcPre · Ascend C

面向 Ascend 910B 的 MhcPre 自定义算子比赛工程：融合 RMS 归一化、投影、门控与输入聚合，记录从正确性验证到实机性能优化的过程。

默认源码为 **BR5-batch-tail**，来自历史 NPU 验证且平台通过的归档。不是当前所有实验中编号最大的版本，也不是通用生产级实现。

## 当前结果

| 版本 | 平台测试点 1 | 平台测试点 2 | 正确性 |
| --- | ---: | ---: | --- |
| BR2 | 7.72 μs | 5.78 μs | 2/2 PASS |
| **BR5（本仓库）** | **7.60 μs** | **5.66 μs** | **2/2 PASS，输出错误 0%** |
| BR10（未采用） | 8.50 μs | 5.82 μs | 2/2 PASS |

BR5 相对 BR2 用时下降约 1.55% / 2.08%。**平台 <3 μs 的目标尚未达到。** 平台隐藏 shape 未公开，不能用自建小 shape 的最小延迟代替平台成绩，也不能把平台的“最优用时”栏当作本工程用时。

上述是历史记录；本次仓库整理未重新执行 NPU 验证，未修改 kernel/tiling 逻辑。详情见 [验证记录](docs/validation.md) 和 [实验历程](docs/experiments.md)。

## 算子接口

设 `x` 为 `[B,S,n,D]`，`K=nD`，`M=n²+2n`。

| 输入 | Shape | Dtype |
| --- | --- | --- |
| x | B,S,n,D | FP16 / BF16 |
| phi | M,K | FP32 |
| alpha | 3 | FP32 |
| bias | M | FP32 |
| gamma（可选） | n,D | FP32 |

对每个 token，记 `x_flat=flatten(x)`，数学流程为：

```text
r    = rsqrt(mean(x²) + normEps)
z    = (flatten(x * gamma) @ phi.T) * r   # 无 gamma 时使用 x
pre  = sigmoid(alpha[0] * z[:n] + bias[:n]) + hcEps
post = 2 * sigmoid(alpha[1] * z[n:2n] + bias[n:2n])
res  = alpha[2] * z[2n:] + bias[2n:]
hIn[d] = sum_i(pre[i] * x[i,d])
```

归一化分母和 hIn 聚合使用原始 x，不使用乘 gamma 后的 x。验证中两个 eps 均为 `1e-6`。

输出：`hIn[B,S,D]` 与输入同 dtype，`hPost[B,S,n]`、`hRes[B,S,n,n]` 为 FP32。

## 实现要点与范围

- 按 token 分配核心，FP32 中间计算与累加。
- 根据对齐和 UB 预算选择权重驻留或分块流式路径。
- BR5 在批量行点积中支持完整 64 元素段加有效尾段 mask，减少尾段场景的重复处理。
- 流式路径复用多个 token 的 phi/gamma 数据。
- 小缓冲区有 `n <= 8` 的实现假设；主要验证 n=4/6/8，另有 n=2 健壮性用例。不要视为任意 shape 支持承诺。

## 项目结构

```text
code/                 BR5 host、tiling、kernel 和 CMake 工程
validation/           NPU 调用、FP64 reference、测试矩阵和 profiler 汇总
docs/validation.md    环境、复现命令、误差规则和历史实机结果
docs/experiments.md   优化尝试、收益与回退
docs/provenance.md    归档校验与参考资料
```

## 在 NPU 环境复现

历史环境为 Ascend 910B3 + CANN 9.0.0，编译目标 `ascend910b`。需配套驱动、固件、CMake、编译工具链、PyTorch/torch_npu 和 msprof；依赖版本须与实际 CANN 环境兼容，本项目不提供自动安装脚本。

所有构建、正确性测试和 benchmark 都应在**远程 NPU 主机**执行，不以本地 CPU 测试替代。

1. 上传或克隆本仓库到 NPU 主机，加载 CANN 环境。
2. 新建构建及独立 OPP 安装目录，不覆盖已验证的基线。
3. 按 [复现指南](docs/validation.md#复现命令) 构建、安装、执行多 shape / 双 dtype 正确性测试。
4. 正确性通过后进行 warmup、msprof 测量并比较基线。

## 后续修改规则

每个新候选都在两个 `mhc_pre.cpp` 首行写入同一版本号及实际修改时间（Asia/Shanghai，UTC+08:00），即使仅一侧逻辑变化。先编译，再 NPU 正确性，再 benchmark；失败或回退必须记录。保留独立基线，未实机验证不得称为完成或建议提交。

历史源码时间戳保持原样，不因仓库整理重打版本号。

## 来源

[比赛页面](https://cannjudge.cn/public/ct_starcup_aiop_g1/mhcpre) · [实验与来源说明](docs/provenance.md)

本仓库暂未指定开源许可证；公开可见不等于授予任意使用或再分发权限。比赛材料与第三方组件的使用仍受各自规则约束。
