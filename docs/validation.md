# NPU 验证与复现

## 状态与环境

以下结果来自 2026-09-28 的 BR5 历史记录，不是本次发布重新测试的结果。Ascend 910B3、CANN 9.0.0、目标 ascend910b；新目录构建 binary/package 后独立安装 OPP，没有覆盖基线。

历史检查包括 aclnn workspace/launch 返回值、NPU tensor/stream、同步与设备结果读回。初始 242 组正确性通过（基础 28 + 随机/零输入 46 + 分支边界 144 + 尾段 24），边界及尾段 168 组执行后复查也通过。最终基础矩阵 28 组和 resident 20 组性能测试的初始及回放检查通过。

未证明所有隐藏输入正确，也未获得 sanitizer 验证结论。部分后续实验专用测试脚本一并收录，但不能把收录脚本等同于 BR5 已通过其全部用例。

## 复现命令

以下命令在远程 NPU 主机的仓库根目录运行。先按实际安装位置加载 CANN 的 set_env.sh，再把 `CANN_ROOT` 设置为该安装目录。新建目录避免污染基线；不包含 SSH 地址或私钥。

```bash
npu-smi info
export CANN_ROOT=/path/to/your/cann

# 每次构建使用新目录。此处未自动覆盖任何旧安装。
RUN_DIR=$(mktemp -d "$PWD/npu-run.XXXXXX")
cmake -S code -B "$RUN_DIR/build" -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT"
cmake --build "$RUN_DIR/build" --target binary -j4
cmake --build "$RUN_DIR/build" --target package -j4

# 历史环境为 aarch64；以构建实际产物名为准。
bash "$RUN_DIR/build/custom_opp_ubuntu_aarch64.run" --quiet --install-path="$RUN_DIR/opp"
source "$RUN_DIR/opp/vendors/custom/bin/set_env.bash"

set -o pipefail
python validation/matrix.py | tee "$RUN_DIR/matrix.log"
python validation/stress.py | tee "$RUN_DIR/stress.log"
python validation/boundary_sweep.py | tee "$RUN_DIR/boundary.log"
python validation/tail_cases.py | tee "$RUN_DIR/tail.log"
```

任何命令失败都应停止后续性能测量并检查日志。确认 `ASCEND_CUSTOM_OPP_PATH` 的首项确为此次独立安装的 vendor 路径：测试脚本从该路径加载 `libcust_opapi.so`。不要把旧安装执行结果记为新候选结果。逻辑设备默认为 0，可由部署环境的设备可见性配置映射。

### 正确性标准

没有官方 golden。reference 从实际 dtype 舍入后的输入计算 FP64 结果，再转换为输出 dtype；reference 在主机计算，受测算子通过 aclnn 在 NPU 执行，两者明确分离。

| 输出 | atol | rtol |
| --- | ---: | ---: |
| hIn FP16 | 2e-5 | 0.002 |
| hIn BF16 | 2e-5 | 0.016 |
| hPost / hRes | 2e-5 | 2e-5 |

所有输出须为 finite，且逐元素满足 `abs(error) <= atol + rtol * abs(reference)`。记录 max absolute / max relative / mean absolute error，初始与回放使用相同阈值。接近零的 reference 会放大相对误差，不能只看单一最大相对误差判定。

### 性能测量

正确性通过后运行：

```bash
msprof --application="python $PWD/validation/matrix.py --warmup 20 --repeat 100" --output="$RUN_DIR/profile" --aic-mode=task-based
python validation/profile_summary.py "$RUN_DIR/profile" --warmup 20 --repeat 100
```

应用路径应不含空格。每个 case 含 1 次初始执行 + 20 warmup + 100 测量；汇总器剔除前 21 次，仅读取 msprof 的 MhcPre `Task Duration(us)`，输出 min / mean / P50 / P90 / P99。若当前 msprof 的 CSV 格式不同，应核对格式再调整解析，不能改用 CPU 时间冒充 kernel 延迟。

测试结束后再次读取结果做 correctness 检查；profiler 成功不代表正确性通过。基线和候选需在同一设备环境独立安装、同参复测，避免首次加载、其他任务负载和单次波动影响结论。

## 历史 NPU Validation Report

shape 为 `[B,S,n,D]`；下表 gamma=1，gamma=0 也在基础测试矩阵内。单位 μs，20 warmup / 100 measured。

| Test | Shape | Dtype | Correctness | min / average |
| --- | --- | --- | --- | ---: |
| small | 1,1,4,16 | FP16 | PASS | 2.8800 / 3.0004 |
| small | 1,1,4,16 | BF16 | PASS | 2.9000 / 3.0144 |
| medium | 2,17,6,64 | FP16 | PASS | 10.5600 / 11.1608 |
| medium | 2,17,6,64 | BF16 | PASS | 10.5000 / 11.1884 |
| large | 2,65,8,1024 | FP16 | PASS | 257.7400 / 260.5024 |
| large | 2,65,8,1024 | BF16 | PASS | 258.8000 / 260.7986 |
| streaming | 1,97,8,256 | FP16 | PASS | 72.1400 / 73.5138 |
| streaming | 1,97,8,256 | BF16 | PASS | 72.2000 / 73.4134 |
| k-tail | 1,3,6,1008 | FP16 | PASS | 44.1000 / 44.5834 |
| k-tail | 1,3,6,1008 | BF16 | PASS | 44.1000 / 44.5832 |
| unaligned | 1,3,4,20 | FP16 | PASS | 10.6800 / 11.1662 |
| unaligned | 1,3,4,20 | BF16 | PASS | 10.7000 / 11.1614 |
| boundary | 1,1,4,1 | FP16 | PASS | 8.4800 / 8.8618 |
| boundary | 1,1,4,1 | BF16 | PASS | 8.4600 / 8.8622 |

结论：历史覆盖范围内没有正确性回退；不能声明所有 shape 性能都提升。resident 复测曾出现最高约 +1.43% 的平均延迟变化，未通过多轮统计判定为稳定回退。平台 7.60 / 5.66 μs 仍高于 3 μs；上表小 shape 的最小值不能证明平台目标达成。
