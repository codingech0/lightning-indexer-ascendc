# 复现步骤

## 无 NPU：算法自测

在仓库根目录执行：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

这些测试在 CPU 上检查算法和参考语义，不会运行 Ascend C Kernel。

## 有 NPU：RUN 包路径

需要 Linux ARM64、可用的 Ascend 910B 设备和匹配 CANN 开发环境。先按安装位置加载 SDK 的 set_env.sh，确认 ASCEND_HOME_PATH 指向 Toolkit。SDK 不包含在本仓库中。

```bash
npu-smi info
cmake --version
cmake -S code -B build_run -DLIGHTNING_PACKAGE_TYPE=RUN
cmake --build build_run --parallel 2
find build_run -name '*.run' -print
```

选择刚生成的 RUN 安装包，先运行其 `--help` 查看本 SDK 版本支持的安装参数，再安装到独立的用户目录。安装完成后，目录中应存在 `op_api/lib/libcust_opapi.so`；不要指向另一份旧包。

```bash
export LIGHTNING_DEPLOY_DIR=/absolute/path/to/installed/package/root
bash run_npu_smoke_run.sh 0
LIGHTNING_TEST=correctness bash run_npu_smoke_run.sh 0
```

其中 0 为测试设备参数，请与可见设备配置保持一致。测试脚本自动从指定安装根目录查找 vendor 库。历史验证使用 9 个 smoke 用例和 31 个生成用例；复现须以你自己的实际输出为准。

## SHARED 已知问题

工程默认 package type 为 SHARED，保留历史配置方便研究。它曾成功编译，但官方路径和本地 SHARED 自测存在同名算子注册/选择问题。`build_local.sh` 仅构建默认配置，不是已验证的端到端复现入口。

归档时仅清理测试启动脚本的个人 SDK 绝对路径，改为要求环境变量。Host/Kernel/测试数值逻辑来自通过自测的基线快照；此次归档没有重新进行 NPU 构建和执行。
