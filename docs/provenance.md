# 来源与版本完整性

本项目整理自用户的 MhcPre 比赛工程及实验记录。默认版本对应历史归档：

`MhcPre_BR5_20260928_172726_NPU_verified.zip`

| 对象 | SHA256 |
| --- | --- |
| 原始 ZIP（未随仓库发布） | `62d92d8ada12ff37a812faacd46a17ee32495b489a817e3eabd46ee2e053dd49` |
| code/op_host/mhc_pre.cpp | `a4b7758fd814ea44924e36df201cee5e34734d1848280a8b08f6ee6c6794933a` |
| code/op_kernel/mhc_pre.cpp | `698e881b470bdaa0b8410cc4029ae9f99e2f1df7d0bd48eea893b2654c545acd` |

两个 cpp 首行均保留 `Modified: 2026-09-28 17:27:26 +08:00 | Version: BR5-batch-tail`。仓库整理不构成新 kernel 候选，CMake 文件可能仅统一文本换行。

文档是对历史记录的公开摘要。原始完整日志含远程环境路径，未直接发布；性能数字未重新测量，平台数据来自用户反馈，隐藏测试 shape 未知。当前没有提供官方 golden，也没有获取排行榜其他参赛者的私有工程。

## 参考资料

- [MhcPre 比赛](https://cannjudge.cn/public/ct_starcup_aiop_g1/mhcpre)
- [CANN cannbot-skills](https://gitcode.com/cann/cannbot-skills)：Ascend C 开发与优化方法参考。
- [cann-bench](https://gitcode.com/cann/cann-bench)：算子评测框架参考；本项目验证脚本不等于该框架，也不声称已完成其集成。
- [BR10 平台记录](https://cannjudge.cn/public/ct_starcup_aiop_g1/mhcpre/submission/6aba6f77694b590c3c0b375d)：可能需平台权限才能访问。

第三方材料权利归各自所有者；本项目未添加统一许可证，避免对原始比赛材料作未经确认的授权。
