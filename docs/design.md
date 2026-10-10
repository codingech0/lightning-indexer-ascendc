# 实现与语义边界

Host 文件 `code/op_host/lightning_indexer.cpp` 负责算子接口注册、shape/dtype 校验和 Tiling。Kernel 文件 `code/op_kernel/lightning_indexer.cpp` 执行数值计算。Tiling 数据和 key 定义位于同目录头文件。

数学基线是 score[j] = sum_h(weights[h] * max(dot(query[h], key[j]), 0))。点积与 head 聚合使用 FP32。设备侧采用单 AIV 核标量循环与 TPipe/TBuf 缓冲，使用有界堆保存候选；score 相同优先较小索引；最终把选中的索引升序排列，score 跟随对应索引。输出 padding 为 -1 / 0。

这不是 Cube/Matmul 高性能版本。缓存复用与有界堆是基线实现选择，没有 profiling 证明其相对收益。

## 暂定语义

题目文字和给出的参考思路存在需要官方澄清的细节。本实现依据开发时的参考计算作以下选择：

- mode=3 边界使用 Klen - 物理 S1 + query_position；actual query length 用于限制有效 Query 行。
- causal 遮挡位置的 score 置零，仍在有效 Klen 候选范围内。因此负权重或 k 超过可见数时，可能选到这类零分位置。这与“严格排除被遮挡位置”的文字理解不同。
- actual key length 之外的 padding 不参与 Top-k。
- BF16 输出采用最近偶数舍入；不把 FP16 参考示例直接当成 BF16 输出类型定义。

这些选择通过了自建对照测试，但未获得官方 correctness 认证。

## 测试的局限

生成用例采用便于精确比较的数值与独立全排序参考；它们覆盖了 dtype、长度、k 边界、相同分数以及堆替换等情况，但不能穷尽合法输入。尤其未证明任意浮点输入在接近 Top-k 分界时与官方矩阵运算的累加误差行为完全一致。
