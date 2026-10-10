// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MhcPreTilingData {
    uint32_t B;             // batch
    uint32_t S;             // seq
    uint32_t n;             // 超连接维度 4/6/8
    uint32_t D;             // 隐藏维度 (16 对齐)
    uint32_t nD;            // n*D
    uint32_t M;             // n*n + 2*n (phi 行数 / hMix 宽)
    uint32_t totalT;        // B*S 总 token 数
    uint32_t tokensPerCore; // 每核负责的 token 数 (ceil)
    uint32_t kChunk;        // nD 方向分块长度 (16 对齐)
    uint32_t dChunk;        // D 方向分块长度 (16 对齐)
    uint32_t hasGamma;      // gamma 是否存在
    uint32_t phiResident;   // 1=phi/gamma 常驻 UB（快路径），0=分块流式（大 nD 兜底）
    float    invNd;         // 1/nD（host 预算，避免 aicore 内 uint->float 转换）
    float    normEps;       // RmsNorm eps
    float    hcEps;         // hPre sigmoid 后 eps
};
