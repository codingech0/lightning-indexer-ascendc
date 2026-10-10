// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LightningIndexerTilingData {
    uint64_t length;
    uint64_t batch;
    uint64_t queryLength;
    uint64_t keyLength;
    uint32_t heads;
    uint32_t sparseCount;
    uint32_t sparseMode;
    uint32_t returnValues;
    uint32_t hasQueryLengths;
    uint32_t hasKeyLengths;
    uint32_t weightsFloat32;
};
