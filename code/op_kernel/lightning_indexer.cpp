// Diagnostic single-AIV baseline: all numerical work executes on device.
#include "kernel_operator.h"

#include "lightning_indexer_tiling.h"
#include "tiling_key_lightning_indexer.h"

template <typename T> __aicore__ inline float LiFloat(T x) { return static_cast<float>(x); }
template <> __aicore__ inline float LiFloat<bfloat16_t>(bfloat16_t x) { return AscendC::ToFloat(x); }
template <typename T> __aicore__ inline T LiOutput(float x) { return static_cast<T>(x); }
template <> __aicore__ inline bfloat16_t LiOutput<bfloat16_t>(float x) {
    // CANN 8.5.2 ToBfloat16 on 910B truncates. Round output scores to nearest,
    // ties to even; the union representation follows kernel_scalar_convert.h.
    union FloatBits {
        __aicore__ FloatBits() {}
        float value;
        uint32_t bits;
    } src;
    union BfloatBits {
        __aicore__ BfloatBits() {}
        uint16_t bits;
        bfloat16_t value;
    } dst;
    src.value = x;
    const uint32_t magnitude = src.bits & 0x7fffffffU;
    if (magnitude > 0x7f800000U) {
        // Preserve NaN rather than rounding a small payload to infinity.
        dst.bits = static_cast<uint16_t>((src.bits >> 16) | 0x0040U);
    } else if (magnitude == 0x7f800000U) {
        dst.bits = static_cast<uint16_t>(src.bits >> 16);
    } else {
        dst.bits = static_cast<uint16_t>((src.bits + 0x7fffU + ((src.bits >> 16) & 1U)) >> 16);
    }
    return dst.value;
}

template <class T>
class KernelLightningIndexer {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR qLengths,
        GM_ADDR kLengths, GM_ADDR indices, GM_ADDR values, const LightningIndexerTilingData &data) {
        t = data;
        qGm.SetGlobalBuffer((__gm__ T *)query);
        kGm.SetGlobalBuffer((__gm__ T *)key);
        if (t.weightsFloat32) w32Gm.SetGlobalBuffer((__gm__ float *)weights);
        else wGm.SetGlobalBuffer((__gm__ T *)weights);
        if (t.hasQueryLengths) qLenGm.SetGlobalBuffer((__gm__ int32_t *)qLengths);
        if (t.hasKeyLengths) kLenGm.SetGlobalBuffer((__gm__ int32_t *)kLengths);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)indices);
        if (t.returnValues) valGm.SetGlobalBuffer((__gm__ T *)values);
        pipe.InitBuffer(qBuf, 64 * 128 * sizeof(float));
        pipe.InitBuffer(kBuf, 128 * sizeof(float));
        pipe.InitBuffer(wBuf, 64 * sizeof(float));
        pipe.InitBuffer(scoreBuf, 8192 * sizeof(float));
        pipe.InitBuffer(indexBuf, 8192 * sizeof(int32_t));
        q = qBuf.Get<float>(); k = kBuf.Get<float>(); w = wBuf.Get<float>();
        scores = scoreBuf.Get<float>(); indicesLocal = indexBuf.Get<int32_t>();
    }
    __aicore__ inline void Process() {
        // One AIV owns all rows, avoiding scalar GM cache-line sharing on short rows.
        for (uint64_t b = 0; b < t.batch; ++b) {
            const int64_t qLen = t.hasQueryLengths ? static_cast<int64_t>(qLenGm.GetValue(b)) : static_cast<int64_t>(t.queryLength);
            const int64_t kLen = t.hasKeyLengths ? static_cast<int64_t>(kLenGm.GetValue(b)) : static_cast<int64_t>(t.keyLength);
            for (uint64_t s = 0; s < t.queryLength; ++s) {
                const uint64_t row = b * t.queryLength + s;
                const uint64_t out = row * t.sparseCount;
                uint32_t size = 0;
                if (static_cast<int64_t>(s) < qLen && kLen > 0) {
                    // Reuse Q/W across all keys; no QK intermediate in GM.
                    for (uint32_t h = 0; h < t.heads; ++h) {
                        const uint64_t wh = row * t.heads + h;
                        w.SetValue(h, t.weightsFloat32 ? w32Gm.GetValue(wh) : LiFloat(wGm.GetValue(wh)));
                        for (uint32_t d = 0; d < 128; ++d)
                            q.SetValue(h * 128 + d, LiFloat(qGm.GetValue(wh * 128 + d)));
                    }
                    for (int64_t j = 0; j < kLen; ++j) {
                        float agg = 0.0f;
                        // Provisional: supplied NumPy example masks to zero but
                        // keeps masked keys in Top-k. Await official clarification.
                        const bool visible = t.sparseMode == 0 ||
                            j <= kLen - static_cast<int64_t>(t.queryLength) + static_cast<int64_t>(s);
                        if (visible) {
                            for (uint32_t d = 0; d < 128; ++d)
                                k.SetValue(d, LiFloat(kGm.GetValue((b * t.keyLength + j) * 128 + d)));
                            for (uint32_t h = 0; h < t.heads; ++h) {
                                float dot = 0.0f;
                                for (uint32_t d = 0; d < 128; ++d)
                                    dot += q.GetValue(h * 128 + d) * k.GetValue(d);
                                agg += w.GetValue(h) * (dot > 0.0f ? dot : 0.0f);
                            }
                        }
                        Insert(agg, static_cast<int32_t>(j), size);
                    }
                    SortIndices(size);
                }
                for (uint32_t i = 0; i < t.sparseCount; ++i) {
                    idxGm.SetValue(out + i, i < size ? indicesLocal.GetValue(i) : -1);
                    if (t.returnValues)
                        valGm.SetValue(out + i, LiOutput<T>(i < size ? scores.GetValue(i) : 0.0f));
                }
            }
        }
        // SetValue initially writes scalar DCache. Explicitly flush all dirty
        // lines so both index and optional value buffers are visible in GM.
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(idxGm);
    }
private:
    __aicore__ inline bool Worse(float a, int32_t ai, float b, int32_t bi) {
        return a < b || (a == b && ai > bi);
    }
    __aicore__ inline bool Before(uint32_t a, uint32_t b, bool byIndex) {
        if (byIndex) return indicesLocal.GetValue(a) > indicesLocal.GetValue(b);
        return Worse(scores.GetValue(a), indicesLocal.GetValue(a), scores.GetValue(b), indicesLocal.GetValue(b));
    }
    __aicore__ inline void Swap(uint32_t a, uint32_t b) {
        float v = scores.GetValue(a); int32_t i = indicesLocal.GetValue(a);
        scores.SetValue(a, scores.GetValue(b)); indicesLocal.SetValue(a, indicesLocal.GetValue(b));
        scores.SetValue(b, v); indicesLocal.SetValue(b, i);
    }
    __aicore__ inline void Sift(uint32_t root, uint32_t size, bool byIndex) {
        while (2 * root + 1 < size) {
            uint32_t child = 2 * root + 1;
            if (child + 1 < size && Before(child + 1, child, byIndex)) ++child;
            if (!Before(child, root, byIndex)) break;
            Swap(root, child); root = child;
        }
    }
    __aicore__ inline void Insert(float score, int32_t index, uint32_t &size) {
        if (size < t.sparseCount) {
            uint32_t pos = size++;
            scores.SetValue(pos, score); indicesLocal.SetValue(pos, index);
            while (pos > 0) {
                uint32_t parent = (pos - 1) / 2;
                if (!Before(pos, parent, false)) break;
                Swap(pos, parent); pos = parent;
            }
        } else if (Worse(scores.GetValue(0), indicesLocal.GetValue(0), score, index)) {
            scores.SetValue(0, score); indicesLocal.SetValue(0, index); Sift(0, size, false);
        }
    }
    __aicore__ inline void SortIndices(uint32_t size) {
        // Rebuild a max-index heap, then heapsort pairs into ascending index order.
        for (uint32_t i = size / 2; i > 0; --i) Sift(i - 1, size, true);
        for (uint32_t end = size; end > 1; --end) { Swap(0, end - 1); Sift(0, end - 1, true); }
    }
    LightningIndexerTilingData t;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qBuf, kBuf, wBuf, scoreBuf, indexBuf;
    AscendC::LocalTensor<float> q, k, w, scores;
    AscendC::LocalTensor<int32_t> indicesLocal;
    AscendC::GlobalTensor<T> qGm, kGm, wGm, valGm;
    AscendC::GlobalTensor<float> w32Gm;
    AscendC::GlobalTensor<int32_t> qLenGm, kLenGm, idxGm;
};

template <typename DT_QUERY>
 __global__ __aicore__ void lightning_indexer(GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR actual_seq_lengths_query, GM_ADDR actual_seq_lengths_key, GM_ADDR sparse_indices, GM_ADDR sparse_values, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LightningIndexerTilingData);
    GET_TILING_DATA_WITH_STRUCT(LightningIndexerTilingData, tiling_data, tiling);
    KernelLightningIndexer<DT_QUERY> op;
    op.Init(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, sparse_indices, sparse_values, tiling_data);
    op.Process();
}
