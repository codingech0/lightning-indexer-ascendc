// Modified: 2026-09-28 17:27:26 +08:00 | Version: BR5-batch-tail
// Kernel侧核函数实现（V3.2 Safe-Vector：基于已 Pass 的纯向量版做 token-tile/phi 复用优化）
//   逐 token：ss=Σx² -> invr -> xg=x⊙gamma -> mix=xg@phi行 -> w=mix*invr
//   -> split -> hPre=σ(pPre*a0+bias0)+hcEps / hPost=2σ(pPost*a1+bias1) / hRes=pRes*a2+bias2
//   -> hIn=Σ_i hPre_i·x[i]
// 全部数值计算在 device。phiResident 快路径保持原路径；非 resident 以 3-token tile 复用 phi/gamma K 分块。
// 32B 对齐纪律：向量运算一律在 offset-0 buffer；归约写 32B 跨步槽 red[(m+1)*8]；mix/ss 标量栈累加。
#include "kernel_operator.h"

#include "mhc_pre_tiling.h"
#include "tiling_key_mhc_pre.h"

using namespace AscendC;

namespace {
constexpr int32_t BUF_NUM = 1;
constexpr int32_t SMALL = 128;
// 非 resident 路径一次处理 3 个 token，phi 每个 K 分块只搬一次。
// 每个 token 最多需要 (1 + M)=81 个 ReduceSum 槽位；每槽按 8 个 float(32B) 对齐。
constexpr uint32_t TOKEN_TILE = 3;
constexpr uint32_t RED_SLOT_STRIDE = 96;
constexpr uint32_t RED_FLOAT_STRIDE = RED_SLOT_STRIDE * 8;
constexpr uint32_t RED_SLOTS_SINGLE = 128;
constexpr uint32_t RED_SLOTS_TILE = TOKEN_TILE * RED_SLOT_STRIDE;
__aicore__ inline uint32_t AlignBytes(uint32_t bytes) { return (bytes + 31u) & ~31u; }
}

template <class DT_X>
class KernelMhcPre {
public:
    __aicore__ inline KernelMhcPre() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                const MhcPreTilingData &tl) {
        n_ = tl.n; D_ = tl.D; nD_ = tl.nD; M_ = tl.M; totalT_ = tl.totalT;
        kChunk_ = tl.kChunk; dChunk_ = tl.dChunk;
        hasGamma_ = tl.hasGamma; phiResident_ = tl.phiResident;
        batchRows_ = phiResident_ && M_ % 8u == 0u && nD_ % 8u == 0u && nD_ / 8u <= 255u &&
                     8u + ((nD_ + 63u) / 64u) * M_ <= RED_SLOTS_SINGLE * 8u;
        invNd_ = tl.invNd; normEps_ = tl.normEps; hcEps_ = tl.hcEps;

        uint32_t blk = GetBlockIdx();
        tStart_ = blk * tl.tokensPerCore;
        tEnd_ = tStart_ + tl.tokensPerCore;
        if (tEnd_ > totalT_) { tEnd_ = totalT_; }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        phiGm_.SetGlobalBuffer((__gm__ float *)phi);
        alphaGm_.SetGlobalBuffer((__gm__ float *)alpha);
        biasGm_.SetGlobalBuffer((__gm__ float *)bias);
        if (hasGamma_) { gammaGm_.SetGlobalBuffer((__gm__ float *)gamma); }
        hInGm_.SetGlobalBuffer((__gm__ DT_X *)hIn);
        hPostGm_.SetGlobalBuffer((__gm__ float *)hPost);
        hResGm_.SetGlobalBuffer((__gm__ float *)hRes);

        pipe_.InitBuffer(inQueX_, BUF_NUM, AlignBytes(kChunk_ * sizeof(DT_X)));
        pipe_.InitBuffer(outQueHin_, BUF_NUM, AlignBytes(dChunk_ * sizeof(DT_X)));
        pipe_.InitBuffer(outSmallQue_, BUF_NUM, SMALL * sizeof(float));
        if (phiResident_) {
            pipe_.InitBuffer(phiBuf_, AlignBytes(M_ * nD_ * sizeof(float)));
            pipe_.InitBuffer(gatherIndexBuf_, 64u * sizeof(uint32_t));
            if (hasGamma_) { pipe_.InitBuffer(gammaBuf_, AlignBytes(nD_ * sizeof(float))); }
        } else {
            // 流式路径：phi 每个 K 分块只搬一次，复用于最多 3 个 token。
            pipe_.InitBuffer(inQuePhi_, BUF_NUM, AlignBytes(kChunk_ * sizeof(float)));
            pipe_.InitBuffer(xgTileBuf_, AlignBytes(TOKEN_TILE * kChunk_ * sizeof(float)));
        }
        pipe_.InitBuffer(xfBuf_, AlignBytes(kChunk_ * sizeof(float)));
        // resident: xg；streaming: gamma 当前 K 分块暂存。保持 baseline 一样总是分配，减少模板分支风险。
        pipe_.InitBuffer(xgBuf_, AlignBytes(kChunk_ * sizeof(float)));
        uint32_t tmpCount = batchRows_ && M_ * 64u > kChunk_ ? M_ * 64u : kChunk_;
        pipe_.InitBuffer(tmpBuf_, AlignBytes(tmpCount * sizeof(float)));
        pipe_.InitBuffer(workBuf_, AlignBytes(kChunk_ * sizeof(float)));
        pipe_.InitBuffer(accBuf_, AlignBytes(dChunk_ * sizeof(float)));
        pipe_.InitBuffer(redBuf_,
                         (phiResident_ ? RED_SLOTS_SINGLE : RED_SLOTS_TILE) * 8 * sizeof(float));
        pipe_.InitBuffer(segBuf_, SMALL * sizeof(float));
        pipe_.InitBuffer(sBuf_, 32 * sizeof(float));
        pipe_.InitBuffer(alphaBuf_, 32 * sizeof(float));
        pipe_.InitBuffer(biasPreBuf_, SMALL * sizeof(float));
        pipe_.InitBuffer(biasPostBuf_, SMALL * sizeof(float));
        pipe_.InitBuffer(biasResBuf_, SMALL * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (tStart_ >= tEnd_) { return; }

        alphaT_ = alphaBuf_.Get<float>();
        // CANN's FP32 sigmoid uses Div(1, x), not the approximate vrec.
        // sBuf[8:16] is a separate 32-byte-aligned constant lane for n <= 8.
        Duplicate(sBuf_.Get<float>()[8], 1.0f, 8);
        biasPreT_ = biasPreBuf_.Get<float>();
        biasPostT_ = biasPostBuf_.Get<float>();
        biasResT_ = biasResBuf_.Get<float>();
        LoadFloat(alphaT_, alphaGm_, 0, 3);
        LoadFloat(biasPreT_, biasGm_, 0, n_);
        LoadFloat(biasPostT_, biasGm_, n_, n_);
        LoadFloat(biasResT_, biasGm_, 2 * n_, n_ * n_);
        if (phiResident_) {
            auto offsets = gatherIndexBuf_.Get<int32_t>();
            CreateVecIndex(offsets, static_cast<int32_t>(0), 64);
            Muls(offsets, offsets, static_cast<int32_t>(batchRows_ ? 4 : 32), 64);
            phiT_ = phiBuf_.Get<float>();
            DataCopyExtParams phiParams{1, static_cast<uint32_t>(M_ * nD_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> phiPad{false, 0, 0, 0};
            DataCopyPad(phiT_, phiGm_, phiParams, phiPad);
            if (hasGamma_) {
                gammaT_ = gammaBuf_.Get<float>();
                DataCopyExtParams gammaParams{1, static_cast<uint32_t>(nD_ * sizeof(float)), 0, 0, 0};
                DataCopyPad(gammaT_, gammaGm_, gammaParams, phiPad);
            }
        }
        SyncMTE2ToS();
        SyncMTE2ToV();
        float a0 = alphaT_.GetValue(0);
        float a1 = alphaT_.GetValue(1);
        float a2 = alphaT_.GetValue(2);

        if (phiResident_) {
            for (uint32_t t = tStart_; t < tEnd_; ++t) { ProcessResidentToken(t, a0, a1, a2); }
        } else {
            for (uint32_t t = tStart_; t < tEnd_; t += TOKEN_TILE) {
                uint32_t valid = tEnd_ - t;
                if (valid > TOKEN_TILE) { valid = TOKEN_TILE; }
                ProcessTokenTile(t, valid, a0, a1, a2);
            }
        }
    }

private:
    __aicore__ inline void LoadFloat(const LocalTensor<float> &dst,
                                     const GlobalTensor<float> &src, uint64_t srcOff, uint32_t cnt) {
        DataCopyExtParams p{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(dst, src[srcOff], p, pad);
    }
    __aicore__ inline void SyncMTE2ToS() { SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0); }
    __aicore__ inline void SyncMTE2ToV() { SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0); }
    __aicore__ inline void SyncVToS() { SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0); }
    __aicore__ inline void SyncSToV() { SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0); }
    __aicore__ inline void SigmoidInplace(const LocalTensor<float> &v, uint32_t cnt) {
        Muls(v, v, static_cast<float>(-1.0), cnt);
        Exp(v, v, cnt);
        Adds(v, v, static_cast<float>(1.0), cnt);
        Div(v, sBuf_.Get<float>()[8], v, cnt);
    }

    // 大 nD 流式路径：以 3-token 小块重排循环。
    // baseline 是 token -> K -> M，因此同一份 phi 会被每个 token 重读。
    // 这里改为 tile -> K -> M -> token，使一个 phi[K] 分块复用于 tile 内所有 token，
    // 不改变每个 token 内 K 分块的归约顺序，尽量保持与已通过版本一致的数值结果。
    __aicore__ inline void ProcessTokenTile(uint32_t t0, uint32_t valid,
                                            float a0, float a1, float a2) {
        LocalTensor<float> xf = xfBuf_.Get<float>();
        LocalTensor<float> gammaChunk = xgBuf_.Get<float>();
        LocalTensor<float> xgTile = xgTileBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> work = workBuf_.Get<float>();
        LocalTensor<float> red = redBuf_.Get<float>();

        float ssAcc[TOKEN_TILE] = {0.0f, 0.0f, 0.0f};
        float mixAcc[TOKEN_TILE * 80];
        for (uint32_t q = 0; q < valid; ++q) {
            for (uint32_t m = 0; m < M_; ++m) { mixAcc[q * 80 + m] = 0.0f; }
        }

        uint32_t off = 0;
        while (off < nD_) {
            uint32_t cnt = (nD_ - off) < kChunk_ ? (nD_ - off) : kChunk_;

            // gamma 对所有 token 相同：每个 K 分块只从 GM 读取一次。
            if (hasGamma_) {
                DataCopyExtParams p{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                DataCopyPad(gammaChunk, gammaGm_[off], p, pad);
                SyncMTE2ToV();
            }

            // 先把 tile 内每个 token 的 xGamma 放到 UB；同时计算各自 RMS 的平方和。
            for (uint32_t q = 0; q < valid; ++q) {
                uint32_t t = t0 + q;
                LocalTensor<DT_X> xin = inQueX_.AllocTensor<DT_X>();
                DataCopyExtParams xp{1, static_cast<uint32_t>(cnt * sizeof(DT_X)), 0, 0, 0};
                DataCopyPadExtParams<DT_X> xpad{false, 0, 0, 0};
                DataCopyPad(xin, xGm_[static_cast<uint64_t>(t) * nD_ + off], xp, xpad);
                inQueX_.EnQue(xin);
                LocalTensor<DT_X> xin2 = inQueX_.DeQue<DT_X>();
                Cast(xf, xin2, RoundMode::CAST_NONE, cnt);
                inQueX_.FreeTensor(xin2);

                uint32_t rbase = q * RED_FLOAT_STRIDE;
                Mul(tmp, xf, xf, cnt);
                ReduceSum<float>(red[rbase], tmp, work, cnt);

                LocalTensor<float> dst = xgTile[q * kChunk_];
                if (hasGamma_) {
                    Mul(dst, xf, gammaChunk, cnt);
                } else {
                    // xGamma=x：本地拷贝比 Adds(x, 0) 少一次向量算术。
                    Adds(dst, xf, static_cast<float>(0.0), cnt);
                }
            }

            // 关键优化：每一行 phi 当前 K 分块只 DMA 一次，然后复用于 valid 个 token。
            for (uint32_t m = 0; m < M_; ++m) {
                LocalTensor<float> pin = inQuePhi_.AllocTensor<float>();
                DataCopyExtParams pp{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> ppad{false, 0, 0, 0};
                DataCopyPad(pin, phiGm_[static_cast<uint64_t>(m) * nD_ + off], pp, ppad);
                inQuePhi_.EnQue(pin);
                LocalTensor<float> pin2 = inQuePhi_.DeQue<float>();
                for (uint32_t q = 0; q < valid; ++q) {
                    uint32_t rbase = q * RED_FLOAT_STRIDE;
                    Mul(tmp, xgTile[q * kChunk_], pin2, cnt);
                    ReduceSum<float>(red[rbase + (m + 1) * 8], tmp, work, cnt);
                }
                inQuePhi_.FreeTensor(pin2);
            }

            // 所有 reduction 完成后只做一次 V->S 同步；这里只读结果，不再做无用的 S->V 回同步。
            SyncVToS();
            for (uint32_t q = 0; q < valid; ++q) {
                uint32_t rbase = q * RED_FLOAT_STRIDE;
                ssAcc[q] += red.GetValue(rbase);
                for (uint32_t m = 0; m < M_; ++m) {
                    mixAcc[q * 80 + m] += red.GetValue(rbase + (m + 1) * 8);
                }
            }
            off += cnt;
        }

        // 后处理仍逐 token，保持已通过版本的公式/舍入路径；hIn 使用原始 x。
        for (uint32_t q = 0; q < valid; ++q) {
            uint32_t t = t0 + q;
            float hpre[8];
            FinalizeMix(t, a0, a1, a2, ssAcc[q], &mixAcc[q * 80], hpre);
            ComputeHin(t, hpre);
        }
    }

    __aicore__ inline void ComputeHin(uint32_t t, const float *hpre) {
        LocalTensor<float> xf = xfBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> acc = accBuf_.Get<float>();

        uint32_t doff = 0;
        while (doff < D_) {
            uint32_t dcnt = (D_ - doff) < dChunk_ ? (D_ - doff) : dChunk_;

            // i=0 直接写 acc，省掉 Duplicate(0) 和第一轮 Add。
            LocalTensor<DT_X> xin0 = inQueX_.AllocTensor<DT_X>();
            DataCopyExtParams xp0{1, static_cast<uint32_t>(dcnt * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> xpad0{false, 0, 0, 0};
            DataCopyPad(xin0, xGm_[static_cast<uint64_t>(t) * nD_ + doff], xp0, xpad0);
            inQueX_.EnQue(xin0);
            LocalTensor<DT_X> xin02 = inQueX_.DeQue<DT_X>();
            Cast(xf, xin02, RoundMode::CAST_NONE, dcnt);
            inQueX_.FreeTensor(xin02);
            Muls(acc, xf, hpre[0], dcnt);

            for (uint32_t i = 1; i < n_; ++i) {
                LocalTensor<DT_X> xin = inQueX_.AllocTensor<DT_X>();
                DataCopyExtParams xp{1, static_cast<uint32_t>(dcnt * sizeof(DT_X)), 0, 0, 0};
                DataCopyPadExtParams<DT_X> xpad{false, 0, 0, 0};
                DataCopyPad(xin,
                            xGm_[static_cast<uint64_t>(t) * nD_ + static_cast<uint64_t>(i) * D_ + doff],
                            xp, xpad);
                inQueX_.EnQue(xin);
                LocalTensor<DT_X> xin2 = inQueX_.DeQue<DT_X>();
                Cast(xf, xin2, RoundMode::CAST_NONE, dcnt);
                inQueX_.FreeTensor(xin2);
                Muls(tmp, xf, hpre[i], dcnt);
                Add(acc, acc, tmp, dcnt);
            }

            LocalTensor<DT_X> hout = outQueHin_.AllocTensor<DT_X>();
            Cast(hout, acc, RoundMode::CAST_RINT, dcnt);
            outQueHin_.EnQue(hout);
            LocalTensor<DT_X> hout2 = outQueHin_.DeQue<DT_X>();
            DataCopyExtParams pw{1, static_cast<uint32_t>(dcnt * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(hInGm_[static_cast<uint64_t>(t) * D_ + doff], hout2, pw);
            outQueHin_.FreeTensor(hout2);
            doff += dcnt;
        }
    }

    __aicore__ inline void FinalizeMix(uint32_t t, float a0, float a1, float a2,
                                       float ssAcc, const float *mixAcc, float *hpre) {
        LocalTensor<float> pre = segBuf_.Get<float>();
        LocalTensor<float> sb = sBuf_.Get<float>();
        uint32_t nn = n_ * n_;

        sb.SetValue(0, ssAcc * invNd_ + normEps_);
        SyncSToV();
        Sqrt(sb, sb, 1);
        Div(sb, sBuf_.Get<float>()[8], sb, 1);
        SyncVToS();
        float invr = sb.GetValue(0);

        // hPost 与 hRes 直接在同一个 VECOUT tensor 中计算，避免 baseline 的
        // post/res -> Adds(copy) -> outQueue 两次额外向量拷贝。
        LocalTensor<float> out = outSmallQue_.AllocTensor<float>();
        LocalTensor<float> post = out;
        LocalTensor<float> res = out[16];  // 64B 对齐，8 + 64 个 float 均可容纳

        for (uint32_t i = 0; i < n_; ++i) { pre.SetValue(i, mixAcc[i] * invr); }
        for (uint32_t j = 0; j < n_; ++j) { post.SetValue(j, mixAcc[n_ + j] * invr); }
        for (uint32_t k = 0; k < nn; ++k) { res.SetValue(k, mixAcc[2 * n_ + k] * invr); }
        SyncSToV();

        ActivateAndStore(t, a0, a1, a2, pre, out, hpre);
    }

    __aicore__ inline void FinalizeResidentMix(uint32_t t, float a0, float a1, float a2,
                                               const LocalTensor<float>& red, float *hpre) {
        LocalTensor<float> sb = sBuf_.Get<float>();
        Muls(sb, red, invNd_, 1);
        Adds(sb, sb, normEps_, 1);
        Sqrt(sb, sb, 1);
        Div(sb, sb[8], sb, 1);
        SyncVToS();
        float invr = sb.GetValue(0);

        LocalTensor<float> pre = segBuf_.Get<float>();
        LocalTensor<float> out = outSmallQue_.AllocTensor<float>();
        auto offsets = gatherIndexBuf_.Get<uint32_t>();
        uint32_t stepBytes = batchRows_ ? 4u : 32u;
        Gather(pre, red, offsets, 32u, n_);
        Gather(out, red, offsets, 32u + n_ * stepBytes, n_);
        Gather(out[16], red, offsets, 32u + 2u * n_ * stepBytes, n_ * n_);
        Muls(pre, pre, invr, n_);
        Muls(out, out, invr, n_);
        Muls(out[16], out[16], invr, n_ * n_);
        ActivateAndStore(t, a0, a1, a2, pre, out, hpre);
    }

    __aicore__ inline void ActivateAndStore(uint32_t t, float a0, float a1, float a2,
                                           const LocalTensor<float>& pre,
                                           const LocalTensor<float>& out, float *hpre) {
        uint32_t nn = n_ * n_;
        LocalTensor<float> post = out;
        LocalTensor<float> res = out[16];

        Muls(pre, pre, a0, n_);   Add(pre, pre, biasPreT_, n_);
        SigmoidInplace(pre, n_);  Adds(pre, pre, hcEps_, n_);
        Muls(post, post, a1, n_); Add(post, post, biasPostT_, n_);
        SigmoidInplace(post, n_); Muls(post, post, static_cast<float>(2.0), n_);
        Muls(res, res, a2, nn);   Add(res, res, biasResT_, nn);

        outSmallQue_.EnQue(out);
        LocalTensor<float> out2 = outSmallQue_.DeQue<float>();
        DataCopyExtParams pp{1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
        DataCopyExtParams pr{1, static_cast<uint32_t>(nn * sizeof(float)), 0, 0, 0};
        DataCopyPad(hPostGm_[static_cast<uint64_t>(t) * n_], out2, pp);
        DataCopyPad(hResGm_[static_cast<uint64_t>(t) * nn], out2[16], pr);
        outSmallQue_.FreeTensor(out2);

        SyncVToS();
        for (uint32_t i = 0; i < n_; ++i) { hpre[i] = pre.GetValue(i); }
    }

    // 快路径：phi/gamma 常驻，x[t] 只 DMA 一次并复用于 hIn
    __aicore__ inline void ProcessResidentToken(uint32_t t, float a0, float a1, float a2) {
        LocalTensor<float> xf = xfBuf_.Get<float>();
        LocalTensor<float> xg = xgBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> work = workBuf_.Get<float>();
        LocalTensor<float> red = redBuf_.Get<float>();

        LocalTensor<DT_X> xin = inQueX_.AllocTensor<DT_X>();
        DataCopyExtParams xp{1, static_cast<uint32_t>(nD_ * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> xpad{false, 0, 0, 0};
        DataCopyPad(xin, xGm_[static_cast<uint64_t>(t) * nD_], xp, xpad);
        inQueX_.EnQue(xin);
        LocalTensor<DT_X> xin2 = inQueX_.DeQue<DT_X>();
        Cast(xf, xin2, RoundMode::CAST_NONE, nD_);
        inQueX_.FreeTensor(xin2);

        Mul(tmp, xf, xf, nD_);
        ReduceSum<float>(red, tmp, work, nD_);
        LocalTensor<float> xgUse = xf;
        if (hasGamma_) { Mul(xg, xf, gammaT_, nD_); xgUse = xg; }
        if (batchRows_) {
            // Broadcast one 64-float x slice across all phi rows per instruction.
            // vcadd writes dense row sums; each slice starts at an aligned M slot.
            BinaryRepeatParams repeat{1, 1, 1, 8, static_cast<uint8_t>(nD_ / 8u), 0};
            for (uint32_t k = 0; k < nD_ / 64u; ++k) {
                Mul(tmp, phiT_[k * 64u], xgUse[k * 64u], 64, M_, repeat);
                WholeReduceSum(red[8u + k * M_], tmp, 64, M_, 1, 1, 8);
            }
            // Mask both multiply and reduction so padded lanes are never read.
            const uint32_t tail = nD_ % 64u;
            if (tail != 0u) {
                const uint32_t k = nD_ / 64u;
                Mul(tmp, phiT_[k * 64u], xgUse[k * 64u], tail, M_, repeat);
                WholeReduceSum(red[8u + k * M_], tmp, tail, M_, 1, 1, 8);
            }
            for (uint32_t k = 1; k < (nD_ + 63u) / 64u; ++k) {
                Add(red[8], red[8], red[8u + k * M_], M_);
            }
        } else {
            for (uint32_t m = 0; m < M_; ++m) {
                Mul(tmp, xgUse, phiT_[static_cast<uint64_t>(m) * nD_], nD_);
                ReduceSum<float>(red[(m + 1) * 8], tmp, work, nD_);
            }
        }
        float hpre[8];
        FinalizeResidentMix(t, a0, a1, a2, red, hpre);

        LocalTensor<float> acc = accBuf_.Get<float>();
        // 首项直接写 acc，省掉清零和一次 Add。
        Muls(acc, xf, hpre[0], D_);
        for (uint32_t i = 1; i < n_; ++i) {
            Muls(tmp, xf[static_cast<uint64_t>(i) * D_], hpre[i], D_);
            Add(acc, acc, tmp, D_);
        }
        LocalTensor<DT_X> hout = outQueHin_.AllocTensor<DT_X>();
        Cast(hout, acc, RoundMode::CAST_RINT, D_);
        outQueHin_.EnQue(hout);
        LocalTensor<DT_X> hout2 = outQueHin_.DeQue<DT_X>();
        DataCopyExtParams pw{1, static_cast<uint32_t>(D_ * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(hInGm_[static_cast<uint64_t>(t) * D_], hout2, pw);
        outQueHin_.FreeTensor(hout2);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUF_NUM> inQueX_;
    TQue<QuePosition::VECIN, BUF_NUM> inQuePhi_;
    TQue<QuePosition::VECOUT, BUF_NUM> outQueHin_;
    TQue<QuePosition::VECOUT, BUF_NUM> outSmallQue_;
    TBuf<QuePosition::VECCALC> xfBuf_, xgBuf_, xgTileBuf_, tmpBuf_, workBuf_, accBuf_;
    TBuf<QuePosition::VECCALC> redBuf_, segBuf_, sBuf_, alphaBuf_;
    TBuf<QuePosition::VECCALC> biasPreBuf_, biasPostBuf_, biasResBuf_;
    TBuf<QuePosition::VECCALC> phiBuf_, gammaBuf_, gatherIndexBuf_;

    GlobalTensor<DT_X> xGm_, hInGm_;
    GlobalTensor<float> phiGm_, alphaGm_, biasGm_, gammaGm_;
    GlobalTensor<float> hPostGm_, hResGm_;
    LocalTensor<float> alphaT_, biasPreT_, biasPostT_, biasResT_, phiT_, gammaT_;

    uint32_t n_, D_, nD_, M_, totalT_, kChunk_, dChunk_, hasGamma_, phiResident_;
    uint32_t tStart_, tEnd_;
    bool batchRows_;
    float invNd_, normEps_, hcEps_;
};

template <typename DT_X>
__global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                   GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes,
                                   GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcPreTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcPreTilingData, tiling_data, tiling);
    KernelMhcPre<DT_X> op;
    op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tiling_data);
    op.Process();
}
