// Modified: 2026-09-28 17:27:26 +08:00 | Version: BR5-batch-tail
// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_pre_tiling.h"
#include "../op_kernel/tiling_key_mhc_pre.h"

namespace {
constexpr uint32_t MAX_KCHUNK = 4096;   // nD 分块上限（UB 预算内）
constexpr uint32_t MAX_DCHUNK = 4096;   // D  分块上限
inline uint32_t AlignDown16(uint32_t v) { return v & ~static_cast<uint32_t>(15); }
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (ub_size == 0) { ub_size = 192u * 1024u; }   // 910B UB 兜底

        // 输入张量：x(B,S,n,D) phi(M,nD) alpha(3) bias(M) gamma(n,D)可选
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_gamma = context->GetOptionalInputTensor(4);

        const gert::Shape &xShape = tensor_x->GetStorageShape();
        uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
        uint32_t S = static_cast<uint32_t>(xShape.GetDim(1));
        uint32_t n = static_cast<uint32_t>(xShape.GetDim(2));
        uint32_t D = static_cast<uint32_t>(xShape.GetDim(3));
        uint32_t nD = n * D;
        uint32_t M = n * n + 2u * n;
        uint32_t totalT = B * S;

        // eps 属性（OPTIONAL，取不到用建议默认值）
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *p_normEps = attrs->GetFloat(0);
        const float *p_hcEps = attrs->GetFloat(1);
        float normEps = (p_normEps != nullptr) ? *p_normEps : 1e-6f;
        float hcEps = (p_hcEps != nullptr) ? *p_hcEps : 1e-6f;

        // 快路径判定：phi(M*nD) + gamma(nD) 常驻 UB 是否放得下（其余暂存约 32KB 余量）
        //   常驻预算 ≈ (M*nD + nD)*4B + 32768，需 < UB - 安全余量
        // gamma 为空时不需要为 gamma 预留常驻 UB；把它计入预算会把
        // 一部分本可走 phi-resident 快路径的无 gamma case 错分到流式路径。
        uint64_t gammaBytes = (tensor_gamma != nullptr) ? static_cast<uint64_t>(nD) : 0u;
        uint64_t residentBytes = (static_cast<uint64_t>(M) * nD + gammaBytes) * sizeof(float) + 32768u;
        bool phiResident = (D % 16u == 0u) && residentBytes < (ub_size > 40960u ? ub_size - 40960u : 0u);

        // 分块：16 对齐，且不超过上限。常驻时单块（kChunk=nD, dChunk=D）
        uint32_t kChunk, dChunk;
        if (phiResident) {
            kChunk = nD;
            dChunk = D;
        } else {
            kChunk = nD < MAX_KCHUNK ? nD : MAX_KCHUNK;
            kChunk = AlignDown16(kChunk);
            if (kChunk == 0) { kChunk = nD; }           // nD<16 兜底（D≥16 时不会发生）
            dChunk = D < MAX_DCHUNK ? D : MAX_DCHUNK;
            dChunk = AlignDown16(dChunk);
            if (dChunk == 0) { dChunk = D; }
        }

        // 核数与每核 token 数：totalT 少于核数时只用 totalT 个核
        uint32_t useCores = static_cast<uint32_t>(num_cores_aiv);
        if (useCores == 0) { useCores = 1; }
        if (totalT < useCores) { useCores = totalT == 0 ? 1 : totalT; }
        uint32_t tokensPerCore = (totalT + useCores - 1) / useCores;

        // 数据类型 -> tiling key（x=fp16/bf16）
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 填充 tiling
        MhcPreTilingData *tiling = context->GetTilingData<MhcPreTilingData>();
        tiling->B = B;
        tiling->S = S;
        tiling->n = n;
        tiling->D = D;
        tiling->nD = nD;
        tiling->M = M;
        tiling->totalT = totalT;
        tiling->tokensPerCore = tokensPerCore;
        tiling->kChunk = kChunk;
        tiling->dChunk = dChunk;
        tiling->hasGamma = (tensor_gamma != nullptr) ? 1u : 0u;
        tiling->phiResident = phiResident ? 1u : 0u;
        tiling->invNd = static_cast<float>(1.0 / static_cast<double>(nD));
        tiling->normEps = normEps;
        tiling->hcEps = hcEps;

        context->SetBlockDim(useCores);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);   // (B,S,n,D)
        int64_t B = xShape->GetDim(0);
        int64_t S = xShape->GetDim(1);
        int64_t n = xShape->GetDim(2);
        int64_t D = xShape->GetDim(3);

        gert::Shape *hIn = context->GetOutputShape(0);           // (B,S,D)
        hIn->SetDimNum(3);
        hIn->SetDim(0, B); hIn->SetDim(1, S); hIn->SetDim(2, D);

        gert::Shape *hPost = context->GetOutputShape(1);         // (B,S,n)
        hPost->SetDimNum(3);
        hPost->SetDim(0, B); hPost->SetDim(1, S); hPost->SetDim(2, n);

        gert::Shape *hRes = context->GetOutputShape(2);          // (B,S,n,n)
        hRes->SetDimNum(4);
        hRes->SetDim(0, B); hRes->SetDim(1, S); hRes->SetDim(2, n); hRes->SetDim(3, n);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));  // hIn 同 x
        context->SetOutputDataType(1, ge::DT_FLOAT);                  // hPost fp32
        context->SetOutputDataType(2, ge::DT_FLOAT);                  // hRes  fp32
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class MhcPre : public OpDef {
    public:
        explicit MhcPre(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("phi")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("alpha")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("bias")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("gamma")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("hIn")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("hPost")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("hRes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("normEps").AttrType(OPTIONAL).Float(1e-06);
            this->Attr("hcEps").AttrType(OPTIONAL).Float(1e-06);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MhcPre);
}  // namespace ops
