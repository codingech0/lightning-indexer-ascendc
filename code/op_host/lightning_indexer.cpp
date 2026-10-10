// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lightning_indexer_tiling.h"
#include "../op_kernel/tiling_key_lightning_indexer.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context == nullptr || context->GetPlatformInfo() == nullptr || context->GetAttrs() == nullptr) {
            return ge::GRAPH_FAILED;
        }
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_query = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_key = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_weights = context->GetRequiredInputTensor(2);
        if (tensor_query == nullptr || tensor_key == nullptr || tensor_weights == nullptr ||
            context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
            context->GetInputShape(2) == nullptr || num_cores_aiv <= 0) {
            return ge::GRAPH_FAILED;
        }
        const auto &q = context->GetInputShape(0)->GetStorageShape();
        const auto &k = context->GetInputShape(1)->GetStorageShape();
        const auto &w = context->GetInputShape(2)->GetStorageShape();
        if (q.GetDimNum() != 4 || k.GetDimNum() != 4 || w.GetDimNum() != 3 ||
            q.GetDim(0) <= 0 || q.GetDim(1) <= 0 || k.GetDim(1) <= 0 ||
            q.GetDim(2) < 1 || q.GetDim(2) > 64 || q.GetDim(3) != 128 ||
            k.GetDim(0) != q.GetDim(0) || k.GetDim(2) != 1 || k.GetDim(3) != 128 ||
            w.GetDim(0) != q.GetDim(0) || w.GetDim(1) != q.GetDim(1) || w.GetDim(2) != q.GetDim(2)) {
            return ge::GRAPH_FAILED;
        }
        const gert::Tensor *tensor_actual_seq_lengths_query = context->GetOptionalInputTensor(3);
        const gert::Tensor *tensor_actual_seq_lengths_key = context->GetOptionalInputTensor(4);
        // Validate only metadata here. Sequence-length values remain device inputs.
        for (size_t input = 3; input < 5; ++input) {
            const auto *tensor = input == 3 ? tensor_actual_seq_lengths_query : tensor_actual_seq_lengths_key;
            if (tensor != nullptr) {
                const auto *shape = context->GetInputShape(input);
                if (shape == nullptr || tensor->GetDataType() != ge::DT_INT32 ||
                    shape->GetStorageShape().GetDimNum() != 1 ||
                    shape->GetStorageShape().GetDim(0) != q.GetDim(0)) {
                    return ge::GRAPH_FAILED;
                }
            }
        }
        ge::DataType dtype_query = tensor_query->GetDataType(); // 获取数据类型
        const auto dtype_weights = tensor_weights->GetDataType();
        if ((dtype_query != ge::DT_FLOAT16 && dtype_query != ge::DT_BF16) ||
            tensor_key->GetDataType() != dtype_query ||
            (dtype_weights != dtype_query && dtype_weights != ge::DT_FLOAT)) {
            return ge::GRAPH_FAILED;
        }
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_sparse_count = attrs->GetInt(0);
        const int64_t *attr_sparse_mode = attrs->GetInt(1);
        const bool *attr_return_values = attrs->GetBool(2);
        if (attr_sparse_count == nullptr || attr_sparse_mode == nullptr || attr_return_values == nullptr ||
            !((*attr_sparse_count >= 1 && *attr_sparse_count <= 2048) ||
              (*attr_sparse_count >= 3072 && *attr_sparse_count <= 8192 && *attr_sparse_count % 1024 == 0)) ||
            (*attr_sparse_mode != 0 && *attr_sparse_mode != 3)) {
            return ge::GRAPH_FAILED;
        }
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_QUERY = static_cast<uint32_t>(dtype_query);
        ASCENDC_TPL_SEL_PARAM(context, DT_QUERY);
        // 示例: 计算tiling方案并填充tiling结构体
        LightningIndexerTilingData *tiling = context->GetTilingData<LightningIndexerTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->length = static_cast<uint64_t>(tensor_query->GetShapeSize());
        tiling->batch = static_cast<uint64_t>(q.GetDim(0));
        tiling->queryLength = static_cast<uint64_t>(q.GetDim(1));
        tiling->keyLength = static_cast<uint64_t>(k.GetDim(1));
        tiling->heads = static_cast<uint32_t>(q.GetDim(2));
        tiling->sparseCount = static_cast<uint32_t>(*attr_sparse_count);
        tiling->sparseMode = static_cast<uint32_t>(*attr_sparse_mode);
        tiling->returnValues = *attr_return_values ? 1U : 0U;
        tiling->hasQueryLengths = tensor_actual_seq_lengths_query != nullptr ? 1U : 0U;
        tiling->hasKeyLengths = tensor_actual_seq_lengths_key != nullptr ? 1U : 0U;
        tiling->weightsFloat32 = dtype_weights == ge::DT_FLOAT ? 1U : 0U;
        // 配置启动核数
        // Single-AIV diagnostic baseline: 32 KiB Q + 512 B K + 256 B W + 64 KiB heap.
        if (ub_size < 99072) {
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim(1);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr || context->GetAttrs() == nullptr) {
            return GRAPH_FAILED;
        }
        const auto *query = context->GetInputShape(0);
        const auto *key = context->GetInputShape(1);
        auto *indices = context->GetOutputShape(0);
        const auto *count = context->GetAttrs()->GetInt(0);
        if (query == nullptr || key == nullptr || indices == nullptr || count == nullptr ||
            query->GetDimNum() != 4 || key->GetDimNum() != 4) {
            return GRAPH_FAILED;
        }
        if (!((*count >= 1 && *count <= 2048) ||
              (*count >= 3072 && *count <= 8192 && *count % 1024 == 0))) {
            return GRAPH_FAILED;
        }
        // BSND: output is [B, S1, 1, sparse_count], not [B, S1, N1, sparse_count].
        *indices = *query;
        indices->SetDim(2, 1);
        indices->SetDim(3, *count);
        // An absent optional output must not be dereferenced.
        auto *values = context->GetOutputShape(1);
        if (values != nullptr) {
            *values = *indices;
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
        const auto queryType = context->GetInputDataType(0);
        const auto weightType = context->GetInputDataType(2);
        if ((queryType != DT_FLOAT16 && queryType != DT_BF16) ||
            context->GetInputDataType(1) != queryType ||
            (weightType != queryType && weightType != DT_FLOAT)) {
            return GRAPH_FAILED;
        }
        if (context->SetOutputDataType(0, DT_INT32) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        // Both output slots are declared by OpDef. Values follow query dtype.
        // Optional-output inference behavior still needs validation with the target ASC package.
        return context->SetOutputDataType(1, queryType);
    }
}  // namespace ge

namespace ops {
    class LightningIndexer : public OpDef {
    public:
        explicit LightningIndexer(const char *name) : OpDef(name) {
            this->Input("query")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("key")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("weights")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("actual_seq_lengths_query")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("actual_seq_lengths_key")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("sparse_indices")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("sparse_values")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("sparse_count").AttrType(OPTIONAL).Int(2048);
            this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
            this->Attr("return_values").AttrType(OPTIONAL).Bool();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LightningIndexer);
}  // namespace ops
