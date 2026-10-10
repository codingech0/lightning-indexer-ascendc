// Self-test executable only: the analytic oracle never feeds the operator output.
#include "acl/acl.h"
#include "aclnn_lightning_indexer.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(call) do { auto status = (call); if (status != 0) \
    throw std::runtime_error(std::string(#call) + " status=" + std::to_string(status)); } while (0)

struct Tensor {
    void *memory = nullptr;
    aclTensor *tensor = nullptr;
    size_t bytes;
    Tensor(const std::vector<int64_t>& shape, aclDataType dtype, const void *data, size_t size): bytes(size) {
        CHECK(aclrtMalloc(&memory, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        try {
            CHECK(aclrtMemcpy(memory, bytes, data, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            tensor = aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0,
                                    ACL_FORMAT_ND, shape.data(), shape.size(), memory);
            if (!tensor) throw std::runtime_error("aclCreateTensor returned null");
        } catch (...) { aclrtFree(memory); memory = nullptr; throw; }
    }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    ~Tensor() { if (tensor) aclDestroyTensor(tensor); if (memory) aclrtFree(memory); }
    void Read(void *out) { CHECK(aclrtMemcpy(out, bytes, memory, bytes, ACL_MEMCPY_DEVICE_TO_HOST)); }
};

struct Workspace {
    void *memory = nullptr;
    ~Workspace() { if (memory) aclrtFree(memory); }
};

// Inputs and expected values are exact small integers for this smoke test.
uint16_t Encode(float x, bool bf16) {
    if (bf16) { uint32_t bits; std::memcpy(&bits, &x, 4); return static_cast<uint16_t>(bits >> 16); }
    auto h = aclFloatToFloat16(x);
    uint16_t bits; static_assert(sizeof(h) == 2, "FP16 storage"); std::memcpy(&bits, &h, 2); return bits;
}

void Run(aclrtStream stream, const char *name, bool bf16, bool fp32Weights,
         int mode, int count, bool lengths, bool returnValues, bool negative = false, int heads = 2) {
    constexpr int B=2, S1=2, S2=4, D=128;
    const aclDataType dtype = bf16 ? ACL_BF16 : ACL_FLOAT16;
    std::vector<uint16_t> q(B*S1*heads*D, Encode(1,bf16)), k(B*S2*D), w(B*S1*heads,Encode(negative?-1:1,bf16));
    std::vector<float> wf(w.size(),negative?-1.0f:1.0f);
    for (int b=0;b<B;++b) for (int j=0;j<S2;++j)
        for (int d=0;d<D;++d) k[(b*S2+j)*D+d]=Encode(j+1,bf16);
    std::vector<int32_t> ql{1,2}, kl{3,4};
    std::vector<int32_t> idx(B*S1*count, -777);
    std::vector<uint16_t> val(idx.size(), Encode(-777,bf16));
    Tensor qt({B,S1,heads,D},dtype,q.data(),q.size()*2);
    Tensor kt({B,S2,1,D},dtype,k.data(),k.size()*2);
    Tensor wt({B,S1,heads},fp32Weights?ACL_FLOAT:dtype,
              fp32Weights?static_cast<const void*>(wf.data()):static_cast<const void*>(w.data()),
              w.size()*(fp32Weights?4:2));
    Tensor qlt({B},ACL_INT32,ql.data(),ql.size()*4), klt({B},ACL_INT32,kl.data(),kl.size()*4);
    Tensor it({B,S1,1,count},ACL_INT32,idx.data(),idx.size()*4);
    Tensor vt({B,S1,1,count},dtype,val.data(),val.size()*2);
    uint64_t workspaceSize=0;
    aclOpExecutor *executor=nullptr;
    CHECK(aclnnLightningIndexerGetWorkspaceSize(qt.tensor,kt.tensor,wt.tensor,
        lengths?qlt.tensor:nullptr,lengths?klt.tensor:nullptr,count,mode,returnValues,
        it.tensor,returnValues?vt.tensor:nullptr,&workspaceSize,&executor));
    Workspace workspace;
    if (workspaceSize) CHECK(aclrtMalloc(&workspace.memory,workspaceSize,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnLightningIndexer(workspace.memory,workspaceSize,executor,stream));
    CHECK(aclrtSynchronizeStream(stream));
    it.Read(idx.data());
    if (returnValues) vt.Read(val.data());
    for (int b=0;b<B;++b) for(int s=0;s<S1;++s) {
        int len=lengths?kl[b]:S2;
        std::vector<float> score(len);
        for(int j=0;j<len;++j) score[j]=(mode==3 && j>len-S1+s)?0.0f:
            static_cast<float>((negative?-1:1)*heads*D*(j+1));
        std::vector<int> selected(len); std::iota(selected.begin(),selected.end(),0);
        std::stable_sort(selected.begin(),selected.end(),[&](int a,int c){return score[a]>score[c];});
        selected.resize(std::min(count,len)); std::sort(selected.begin(),selected.end());
        if(lengths && s>=ql[b]) selected.clear();
        for(int i=0;i<count;++i) {
            const size_t offset=(b*S1+s)*count+i;
            int expected=i<static_cast<int>(selected.size())?selected[i]:-1;
            float expectedValue=expected<0?0.0f:score[expected];
            if(idx[offset]!=expected || (returnValues && val[offset]!=Encode(expectedValue,bf16))) {
                std::cerr<<name<<" b="<<b<<" s="<<s<<" slot="<<i<<" index="<<idx[offset]
                    <<" expected="<<expected<<" value_bits="<<val[offset]
                    <<" expected_bits="<<Encode(expectedValue,bf16)<<"\n";
                throw std::runtime_error("output mismatch (provisional statement semantics)");
            }
        }
    }
    std::cout<<"PASS "<<name<<std::endl;
}

#ifndef LIGHTNING_TEST_NO_MAIN
int main(int argc,char **argv) {
    const int device=argc>1?std::stoi(argv[1]):0;
    aclrtStream stream=nullptr;
    bool initialized=false,selected=false;
    int result=0;
    try {
        CHECK(aclInit(nullptr)); initialized=true;
        CHECK(aclrtSetDevice(device)); selected=true;
        CHECK(aclrtCreateStream(&stream));
        Run(stream,"fp16_default",false,false,0,2,false,true);
        Run(stream,"fp16_causal",false,false,3,2,false,true);
        Run(stream,"variable_lengths_padding",false,false,0,5,true,true);
        Run(stream,"values_absent",false,false,0,2,false,false);
        Run(stream,"fp16_fp32_weights",false,true,0,2,false,true);
        Run(stream,"bf16",true,false,0,2,false,true);
        Run(stream,"bf16_fp32_weights",true,true,0,2,false,true);
        Run(stream,"causal_negative_statement_semantics",false,false,3,1,false,true,true);
        Run(stream,"heads_one",false,false,0,2,false,true,false,1);
        std::cout<<"9 NPU smoke cases passed; NOT official correctness certification.\n";
    } catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<std::endl; result=1; }
    if(stream && aclrtDestroyStream(stream)!=0) result=1;
    if(selected && aclrtResetDevice(device)!=0) result=1;
    if(initialized && aclFinalize()!=0) result=1;
    return result;
}
#endif
