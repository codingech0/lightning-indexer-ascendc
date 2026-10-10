// Test-only CPU oracle: full double-precision scoring and sorting, never output fallback.
#define LIGHTNING_TEST_NO_MAIN
#include "npu_smoke.cpp"
#include <random>

// Finite dyadic inputs; BF16 output uses round-to-nearest, ties-to-even.
uint16_t Pack(float x, bool bf16) {
    if (!bf16) return Encode(x, false);
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(bits >> 16);
}

struct Case {
    std::string name;
    int batch, queries, keys, heads, count, mode;
    bool bf16, fp32Weights, queryLengths, keyLengths, values;
    int pattern; // 0: random dyadics; 1: all ties; 2: late key wins; 3: BF16 midpoints.
    uint32_t seed;
};

void RunGenerated(aclrtStream stream, const Case& c) {
    constexpr int D = 128;
    const auto dtype = c.bf16 ? ACL_BF16 : ACL_FLOAT16;
    std::mt19937 rng(c.seed);
    auto next = [&]() { return (static_cast<int>(rng() % 5) - 2) * 0.25f; };
    std::vector<float> q(c.batch*c.queries*c.heads*D), k(c.batch*c.keys*D);
    std::vector<float> w(c.batch*c.queries*c.heads);
    for (auto& x : q) x = next();
    for (auto& x : k) x = next();
    for (auto& x : w) x = next();
    if (c.pattern == 1) std::fill(w.begin(), w.end(), 0.0f);
    if (c.pattern == 2) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
        std::fill(w.begin(), w.end(), 1.0f);
        for (size_t i=0; i<q.size(); i+=D) q[i]=1.0f;
        for (int b=0; b<c.batch; ++b) k[(b*c.keys+c.keys-1)*D]=2.0f;
    }
    if (c.pattern == 3) {
        std::fill(q.begin(),q.end(),0.0f);
        std::fill(k.begin(),k.end(),0.0f);
        for (size_t i=0; i<q.size(); i+=D) q[i]=q[i+1]=q[i+2]=1.0f;
        for (int b=0; b<c.batch; ++b) {
            for (int j=0; j<c.keys; ++j) {
                k[(b*c.keys+j)*D]=5.0f;
                k[(b*c.keys+j)*D+1]=0.375f;
                k[(b*c.keys+j)*D+2]=0.046875f+j*0.015625f;
            }
            for (int s=0; s<c.queries; ++s) for (int h=0; h<c.heads; ++h)
                w[(b*c.queries+s)*c.heads+h]=b%2?-1.0f:1.0f;
        }
    }
    std::vector<int32_t> ql(c.batch), kl(c.batch);
    for (int b=0; b<c.batch; ++b) {
        ql[b] = b==0 ? c.queries : (b==1 ? 0 : std::max(0,c.queries-1));
        kl[b] = b==0 ? c.keys : (b==1 ? 0 : std::max(0,c.keys-2));
        // Padding has large positive data, so accidental inclusion is observable.
        if (c.keyLengths)
            for (int j=kl[b]; j<c.keys; ++j)
                for (int d=0; d<D; ++d) k[(b*c.keys+j)*D+d]=16.0f;
    }
    auto encode = [&](const std::vector<float>& data) {
        std::vector<uint16_t> result(data.size());
        for (size_t i=0; i<data.size(); ++i) result[i]=Pack(data[i],c.bf16);
        return result;
    };
    auto qb=encode(q), kb=encode(k), wb=encode(w);
    std::vector<int32_t> indices(c.batch*c.queries*c.count,-777);
    std::vector<uint16_t> values(indices.size(),Pack(-777,c.bf16));
    Tensor qt({c.batch,c.queries,c.heads,D},dtype,qb.data(),qb.size()*2);
    Tensor kt({c.batch,c.keys,1,D},dtype,kb.data(),kb.size()*2);
    Tensor wt({c.batch,c.queries,c.heads},c.fp32Weights?ACL_FLOAT:dtype,
        c.fp32Weights?static_cast<const void*>(w.data()):static_cast<const void*>(wb.data()),
        w.size()*(c.fp32Weights?4:2));
    Tensor qlt({c.batch},ACL_INT32,ql.data(),ql.size()*4);
    Tensor klt({c.batch},ACL_INT32,kl.data(),kl.size()*4);
    Tensor it({c.batch,c.queries,1,c.count},ACL_INT32,indices.data(),indices.size()*4);
    Tensor vt({c.batch,c.queries,1,c.count},dtype,values.data(),values.size()*2);
    uint64_t workspaceSize=0;
    aclOpExecutor* executor=nullptr;
    std::cout << "RUN " << c.name << " seed=" << c.seed << std::endl;
    CHECK(aclnnLightningIndexerGetWorkspaceSize(qt.tensor,kt.tensor,wt.tensor,
        c.queryLengths?qlt.tensor:nullptr,c.keyLengths?klt.tensor:nullptr,
        c.count,c.mode,c.values,it.tensor,c.values?vt.tensor:nullptr,&workspaceSize,&executor));
    Workspace workspace;
    if (workspaceSize) CHECK(aclrtMalloc(&workspace.memory,workspaceSize,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnLightningIndexer(workspace.memory,workspaceSize,executor,stream));
    CHECK(aclrtSynchronizeStream(stream));
    it.Read(indices.data());
    if (c.values) vt.Read(values.data());

    for (int b=0; b<c.batch; ++b) for (int s=0; s<c.queries; ++s) {
        const int len=c.keyLengths?kl[b]:c.keys;
        std::vector<double> scores(len,0.0);
        std::vector<int> selected;
        if (!c.queryLengths || s<ql[b]) {
            selected.resize(len);
            std::iota(selected.begin(),selected.end(),0);
            for (int j=0; j<len; ++j) {
                // Provisional semantics from the supplied statement example.
                if (c.mode==3 && j>len-c.queries+s) continue;
                for (int h=0; h<c.heads; ++h) {
                    double dot=0;
                    for (int d=0; d<D; ++d)
                        dot += static_cast<double>(q[((b*c.queries+s)*c.heads+h)*D+d]) * k[(b*c.keys+j)*D+d];
                    scores[j] += std::max(0.0,dot)*w[(b*c.queries+s)*c.heads+h];
                }
            }
            // Full sort is independent of the device's bounded heap implementation.
            std::sort(selected.begin(),selected.end(),[&](int a,int z) {
                return scores[a]!=scores[z] ? scores[a]>scores[z] : a<z;
            });
            selected.resize(std::min(c.count,len));
            std::sort(selected.begin(),selected.end());
        }
        for (int slot=0; slot<c.count; ++slot) {
            const size_t offset=(b*c.queries+s)*c.count+slot;
            const int expected=slot<static_cast<int>(selected.size())?selected[slot]:-1;
            const auto expectedBits=Pack(expected<0?0.0f:static_cast<float>(scores[expected]),c.bf16);
            if (indices[offset]!=expected || (c.values && values[offset]!=expectedBits)) {
                std::cerr << c.name << " b=" << b << " s=" << s << " slot=" << slot
                          << " index=" << indices[offset] << " expected=" << expected
                          << " value_bits=" << values[offset] << " expected_bits=" << expectedBits << '\n';
                throw std::runtime_error("generated case mismatch; seed="+std::to_string(c.seed));
            }
        }
    }
    std::cout << "PASS " << c.name << std::endl;
}

int main(int argc,char** argv) {
    aclrtStream stream=nullptr;
    bool initialized=false,selected=false;
    int result=0;
    const int device=argc>1?std::stoi(argv[1]):0;
    try {
        CHECK(aclInit(nullptr)); initialized=true;
        CHECK(aclrtSetDevice(device)); selected=true;
        CHECK(aclrtCreateStream(&stream));
        int passed=0;
        for (int i=0; i<24; ++i) {
            const int hs[]={1,3,8,24,64,2};
            const int ks[]={1,7,17,31,65,129};
            const int keys=ks[i%6];
            const int counts[]={1,keys,std::min(2048,keys+3)};
            Case c{"random_"+std::to_string(i),3,3,keys,hs[i%6],counts[i%3],
                i%2?3:0,static_cast<bool>((i/2)%2),static_cast<bool>((i/4)%2),
                static_cast<bool>(i&1),static_cast<bool>(i&2),i%3!=0,0,20260913U+static_cast<uint32_t>(i)};
            RunGenerated(stream,c); ++passed;
        }
        const Case edges[]={
            {"bf16_signed_midpoint_rounding",2,1,3,1,3,0,true,false,false,false,true,3,0},
            {"ties_keep_small_indices",1,2,257,3,129,0,false,false,false,false,true,1,1},
            {"late_key_replaces_heap_root",1,1,257,1,1,0,false,false,false,false,true,2,2},
            {"max_k_heap_capacity",1,1,8193,1,8192,0,false,false,false,false,true,1,3},
            {"k_2048_equal_keys",1,1,2048,1,2048,0,true,true,false,false,true,1,4},
            {"max_k_padding",3,1,17,1,8192,0,true,false,true,true,true,0,5},
            {"causal_negative_offset",3,5,2,3,3,3,false,true,true,true,true,0,6}
        };
        for (const auto& c : edges) { RunGenerated(stream,c); ++passed; }
        std::cout << passed << " NPU generated cases passed; provisional semantics, NOT official certification.\n";
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; result=1; }
    if (stream && aclrtDestroyStream(stream)!=0) result=1;
    if (selected && aclrtResetDevice(device)!=0) result=1;
    if (initialized && aclFinalize()!=0) result=1;
    return result;
}
