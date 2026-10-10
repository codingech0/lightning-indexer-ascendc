"""CPU algorithm tests, not Ascend C emulation or NPU correctness certification.

The oracle below uses matrix multiplication/full sort. The candidate uses
sequential FP32 accumulation/streaming heap, as planned for the device baseline.
Only tests/ imports NumPy or executes host numerical reference calculations.
"""
import heapq
import unittest
import numpy as np
from ml_dtypes import bfloat16
from statement_reference import lightning_indexer as statement


def compute(q, k, w, count, mode=0, qlens=None, klens=None, values=True, streaming=False):
    bsz, sq, heads, dim = q.shape
    sk = k.shape[1]
    dtype = q.dtype
    q, k, w = (x.astype(np.float32) for x in (q, k, w))
    indices = np.full((bsz, sq, 1, count), -1, np.int32)
    output = np.zeros(indices.shape, np.float32)
    for b in range(bsz):
        qlen = sq if qlens is None else int(qlens[b])
        klen = sk if klens is None else int(klens[b])
        for s in range(qlen):
            if streaming:
                heap = []
                for j in range(klen):
                    score = np.float32(0)
                    if mode == 0 or j <= klen - sq + s:
                        for h in range(heads):
                            dot = np.float32(0)
                            for d in range(dim):
                                dot = np.float32(dot + np.float32(q[b,s,h,d] * k[b,j,0,d]))
                            score = np.float32(score + np.float32(w[b,s,h] * max(dot, np.float32(0))))
                    item = (float(score), -j)
                    if len(heap) < count:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
                pairs = sorted((-j, v) for v, j in heap)
            else:
                scores = np.maximum(q[b,s] @ k[b,:klen,0].T, 0)
                if mode == 3:
                    scores *= (np.arange(klen) <= klen-sq+s)[None,:]
                scores = (w[b,s,:,None] * scores).sum(axis=0)
                selected = np.sort(np.argsort(-scores, kind='stable')[:count])
                pairs = [(int(j), scores[j]) for j in selected]
            for i, (j, v) in enumerate(pairs):
                indices[b,s,0,i] = j
                output[b,s,0,i] = v
    return indices, output.astype(dtype) if values else None


class SemanticsTests(unittest.TestCase):
    def inputs(self, dtype=np.float16, heads=2, batches=2, sq=3, sk=7):
        # Bounded integers keep dot products exact so selection bugs are not
        # hidden by BLAS-versus-device floating-point reduction differences.
        rng = np.random.default_rng(42)
        return (rng.integers(-2, 3, (batches,sq,heads,128)).astype(dtype),
                rng.integers(-2, 3, (batches,sk,1,128)).astype(dtype),
                rng.integers(-2, 3, (batches,sq,heads)).astype(dtype))

    def compare(self, inputs=None, **kwargs):
        inputs = self.inputs() if inputs is None else inputs
        expected = compute(*inputs, streaming=False, **kwargs)
        candidate = compute(*inputs, streaming=True, **kwargs)
        np.testing.assert_array_equal(candidate[0], expected[0])
        if expected[1] is None:
            self.assertIsNone(candidate[1])
        else:
            np.testing.assert_array_equal(candidate[1], expected[1])
        self.assertEqual(candidate[0].dtype, np.int32)
        return candidate

    def test_mode_zero(self): self.compare(count=3)
    def test_causal(self): self.compare(count=3, mode=3)
    def test_short_key(self):
        idx, _ = self.compare(count=5, klens=[2,2])
        self.assertTrue(np.all(idx[...,2:] == -1))
    def test_different_batch_lengths(self): self.compare(count=3, klens=[2,6])
    def test_k_less(self): self.compare(count=1)
    def test_k_equal_or_more(self):
        for count in (7,8,2048,3072,4096,5120,6144,7168,8192):
            with self.subTest(count=count): self.compare(count=count)
    def test_heads(self):
        for heads in (1,3,8,24,64):
            with self.subTest(heads=heads): self.compare(self.inputs(heads=heads), count=3)
    def test_fp16(self): self.compare(self.inputs(np.float16), count=3)
    def test_bf16_cpu_representation(self): self.compare(self.inputs(bfloat16), count=3)
    def test_fp32_weights(self):
        for dtype in (np.float16,bfloat16):
            q,k,w = self.inputs(dtype)
            self.compare((q,k,w.astype(np.float32)), count=3)
    def test_no_values(self): self.compare(count=3, values=False)
    def test_values_dtype(self):
        for dtype in (np.float16,bfloat16):
            _,v = self.compare(self.inputs(dtype), count=3, values=True)
            self.assertEqual(v.dtype, np.dtype(dtype))
    def test_query_padding_and_empty_keys(self):
        idx,v = self.compare(count=3, qlens=[1,0], klens=[0,7])
        self.assertTrue(np.all(idx == -1)); self.assertTrue(np.all(v == 0))
    def test_stable_ties(self):
        q,k,w = self.inputs(); w.fill(0)
        idx,_ = self.compare((q,k,w), count=3)
        np.testing.assert_array_equal(idx[0,0,0], [0,1,2])
    def test_padding_with_large_scores(self):
        q,k,w = self.inputs(); k[:,2:] = 100
        idx,_ = self.compare((q,k,w), count=3, klens=[2,2])
        self.assertTrue(np.all(idx < 2))
    def test_statement_agreement(self):
        for mode in (0,3):
            q,k,w = self.inputs()
            idx,val = statement(q,k,w,5,mode,act_seq_k=[2,7])
            actual = self.compare((q,k,w), count=5, mode=mode, klens=[2,7])
            np.testing.assert_array_equal(actual[0],idx)
            np.testing.assert_array_equal(actual[1],val)
    def test_mask_ambiguity_counterexample(self):
        q = np.ones((1,2,1,128), np.float16)
        k = np.ones((1,2,1,128), np.float16)
        w = -np.ones((1,2,1), np.float16)
        idx,_ = self.compare((q,k,w),count=1,mode=3)
        self.assertEqual(idx[0,0,0,0], 1)  # statement selects the invisible zero
    def test_random_shapes(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            sq,sk,h = (int(x) for x in rng.integers(1,6,3))
            with self.subTest(seed=seed):
                self.compare(self.inputs(heads=h,batches=1,sq=sq,sk=sk),
                             count=int(rng.integers(1,9)),mode=3 if seed%2 else 0)


if __name__ == '__main__': unittest.main(verbosity=2)
