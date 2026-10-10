# Extracted verbatim from the task statement. CPU self-test only.

import numpy as np

def lightning_indexer(query, key, weights, sparse_count, sparse_mode=3,
                      act_seq_k=None):
    # query:(B,S1,N1,D)  key:(B,S2,N2,D), N2=1  weights:(B,S1,N1)
    # 输出 sparse_indices:(B,S1,N2,sparse_count) int32
    q = query.astype(np.float32); k = key.astype(np.float32); w = weights.astype(np.float32)
    B, S1, N1, D = q.shape
    _, S2, N2, _ = k.shape
    indices = np.full((B, S1, N2, sparse_count), -1, dtype=np.int32)
    values   = np.full((B, S1, N2, sparse_count), 0.0, dtype=np.float32)
    for b in range(B):
        Klen = S2 if act_seq_k is None else int(act_seq_k[b])   # 当前 batch 有效 key 长度
        K = k[b, :Klen, 0, :]                                    # (Klen, D)
        for s in range(S1):
            Q = q[b, s, :, :]                                    # (N1, D) = (g, D)
            W = w[b, s, :].reshape(-1, 1)                        # (N1, 1) = (g, 1)
            score = Q @ K.T                                      # (g, Klen) 相似度
            score = np.maximum(score, 0.0)                      # ReLU
            if sparse_mode == 3:                                # rightDownCausal 掩码
                j = np.arange(Klen)
                mask = (j <= (Klen - S1 + s)).astype(np.float32)
                score = score * mask[None, :]
            agg = (W * score).sum(axis=0)                       # (Klen,) group 维加权求和
            kk = min(sparse_count, Klen)
            order = np.argsort(-agg, kind='stable')[:kk]        # 降序 Top-k，同分按索引升序
            order = np.sort(order)
            indices[b, s, 0, :kk] = order
            values[b, s, 0, :kk] = agg[order]
    return indices, values.astype(np.float16)

