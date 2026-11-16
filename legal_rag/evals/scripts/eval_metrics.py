"""检索指标的唯一实现：评测脚本、全流程管道与出图脚本共用，避免口径分叉。

NDCG 口径：per_k_truncated_v2
  IDCG@k = Σ gains[:min(k, |gold|)] / log2(i+2)
  DCG@k  = Σ gain / log2(pos+1)，pos < min(k, len(retrieved))
  gold 为空或 IDCG=0 时该 k 记 None（缺失），不参与聚合。
与 v1 历史结果（三 k 共用 IDCG@20 且 gold<k 归零）不可直接比较。
"""
import math

NDCG_DEFINITION = "per_k_truncated_v2"
NDCG_KS = (5, 10, 20)


def ndcg_by_k(retrieved_ids, get_gain, gold_pairs, ks=NDCG_KS):
    """返回 {k: ndcg 或 None}。get_gain(law_id, article) -> 增益。"""
    gains = sorted((get_gain(lid, art) for lid, art in gold_pairs), reverse=True)
    out = {k: None for k in ks}
    for k in ks:
        ideal_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:min(k, len(gains))]))
        if ideal_dcg <= 0:
            continue
        dcg = sum(get_gain(lid, art) / math.log2(i + 2)
                  for i, (lid, art) in enumerate(retrieved_ids[:min(k, len(retrieved_ids))]))
        out[k] = dcg / ideal_dcg
    return out


def average_valid(values):
    """忽略 None 求均值；返回 (均值或 None, 有效条数)。"""
    valid = [v for v in values if v is not None]
    return (round(sum(valid) / len(valid), 4) if valid else None), len(valid)
