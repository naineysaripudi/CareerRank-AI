"""NDCG implementation for labeled ranked results."""

import math


def ndcg_at_k(relevance: list[int], k: int) -> float:
    values = relevance[:k]
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(values))
    ideal = sorted(values, reverse=True)
    idcg = sum(value / math.log2(index + 2) for index, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
