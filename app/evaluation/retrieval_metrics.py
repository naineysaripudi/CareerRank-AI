"""Ranking metrics over relevance labels."""


def precision_at_k(relevance: list[int], k: int) -> float:
    values = relevance[:k]
    return sum(values) / max(1, len(values))


def recall_at_k(relevance: list[int], total_relevant: int, k: int) -> float:
    return sum(relevance[:k]) / max(1, total_relevant)


def mean_reciprocal_rank(relevance: list[int]) -> float:
    for index, value in enumerate(relevance, start=1):
        if value:
            return 1 / index
    return 0.0
