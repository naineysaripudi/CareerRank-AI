# Evaluation

The evaluation layer uses three explicit synthetic candidate profiles with role-based relevance labels and reports Recall@K, Precision@K, MRR, and NDCG@K. Run it with:

```powershell
python scripts\evaluate.py
```

The current development run produced:

| System | Precision@5 | Recall@5 | MRR | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Semantic only | 0.20 | 0.11 | 0.38 | 0.32 |
| Personalized ranker | 1.00 | 0.54 | 1.00 | 1.00 |

These results are calculated from the checked-in synthetic dataset, not copied from an external benchmark. The relevance label is whether the job title exactly matches the candidate's declared target role, so the result demonstrates the experiment mechanics rather than real-world hiring quality.

An optional `MLRanker` is implemented in `app/ranking/ml_ranker.py`; it requires labeled feature rows and is intentionally not claimed as superior without a trained evaluation set.

The synthetic job dataset is for development and does not represent real-world hiring performance.
