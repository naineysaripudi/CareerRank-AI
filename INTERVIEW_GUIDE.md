# Interview Guide

## 30-second explanation
CareerRank AI parses a resume, extracts controlled-vocabulary skills, retrieves semantically relevant jobs with sentence embeddings and FAISS, and reranks them with transparent preference-aware features. It explains matches and skill gaps with deterministic facts, with an optional LLM layer for improved wording.

## Key design choices
- Embeddings capture meaning beyond exact keywords.
- FAISS provides fast local vector retrieval.
- Weighted scoring keeps ranking reproducible and explainable.
- The LLM explains structured facts rather than inventing scores or skills.
- Demo mode works without an API key, solving cold start for new candidates.

## Questions to prepare
1. Why use embeddings? To retrieve semantically related descriptions.
2. Why FAISS? It is fast, local, and simple to persist.
3. Why not use an LLM for ranking? A transparent score is easier to test and audit.
4. How are skills normalized? Alias terms map to canonical taxonomy names.
5. How is personalization implemented? Role, location, experience, employment, and skill features affect the score.
6. What are limitations? Synthetic data, taxonomy coverage, PDF extraction quality, and limited labels.
7. What would you improve? Calibrate weights from labeled feedback and add fairness monitoring.

## Additional questions

8. What is RAG here? Retrieval supplies relevant job context before explanation generation.
9. Why cosine similarity? Normalized vectors make dot product equivalent to cosine similarity.
10. How does cold start work? A new candidate needs only a resume and preferences.
11. What happens with an empty resume? The API returns a validation error and the UI asks for input.
12. What happens when the LLM is unavailable? Deterministic template explanations are returned.
13. How are missing skills found? Candidate canonical skills are compared with required skills.
14. Why use a taxonomy? It makes aliases and comparisons consistent.
15. Why include preferred skills? They provide a softer signal than required skills.
16. Why normalize the score? A 0-100 score is easier to display and explain.
17. How is the index persisted? FAISS stores vectors and a NumPy sidecar stores job IDs.
18. When is the index rebuilt? When it is missing or its job IDs differ from the dataset.
19. What does Precision@5 mean? The fraction of the first five results that are relevant.
20. What does Recall@5 mean? The fraction of all relevant jobs found in the first five.
21. What does MRR measure? How early the first relevant result appears.
22. What does NDCG measure? Ranked relevance with decreasing credit for lower positions.
23. Why use synthetic labels? They make the development experiment reproducible and transparent.
24. Are the evaluation scores production quality? No, they are not a hiring benchmark.
25. What is the baseline? Semantic similarity alone.
26. What is the personalized model? Semantic similarity plus skill and preference features.
27. What is the ML ranker? An optional Random Forest trained on labeled matching features.
28. What data should train it? Human-reviewed candidate-job relevance judgments.
29. How are secrets protected? API keys are read from `.env` and never logged.
30. What personal data reaches the LLM? Only structured matching facts, not unnecessary contact data.
31. What is the main limitation? Resume parsing and taxonomy coverage can miss valid evidence.
32. What is the next production step? Add feedback labels, monitoring, calibration, and fairness tests.
