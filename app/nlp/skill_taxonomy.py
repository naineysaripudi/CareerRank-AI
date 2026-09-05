"""Controlled skill vocabulary and common aliases."""

SKILL_ALIASES = {
    "python": "Python", "py": "Python", "sql": "SQL", "java": "Java", "c++": "C++",
    "javascript": "JavaScript", "js": "JavaScript", "machine learning": "Machine Learning",
    "ml": "Machine Learning", "deep learning": "Deep Learning", "dl": "Deep Learning",
    "scikit-learn": "Scikit-learn", "scikit learn": "Scikit-learn", "sklearn": "Scikit-learn",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "nlp": "NLP", "natural language processing": "NLP",
    "llm": "LLM", "rag": "RAG", "generative ai": "Generative AI", "computer vision": "Computer Vision",
    "pandas": "Pandas", "numpy": "NumPy", "power bi": "Power BI", "tableau": "Tableau",
    "aws": "AWS", "azure": "Azure", "gcp": "GCP", "docker": "Docker", "kubernetes": "Kubernetes",
    "mlflow": "MLflow", "git": "Git", "fastapi": "FastAPI", "rest api": "REST APIs",
    "rest apis": "REST APIs", "apache spark": "Apache Spark", "airflow": "Airflow",
}

SKILL_TERMS = sorted(SKILL_ALIASES, key=len, reverse=True)
