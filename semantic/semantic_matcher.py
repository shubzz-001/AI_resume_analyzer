import pandas as pd
from typing import List, Dict
import logging

from semantic.embeddings import embed_text, batch_cosine_similarity
from config import JOBS_DIR

logger = logging.getLogger(__name__)


def semantic_recommend_jobs(
        resume_text: str,
        top_n: int = 5,
        threshold: float = 0.5
) -> pd.DataFrame:
    """
    Recommend jobs using semantic similarity.
    """
    jobs_file = JOBS_DIR / "job_descriptions.csv"

    if not jobs_file.exists():
        logger.warning(f"Job descriptions file not found: {jobs_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(jobs_file)

        # Generate embeddings
        resume_vec = embed_text(resume_text)
        job_vecs = embed_text(df["job_description"].tolist())

        # Calculate similarities
        similarities = batch_cosine_similarity(resume_vec, job_vecs)

        # Add to dataframe
        df["semantic_match"] = [round(s * 100, 1) for s in similarities]

        # Filter by threshold
        df = df[df["semantic_match"] >= threshold * 100]

        # Sort and return top N
        return df.sort_values("semantic_match", ascending=False).head(top_n)

    except Exception as e:
        logger.error(f"Error in semantic job matching: {str(e)}")
        return pd.DataFrame()


def semantic_recommend_jobs_v2(
        resume_text: str,
        top_n: int = 3,
        threshold: float = 0.5
) -> List[Dict]:
    """
    Enhanced semantic job recommendations with explanations.
    """
    df = semantic_recommend_jobs(resume_text, top_n, threshold)

    if df.empty:
        return []

    results = []
    for _, row in df.iterrows():
        results.append({
            "job_title": row.get("job_title", "Unknown"),
            "semantic_match": row.get("semantic_match", 0),
            "explanation": f"Strong semantic match based on skills and experience"
        })

    return results