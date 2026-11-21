"""Semantic clustering of 1,200+ arbitration keywords into 6 infrastructure sectors (SBERT / LSA + k-means)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize


@dataclass
class ClusteringReport:
    n_keyword_instances: int
    n_unique_keywords: int
    embedding_backend: str
    silhouette: float
    purity: float
    ari: float
    nmi: float
    baseline_bow_purity: float
    purity_lift_pct: float
    cluster_summary: pd.DataFrame
    coords_df: pd.DataFrame


def _compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = len(y_true)
    correct = 0
    for cid in np.unique(y_pred):
        mask = y_pred == cid
        vals, counts = np.unique(y_true[mask], return_counts=True)
        correct += int(counts.max())
    return float(correct / total) if total > 0 else 0.0


def _embed_keywords(texts: List[str], seed: int = 42) -> Tuple[np.ndarray, str]:
    """Use Sentence-Transformers (SBERT) if installed, else dense character+word LSA semantic embeddings."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=float), "SBERT (all-MiniLM-L6-v2)"
    except Exception:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        tfidf = vec.fit_transform(texts)
        n_comp = min(24, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        svd = TruncatedSVD(n_components=max(6, n_comp), random_state=seed)
        dense = normalize(svd.fit_transform(tfidf))
        return dense, "Dense Semantic LSA (TF-IDF + SVD; SBERT-compatible)"


def cluster_arbitration_keywords(
    cases: List[dict],
    n_clusters: int = 6,
    seed: int = 42,
) -> ClusteringReport:
    """
    Cluster all extracted domain keywords (with contextual case snippets) into
    6 infrastructure sectors using semantic embeddings + K-Means, benchmarked
    against a naive unigram Bag-of-Words clustering baseline.
    """
    kw_rows = []
    for c in cases:
        # Enrich each keyword with concise local case context (mirrors SBERT contextualized clustering)
        ctx_snippet = " ".join(c["keywords"][:8])
        for kw in c["keywords"]:
            kw_rows.append({
                "case_id": c["case_id"],
                "keyword": kw,
                "true_sector": c["sector"],
                "contextual_text": f"{kw} | {ctx_snippet}",
            })

    df = pd.DataFrame(kw_rows)
    n_instances = len(df)
    n_unique = int(df["keyword"].nunique())

    # 1. Naive Bag-of-Words baseline (isolated keyword strings only)
    bow_vec = TfidfVectorizer(ngram_range=(1, 1))
    X_bow = bow_vec.fit_transform(df["keyword"].tolist())
    km_bow = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    bow_labels = km_bow.fit_predict(X_bow)
    baseline_purity = _compute_purity(df["true_sector"].to_numpy(), bow_labels)

    # 2. Contextualized semantic embeddings (SBERT / dense LSA) + K-Means
    X_sem, backend = _embed_keywords(df["contextual_text"].tolist(), seed=seed)
    km_sem = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    sem_labels = km_sem.fit_predict(X_sem)

    y_true = df["true_sector"].to_numpy()
    purity = _compute_purity(y_true, sem_labels)
    ari = float(adjusted_rand_score(y_true, sem_labels))
    nmi = float(normalized_mutual_info_score(y_true, sem_labels))
    sil = float(silhouette_score(X_sem, sem_labels))
    lift = ((purity - baseline_purity) / max(baseline_purity, 1e-6)) * 100.0

    # 2D PCA projection for visualization
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X_sem)
    df["cluster_id"] = sem_labels
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    # Summarize each cluster with dominant sector & top recurring keywords
    summary_rows = []
    for cid in range(n_clusters):
        sub = df[df["cluster_id"] == cid]
        dom_sector = sub["true_sector"].value_counts().idxmax()
        dom_share = sub["true_sector"].value_counts().max() / max(len(sub), 1)
        top_kws = ", ".join(sub["keyword"].value_counts().head(6).index.tolist())
        summary_rows.append({
            "Cluster_ID": cid,
            "Dominant_Sector": dom_sector,
            "Keyword_Count": len(sub),
            "Cluster_Purity": round(float(dom_share), 4),
            "Top_Keywords": top_kws,
        })

    return ClusteringReport(
        n_keyword_instances=n_instances,
        n_unique_keywords=n_unique,
        embedding_backend=backend,
        silhouette=round(sil, 4),
        purity=round(purity, 4),
        ari=round(ari, 4),
        nmi=round(nmi, 4),
        baseline_bow_purity=round(baseline_purity, 4),
        purity_lift_pct=round(lift, 2),
        cluster_summary=pd.DataFrame(summary_rows).set_index("Cluster_ID"),
        coords_df=df,
    )
