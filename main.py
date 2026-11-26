"""End-to-end runner: Arbitration Corpus -> RAG-NER -> 6-Sector Semantic Clustering -> Award Estimation."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ArbitrationConfig
from src.award_estimator import evaluate_award_estimators
from src.dataset_builder import build_arbitration_corpus
from src.rag_ner import evaluate_rag_ner
from src.sector_clustering import cluster_arbitration_keywords


SECTOR_COLORS = {
    "Highways & Roads": "#1E3A8A",
    "Railways & Metro": "#0F766E",
    "Power & Energy": "#D97706",
    "Ports & Maritime": "#7C3AED",
    "Aviation & Airports": "#DC2626",
    "Water & Urban EPC": "#0284C7",
}


def setup_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FAF9F6",
        "axes.edgecolor": "#D1D5DB",
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.size": 10,
        "axes.titlesize": 12.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.2,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#E5E7EB",
    })


def plot_award_diagnostics(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, best_name: str, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.1))

    # Left: Actual vs Out-of-Fold Predicted Award (INR Cr)
    for sec, grp in pred_df.groupby("sector"):
        ax1.scatter(
            grp["actual_award_cr"],
            grp["predicted_award_cr"],
            label=sec,
            color=SECTOR_COLORS.get(sec, "#334155"),
            s=55,
            alpha=0.86,
            edgecolors="white",
            linewidth=0.6,
        )
    lim = max(pred_df["actual_award_cr"].max(), pred_df["predicted_award_cr"].max()) * 1.06
    ax1.plot([0, lim], [0, lim], color="#111827", ls="--", lw=1.3, label="Perfect Prediction (y = x)")
    best_mae = metrics_df.loc[best_name, "MAE_INR_Cr"]
    best_r2 = metrics_df.loc[best_name, "R2"]
    ax1.set_title(f"Out-of-Fold Award Prediction ({best_name})\nMAE = INR {best_mae:.2f} Cr | R² = {best_r2:.3f}")
    ax1.set_xlabel("Actual Arbitral Award (INR Crore)")
    ax1.set_ylabel("Predicted Arbitral Award (INR Crore)")
    ax1.legend(loc="upper left", fontsize=7.8, ncol=1)

    # Right: Model Benchmark MAE Comparison
    m_sorted = metrics_df.sort_values("MAE_INR_Cr", ascending=False)
    colors = ["#0F766E" if idx == best_name else "#2B4485" for idx in m_sorted.index]
    bars = ax2.barh(m_sorted.index, m_sorted["MAE_INR_Cr"], color=colors, alpha=0.88)
    ax2.set_title("5-Fold Cross-Validation MAE Comparison")
    ax2.set_xlabel("Mean Absolute Error (INR Crore, Lower is Better)")
    for bar, val in zip(bars, m_sorted["MAE_INR_Cr"]):
        ax2.text(val + 0.4, bar.get_y() + bar.get_height() / 2, f"{val:.2f} Cr", va="center", fontsize=8.8, fontweight="bold")
    ax2.set_xlim(0, m_sorted["MAE_INR_Cr"].max() * 1.24)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sector_clusters(coords_df: pd.DataFrame, purity: float, sil: float, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    for sec, grp in coords_df.groupby("true_sector"):
        ax.scatter(
            grp["pc1"],
            grp["pc2"],
            label=f"{sec} ({len(grp)})",
            color=SECTOR_COLORS.get(sec, "#334155"),
            s=28,
            alpha=0.72,
            edgecolors="none",
        )
    ax.set_title(f"Semantic Clustering of {len(coords_df):,} Arbitration Keywords across 6 Infrastructure Sectors\nCluster Purity = {purity*100:.1f}% | Silhouette = {sil:.3f}")
    ax.set_xlabel("Semantic Principal Component 1")
    ax.set_ylabel("Semantic Principal Component 2")
    ax.legend(loc="best", fontsize=8.4, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ner_breakdown(per_type: pd.DataFrame, overall_f1: float, total_ents: int, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    df_p = per_type.sort_values("F1", ascending=True)
    y = np.arange(len(df_p))
    ax.barh(y - 0.18, df_p["Precision"], height=0.34, color="#2B4485", alpha=0.88, label="Precision")
    ax.barh(y + 0.18, df_p["Recall"], height=0.34, color="#0F766E", alpha=0.88, label="Recall")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{idx} (n={int(row['Support'])})" for idx, row in df_p.iterrows()])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score (0.0 – 1.0)")
    ax.set_title(f"Hybrid RAG-NER Entity Extraction Performance ({total_ents:,} Entities | Micro-F1 = {overall_f1:.3f})")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sector_recovery_and_importance(pred_df: pd.DataFrame, feat_imp: pd.Series, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    rec = pred_df.copy()
    rec["recovery_pct"] = (rec["actual_award_cr"] / rec["claim_amount_cr"]) * 100.0
    sec_avg = rec.groupby("sector")["recovery_pct"].mean().sort_values()
    colors = [SECTOR_COLORS.get(s, "#2B4485") for s in sec_avg.index]
    bars = ax1.barh(sec_avg.index, sec_avg.values, color=colors, alpha=0.88)
    ax1.set_title("Mean Claim-to-Award Recovery Ratio by Sector")
    ax1.set_xlabel("Award / Claim Recovery (%)")
    ax1.set_xlim(0, max(sec_avg.values) + 14)
    for b, v in zip(bars, sec_avg.values):
        ax1.text(v + 1.0, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.8, fontweight="bold")

    top_imp = feat_imp.head(9).iloc[::-1]
    ax2.barh(top_imp.index, top_imp.values * 100.0, color="#1E3A8A", alpha=0.88)
    ax2.set_title("Top Drivers of Arbitral Award Magnitude (RF Importance)")
    ax2.set_xlabel("Relative Feature Importance (%)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian Infrastructure Arbitration Award Estimation using RAG")
    parser.add_argument("--rebuild-corpus", action="store_true", help="Regenerate cached benchmark corpus")
    parser.add_argument("--use-deepseek", action="store_true", help="Use live DeepSeek API for RAG-NER (requires DEEPSEEK_API_KEY)")
    args = parser.parse_args()

    cfg = ArbitrationConfig()
    os.makedirs("results/figures", exist_ok=True)
    setup_plot_style()

    print("1. Loading Indian infrastructure arbitration corpus...")
    cases = build_arbitration_corpus(
        n_cases=cfg.n_cases,
        seed=cfg.random_seed,
        force_rebuild=args.rebuild_corpus,
    )
    print(f"   Loaded {len(cases)} arbitration cases across {len(cfg.sectors)} infrastructure sectors.")

    print("2. Running Hybrid RAG-NER entity extraction & span evaluation...")
    ner_rep = evaluate_rag_ner(cases, use_deepseek=args.use_deepseek)
    ner_rep.per_type.to_csv("results/ner_metrics_by_type.csv")
    ner_rep.extracted_df.to_csv("results/extracted_entities.csv", index=False)
    print(f"   Extracted {ner_rep.total_predicted:,} entities | Micro-F1 = {ner_rep.overall_f1:.4f}")

    print("3. Clustering arbitration domain keywords into 6 infrastructure sectors...")
    cl_rep = cluster_arbitration_keywords(cases, n_clusters=cfg.n_clusters, seed=cfg.random_seed)
    cl_rep.cluster_summary.to_csv("results/sector_clusters.csv")
    print(
        f"   Clustered {cl_rep.n_keyword_instances:,} keywords ({cl_rep.n_unique_keywords} unique) | "
        f"Purity = {cl_rep.purity*100:.1f}% (+{cl_rep.purity_lift_pct:.1f}% vs BoW baseline {cl_rep.baseline_bow_purity*100:.1f}%)"
    )

    print("4. Evaluating 5-fold cross-validated arbitration award estimators...")
    aw_rep = evaluate_award_estimators(cases, cv_folds=cfg.cv_folds, seed=cfg.random_seed)
    aw_rep.model_metrics.to_csv("results/award_model_metrics.csv")
    aw_rep.predictions_df.to_csv("results/award_predictions.csv", index=False)

    summary = {
        "n_cases": len(cases),
        "n_sectors": len(cfg.sectors),
        "ner": {
            "total_gold_entities": ner_rep.total_gold,
            "total_extracted_entities": ner_rep.total_predicted,
            "micro_precision": ner_rep.overall_precision,
            "micro_recall": ner_rep.overall_recall,
            "micro_f1": ner_rep.overall_f1,
        },
        "clustering": {
            "embedding_backend": cl_rep.embedding_backend,
            "keyword_instances": cl_rep.n_keyword_instances,
            "unique_keywords": cl_rep.n_unique_keywords,
            "silhouette": cl_rep.silhouette,
            "purity": cl_rep.purity,
            "baseline_bow_purity": cl_rep.baseline_bow_purity,
            "purity_lift_pct": cl_rep.purity_lift_pct,
            "ari": cl_rep.ari,
            "nmi": cl_rep.nmi,
        },
        "award_estimation": {
            "best_model": aw_rep.best_model_name,
            "best_cv_mae_inr_cr": aw_rep.best_mae_cr,
            "models": aw_rep.model_metrics.to_dict(orient="index"),
        },
    }
    with open("results/summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("5. Generating research figures...")
    plot_award_diagnostics(aw_rep.predictions_df, aw_rep.model_metrics, aw_rep.best_model_name, "results/figures/01_award_prediction_diagnostics.png")
    plot_sector_clusters(cl_rep.coords_df, cl_rep.purity, cl_rep.silhouette, "results/figures/02_sector_clusters_pca.png")
    plot_ner_breakdown(ner_rep.per_type, ner_rep.overall_f1, ner_rep.total_predicted, "results/figures/03_ner_performance_breakdown.png")
    plot_sector_recovery_and_importance(aw_rep.predictions_df, aw_rep.feature_importance, "results/figures/04_sector_award_recovery.png")

    print("\n=== 5-FOLD CV AWARD PREDICTION SUMMARY (INR CRORES) ===")
    print(aw_rep.model_metrics.to_string())


if __name__ == "__main__":
    main()
