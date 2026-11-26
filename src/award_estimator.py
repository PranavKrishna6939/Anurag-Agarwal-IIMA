"""Multi-model 5-fold cross-validated arbitration award prediction engine (INR Crores)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class AwardEstimationReport:
    model_metrics: pd.DataFrame
    best_model_name: str
    best_mae_cr: float
    predictions_df: pd.DataFrame
    feature_importance: pd.Series


def build_feature_table(cases: List[dict]) -> pd.DataFrame:
    """Assemble structured tabular modeling features from extracted arbitration case records."""
    rows = []
    for c in cases:
        claim = float(c["claim_amount_cr"])
        counter = float(c["counterclaim_cr"])
        rows.append({
            "case_id": c["case_id"],
            "sector": c["sector"],
            "clause_type": c["clause_type"],
            "forum": c["forum"],
            "claim_amount_cr": claim,
            "counterclaim_cr": counter,
            "counterclaim_ratio": counter / max(claim, 1.0),
            "delay_months": float(c["delay_months"]),
            "respondent_delay_share": float(c["respondent_delay_share"]),
            "interest_rate": float(c["interest_rate"]),
            "has_escalation_clause": int(c["has_escalation_clause"]),
            "tribunal_unanimous": int(c["tribunal_unanimous"]),
            "evidence_strength": float(c["evidence_strength"]),
            # Interaction terms reflecting arbitral quantum mechanics
            "claim_x_delay_share": claim * float(c["respondent_delay_share"]),
            "claim_x_evidence": claim * float(c["evidence_strength"]),
            "award_amount_cr": float(c["award_amount_cr"]),
        })
    return pd.DataFrame(rows)


def evaluate_award_estimators(
    cases: List[dict],
    cv_folds: int = 5,
    seed: int = 42,
) -> AwardEstimationReport:
    """
    Benchmark multiple regression architectures via strict 5-fold cross-validation
    to predict net arbitral award amounts (INR Crores).
    """
    df = build_feature_table(cases)
    y = df["award_amount_cr"].to_numpy(dtype=float)

    num_cols = [
        "claim_amount_cr",
        "counterclaim_cr",
        "counterclaim_ratio",
        "delay_months",
        "respondent_delay_share",
        "interest_rate",
        "has_escalation_clause",
        "tribunal_unanimous",
        "evidence_strength",
        "claim_x_delay_share",
        "claim_x_evidence",
    ]
    cat_cols = ["sector", "clause_type"]

    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    models = {
        "Linear_Claim_Baseline": Pipeline([
            ("pre", ColumnTransformer([("claim", StandardScaler(), ["claim_amount_cr"])])),
            ("reg", LinearRegression()),
        ]),
        "Ridge_Structured_RAG": Pipeline([
            ("pre", preproc),
            ("reg", Ridge(alpha=5.0, random_state=seed)),
        ]),
        "RandomForest_RAG": Pipeline([
            ("pre", preproc),
            ("reg", RandomForestRegressor(n_estimators=220, max_depth=6, random_state=seed)),
        ]),
        "GradientBoosting_RAG": Pipeline([
            ("pre", preproc),
            ("reg", GradientBoostingRegressor(n_estimators=140, max_depth=3, learning_rate=0.05, random_state=seed)),
        ]),
        "Stacked_Ensemble_RAG": Pipeline([
            ("pre", preproc),
            (
                "reg",
                StackingRegressor(
                    estimators=[
                        ("ridge", Ridge(alpha=5.0, random_state=seed)),
                        ("rf", RandomForestRegressor(n_estimators=180, max_depth=6, random_state=seed)),
                        ("gbr", GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=seed)),
                    ],
                    final_estimator=Ridge(alpha=2.0),
                    cv=4,
                ),
            ),
        ]),
    }

    X = df[num_cols + cat_cols]
    metric_rows = []
    oof_preds: Dict[str, np.ndarray] = {}

    for name, pipe in models.items():
        pred = np.clip(cross_val_predict(pipe, X, y, cv=kf), 0.0, None)
        oof_preds[name] = pred
        mae = float(mean_absolute_error(y, pred))
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        medae = float(median_absolute_error(y, pred))
        r2 = float(r2_score(y, pred))
        metric_rows.append({
            "Model": name,
            "MAE_INR_Cr": round(mae, 2),
            "RMSE_INR_Cr": round(rmse, 2),
            "MedAE_INR_Cr": round(medae, 2),
            "R2": round(r2, 4),
        })

    metrics_df = pd.DataFrame(metric_rows).set_index("Model").sort_values("MAE_INR_Cr")
    best_model = str(metrics_df.index[0])
    best_mae = float(metrics_df.loc[best_model, "MAE_INR_Cr"])

    # Fit RandomForest on full dataset to inspect interpretable feature importances
    rf_pipe = models["RandomForest_RAG"].fit(X, y)
    feat_names = num_cols + list(rf_pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(cat_cols))
    importances = pd.Series(rf_pipe.named_steps["reg"].feature_importances_, index=feat_names).sort_values(ascending=False)

    pred_df = pd.DataFrame({
        "case_id": df["case_id"],
        "sector": df["sector"],
        "claim_amount_cr": df["claim_amount_cr"],
        "actual_award_cr": y,
        "predicted_award_cr": np.round(oof_preds[best_model], 2),
        "abs_error_cr": np.round(np.abs(y - oof_preds[best_model]), 2),
        "baseline_pred_cr": np.round(oof_preds["Linear_Claim_Baseline"], 2),
    })

    return AwardEstimationReport(
        model_metrics=metrics_df,
        best_model_name=best_model,
        best_mae_cr=best_mae,
        predictions_df=pred_df,
        feature_importance=importances,
    )
