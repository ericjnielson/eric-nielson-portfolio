"""
Train the college-football margin/total models.

Leakage-free, temporally-validated pipeline:
  * Features come from a pre-game whitelist (see predictor.FEATURES).
  * Train/test split is a true temporal holdout (latest season is the test set).
  * Imputation lives inside an sklearn Pipeline, so it is fit on train only.
  * Models predict game margin (home-away) and total (home+away); scores are
    recovered downstream as home=(total+margin)/2, away=(total-margin)/2.
  * Quantile models give prediction intervals.
  * Results are benchmarked against naive and Elo-only baselines and saved with
    full metadata (versions, seasons, metrics) for reproducibility.

Usage:
    python models/train.py
    python models/train.py --data models/data/processed_games.csv --test-season 2011
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Import the shared feature definitions so train and inference never drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import (  # noqa: E402
    FEATURES,
    MODEL_PATH,
    SCHEMA_VERSION,
    build_feature_frame,
    compute_targets,
)

READABLE = {
    "home_elo": "Home Elo",
    "away_elo": "Away Elo",
    "elo_diff": "Elo Difference",
    "elo_missing": "Elo Availability",
    "home_rest_days": "Home Rest",
    "away_rest_days": "Away Rest",
    "rest_advantage": "Rest Advantage",
    "neutral_site": "Neutral Site",
    "conference_game": "Conference Game",
    "week": "Week",
    "temperature": "Temperature",
    "wind_speed": "Wind",
    "humidity": "Humidity",
    "precipitation": "Precipitation",
}


def make_model(objective: str = "reg:squarederror", **kw) -> Pipeline:
    """An imputer + gradient-boosted regressor, fit only on training data."""
    params = dict(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        objective=objective,
    )
    params.update(kw)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(**params)),
    ])


def metrics(y_true, y_pred) -> dict:
    diff = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    for m in (7, 10, 14):
        out[f"within_{m}"] = float((diff <= m).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train CFB margin/total models.")
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(MODEL_PATH), "..", "data", "processed_games.csv"))
    ap.add_argument("--out", default=MODEL_PATH)
    ap.add_argument("--test-season", type=int, default=None,
                    help="Season to hold out for testing (default: latest).")
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    raw = raw.dropna(subset=["homePoints", "awayPoints", "season"]).reset_index(drop=True)

    X = build_feature_frame(raw)
    y = compute_targets(raw)
    seasons = raw["season"].astype(int)

    test_season = args.test_season or int(seasons.max())
    train_mask = seasons < test_season
    test_mask = seasons == test_season
    if test_mask.sum() == 0:
        raise SystemExit(f"No games found for test season {test_season}.")

    Xtr, Xte = X[train_mask], X[test_mask]
    ytr, yte = y[train_mask], y[test_mask]
    print(f"Train: {len(Xtr)} games (seasons {int(seasons[train_mask].min())}-"
          f"{int(seasons[train_mask].max())})  |  Test: {len(Xte)} games "
          f"(season {test_season})")

    # ---- Core models -----------------------------------------------------
    margin_model = make_model().fit(Xtr, ytr["margin"])
    total_model = make_model().fit(Xtr, ytr["total"])

    margin_pred = margin_model.predict(Xte)
    total_pred = total_model.predict(Xte)
    m_metrics = metrics(yte["margin"], margin_pred)
    t_metrics = metrics(yte["total"], total_pred)

    # Recovered-score accuracy
    home_pred = (total_pred + margin_pred) / 2.0
    away_pred = (total_pred - margin_pred) / 2.0
    home_true = (yte["total"] + yte["margin"]) / 2.0
    away_true = (yte["total"] - yte["margin"]) / 2.0
    score_mae = float((np.abs(home_pred - home_true).mean()
                       + np.abs(away_pred - away_true).mean()) / 2.0)

    # Straight-up winner accuracy (exclude exact ties in truth)
    nontie = yte["margin"] != 0
    winner_acc = float((np.sign(margin_pred[nontie.values])
                        == np.sign(yte["margin"][nontie].values)).mean())

    # ---- Baselines (honest comparison) ----------------------------------
    base_margin = float(ytr["margin"].mean())
    base_total = float(ytr["total"].mean())
    baselines = {
        "naive_mean_margin_mae": float(mean_absolute_error(
            yte["margin"], np.full(len(yte), base_margin))),
        "naive_mean_total_mae": float(mean_absolute_error(
            yte["total"], np.full(len(yte), base_total))),
    }
    # Elo-only linear baseline on margin (uses just the Elo difference).
    elo_tr = Xtr[["elo_diff"]].fillna(0.0)
    elo_te = Xte[["elo_diff"]].fillna(0.0)
    elo_lin = LinearRegression().fit(elo_tr, ytr["margin"])
    baselines["elo_only_margin_mae"] = float(mean_absolute_error(
        yte["margin"], elo_lin.predict(elo_te)))

    # ---- Time-series CV (reported for transparency) ----------------------
    cv = TimeSeriesSplit(n_splits=4)
    Xtr_sorted = Xtr.reset_index(drop=True)
    ytr_sorted = ytr.reset_index(drop=True)
    cv_mae = []
    for tr_idx, va_idx in cv.split(Xtr_sorted):
        mdl = make_model().fit(Xtr_sorted.iloc[tr_idx], ytr_sorted["margin"].iloc[tr_idx])
        cv_mae.append(mean_absolute_error(
            ytr_sorted["margin"].iloc[va_idx], mdl.predict(Xtr_sorted.iloc[va_idx])))
    margin_cv_mae = float(np.mean(cv_mae))

    # ---- Quantile models for prediction intervals -----------------------
    quantile_models = {
        "margin_lo": make_model("reg:quantileerror", quantile_alpha=0.1).fit(Xtr, ytr["margin"]),
        "margin_hi": make_model("reg:quantileerror", quantile_alpha=0.9).fit(Xtr, ytr["margin"]),
        "total_lo": make_model("reg:quantileerror", quantile_alpha=0.1).fit(Xtr, ytr["total"]),
        "total_hi": make_model("reg:quantileerror", quantile_alpha=0.9).fit(Xtr, ytr["total"]),
    }

    # ---- Feature importances (for the UI "weights" panel) ---------------
    importances = margin_model.named_steps["model"].feature_importances_
    imp_pairs = sorted(zip(FEATURES, importances), key=lambda kv: kv[1], reverse=True)[:6]
    total_imp = float(sum(float(v) for _, v in imp_pairs)) or 1.0
    feature_importances = {READABLE.get(k, k): round(float(v) / total_imp, 4)
                           for k, v in imp_pairs}

    # ---- Inference defaults (train medians) -----------------------------
    inference_defaults = {
        "home_rest_days": float(Xtr["home_rest_days"].median()),
        "away_rest_days": float(Xtr["away_rest_days"].median()),
        "week": float(Xtr["week"].median()),
        "temperature": float(Xtr["temperature"].median()),
        "wind_speed": float(Xtr["wind_speed"].median()),
        "humidity": float(Xtr["humidity"].median()),
        "precipitation": float(Xtr["precipitation"].median()),
        "default_elo": float(pd.concat([Xtr["home_elo"], Xtr["away_elo"]]).median()),
    }

    league = {
        "home_avg": float((ytr["total"] + ytr["margin"]).mean() / 2.0),
        "away_avg": float((ytr["total"] - ytr["margin"]).mean() / 2.0),
        "home_edge": base_margin,
    }

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "trained_at": dt.datetime.utcnow().isoformat() + "Z",
        "features": FEATURES,
        "margin_model": margin_model,
        "total_model": total_model,
        "quantile_models": quantile_models,
        "feature_importances": feature_importances,
        "inference_defaults": inference_defaults,
        "league": league,
        "train_seasons": [int(seasons[train_mask].min()), int(seasons[train_mask].max())],
        "test_season": int(test_season),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "metrics": {
            "margin": m_metrics,
            "total": t_metrics,
            "recovered_score_mae": score_mae,
            "winner_accuracy": winner_acc,
            "margin_cv_mae": margin_cv_mae,
            "baselines": baselines,
            "note": "Vegas spread/total columns were empty in the dataset, so "
                    "against-the-spread benchmarking was not possible; baselines "
                    "are naive-mean and Elo-only.",
        },
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(bundle, args.out)

    # ---- Report ----------------------------------------------------------
    print("\n================ RESULTS (temporal holdout) ================")
    print(f"Margin  MAE {m_metrics['mae']:.2f} | RMSE {m_metrics['rmse']:.2f} "
          f"| R2 {m_metrics['r2']:.3f} | within 14: {m_metrics['within_14']*100:.0f}%")
    print(f"Total   MAE {t_metrics['mae']:.2f} | RMSE {t_metrics['rmse']:.2f} "
          f"| R2 {t_metrics['r2']:.3f} | within 14: {t_metrics['within_14']*100:.0f}%")
    print(f"Per-team score MAE: {score_mae:.2f}")
    print(f"Straight-up winner accuracy: {winner_acc*100:.1f}%")
    print(f"Margin TimeSeriesCV MAE: {margin_cv_mae:.2f}")
    print("\nBaselines (margin MAE):")
    print(f"  model      {m_metrics['mae']:.2f}")
    print(f"  elo-only   {baselines['elo_only_margin_mae']:.2f}")
    print(f"  naive mean {baselines['naive_mean_margin_mae']:.2f}")
    print("\nTop feature importances:")
    for k, v in feature_importances.items():
        print(f"  {k:<18} {v*100:5.1f}%")
    print(f"\nSaved bundle -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
