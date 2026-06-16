"""
College Football Score Prediction — inference module.

This module loads a leakage-free, temporally-validated model trained by
``models/train.py`` and exposes a UI-friendly ``predict_score`` function that
preserves the JSON contract the Flask app and front-end expect.

Design notes
------------
* The model predicts game **margin** (home - away) and **total** (home + away)
  directly, then recovers the two scores:  home = (total + margin) / 2,
  away = (total - margin) / 2.  This is easier to calibrate and benchmark than
  two independent score regressors.
* Features are restricted to a **pre-game whitelist** (Elo ratings known before
  kickoff, schedule/rest context, and weather).  The rich rolling/EPA columns in
  the source CSV were excluded because they were pre-computed *with look-ahead
  leakage* (no ``shift(1)``) and the raw per-game values needed to recompute them
  correctly are not present in the processed dataset.
* Loading is **lazy and cached** — importing this module has no side effects.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# =========================================================================
# Paths / schema
# =========================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(CURRENT_DIR, "saved")
DATA_PATH = os.path.join(CURRENT_DIR, "data", "processed_games.csv")
MODEL_PATH = os.path.join(SAVE_DIR, "cfb_model.joblib")

SCHEMA_VERSION = 2

# The ONLY columns the model is allowed to see. Every one of these is knowable
# before kickoff, so there is no target leakage.
FEATURES: List[str] = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_missing",
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "neutral_site",
    "conference_game",
    "week",
    "temperature",
    "wind_speed",
    "humidity",
    "precipitation",
]

# Sensible fallbacks used when a column is entirely absent. Real medians are
# computed at train time and stored in the model bundle (``inference_defaults``).
_FALLBACK_DEFAULTS: Dict[str, float] = {
    "home_rest_days": 7.0,
    "away_rest_days": 7.0,
    "week": 7.0,
    "temperature": 65.0,
    "wind_speed": 6.0,
    "humidity": 60.0,
    "precipitation": 0.0,
    "default_elo": 1500.0,
}


# =========================================================================
# Feature engineering — single source of truth for train AND inference
# =========================================================================
def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw game rows into the leakage-free feature matrix.

    Used by both ``train.py`` (on historical games) and inference (on a single
    synthetic matchup row), guaranteeing identical feature construction.
    """
    df = raw.copy()
    out = pd.DataFrame(index=df.index)

    home_elo = pd.to_numeric(df.get("homeStartElo"), errors="coerce")
    away_elo = pd.to_numeric(df.get("awayStartElo"), errors="coerce")
    out["home_elo"] = home_elo
    out["away_elo"] = away_elo
    out["elo_diff"] = home_elo - away_elo
    out["elo_missing"] = (home_elo.isna() | away_elo.isna()).astype(int)

    home_rest = pd.to_numeric(df.get("home_rest_days"), errors="coerce")
    away_rest = pd.to_numeric(df.get("away_rest_days"), errors="coerce")
    out["home_rest_days"] = home_rest
    out["away_rest_days"] = away_rest
    out["rest_advantage"] = home_rest - away_rest

    out["neutral_site"] = _as_int(df.get("neutralSite"))
    out["conference_game"] = _as_int(df.get("conferenceGame"))
    out["week"] = pd.to_numeric(df.get("week"), errors="coerce")

    out["temperature"] = pd.to_numeric(df.get("temperature"), errors="coerce")
    out["wind_speed"] = pd.to_numeric(df.get("windSpeed"), errors="coerce")
    out["humidity"] = pd.to_numeric(df.get("humidity"), errors="coerce")
    out["precipitation"] = pd.to_numeric(df.get("precipitation"), errors="coerce")

    return out[FEATURES]


def compute_targets(raw: pd.DataFrame) -> pd.DataFrame:
    """Return margin (home-away) and total (home+away) targets."""
    home = pd.to_numeric(raw["homePoints"], errors="coerce")
    away = pd.to_numeric(raw["awayPoints"], errors="coerce")
    return pd.DataFrame({"margin": home - away, "total": home + away})


def _as_int(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(0, dtype=int)
    return (
        pd.Series(series).map({True: 1, False: 0, "true": 1, "false": 0})
        .fillna(pd.to_numeric(series, errors="coerce"))
        .fillna(0)
        .astype(int)
    )


# =========================================================================
# Lazy, cached loaders (no import-time side effects)
# =========================================================================
@lru_cache(maxsize=1)
def get_games_df() -> pd.DataFrame:
    """Load and lightly clean the historical games used for team lookups."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    if "startDate" in df.columns:
        df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def get_model_bundle() -> Optional[Dict[str, Any]]:
    """Load the trained model bundle (or ``None`` if it has not been built)."""
    if not os.path.exists(MODEL_PATH):
        print(
            f"[predictor] No model found at {MODEL_PATH}. "
            "Run `python models/train.py` to build it."
        )
        return None
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _team_elo_state() -> Dict[str, float]:
    """Most recent pre-game Elo for each team, for as-of inference."""
    df = get_games_df()
    if df.empty:
        return {}

    rows = []
    if {"homeTeam", "homeStartElo"}.issubset(df.columns):
        rows.append(df[["homeTeam", "homeStartElo", "startDate"]].rename(
            columns={"homeTeam": "team", "homeStartElo": "elo"}))
    if {"awayTeam", "awayStartElo"}.issubset(df.columns):
        rows.append(df[["awayTeam", "awayStartElo", "startDate"]].rename(
            columns={"awayTeam": "team", "awayStartElo": "elo"}))
    if not rows:
        return {}

    long = pd.concat(rows, ignore_index=True).dropna(subset=["team", "elo"])
    if "startDate" in long.columns:
        long = long.sort_values("startDate")
    latest = long.groupby("team")["elo"].last()
    return latest.to_dict()


def _default(bundle: Optional[Dict[str, Any]], key: str) -> float:
    if bundle and key in bundle.get("inference_defaults", {}):
        return float(bundle["inference_defaults"][key])
    return float(_FALLBACK_DEFAULTS[key])


def _team_conference(df: pd.DataFrame, team: str) -> Optional[str]:
    if "home_conference" in df.columns:
        m = df.loc[df.get("homeTeam") == team, "home_conference"].dropna()
        if not m.empty:
            return m.iloc[-1]
    if "away_conference" in df.columns:
        m = df.loc[df.get("awayTeam") == team, "away_conference"].dropna()
        if not m.empty:
            return m.iloc[-1]
    return None


# =========================================================================
# Prediction
# =========================================================================
def predict_score(
    home_team: str,
    away_team: str,
    df: Optional[pd.DataFrame] = None,
    season: Optional[int] = None,
) -> Dict[str, Any]:
    """Predict a matchup and return a UI-friendly dict.

    The output shape matches what ``app.py`` / ``predictions.js`` consume:
    ``scores``, ``prediction`` (favorite/underdog/spread/total), ``factors`` and
    ``weights`` (repurposed to model feature importances).
    """
    bundle = get_model_bundle()
    if bundle is None:
        return _error("Model is not available. Run models/train.py to build it.")

    if df is None:
        df = get_games_df()

    elo_state = _team_elo_state()
    default_elo = _default(bundle, "default_elo")
    home_elo = elo_state.get(home_team, np.nan)
    away_elo = elo_state.get(away_team, np.nan)
    low_confidence = bool(pd.isna(home_elo) or pd.isna(away_elo))

    h_elo = default_elo if pd.isna(home_elo) else float(home_elo)
    a_elo = default_elo if pd.isna(away_elo) else float(away_elo)

    same_conf = (
        _team_conference(df, home_team) is not None
        and _team_conference(df, home_team) == _team_conference(df, away_team)
    )

    # Build a single synthetic, pre-game matchup row using as-of team state and
    # neutral defaults for unknowable game-day context.
    synthetic = pd.DataFrame([{
        "homeStartElo": h_elo,
        "awayStartElo": a_elo,
        "home_rest_days": _default(bundle, "home_rest_days"),
        "away_rest_days": _default(bundle, "away_rest_days"),
        "neutralSite": False,
        "conferenceGame": same_conf,
        "week": _default(bundle, "week"),
        "temperature": _default(bundle, "temperature"),
        "windSpeed": _default(bundle, "wind_speed"),
        "humidity": _default(bundle, "humidity"),
        "precipitation": _default(bundle, "precipitation"),
    }])
    X = build_feature_frame(synthetic)
    X.loc[:, "elo_missing"] = int(low_confidence)

    margin = float(bundle["margin_model"].predict(X)[0])
    total = float(bundle["total_model"].predict(X)[0])

    home_score = (total + margin) / 2.0
    away_score = (total - margin) / 2.0
    home_score = max(0.0, home_score)
    away_score = max(0.0, away_score)

    # Prediction intervals from quantile models (if present).
    intervals = _intervals(bundle, X, home_score, away_score)

    raw_spread = home_score - away_score
    if raw_spread >= 0:
        favorite, underdog, spread = home_team, away_team, -raw_spread
    else:
        favorite, underdog, spread = away_team, home_team, raw_spread

    league_home_avg = bundle.get("league", {}).get("home_avg", 28.0)
    league_away_avg = bundle.get("league", {}).get("away_avg", 24.0)

    return {
        "error": None,
        "scores": {
            "home": {
                "team": home_team,
                "score": round(home_score, 1),
                "stats": {
                    "avg_score": round(float(league_home_avg), 1),
                    "elo": round(h_elo, 0),
                    "score_low": intervals["home_low"],
                    "score_high": intervals["home_high"],
                },
            },
            "away": {
                "team": away_team,
                "score": round(away_score, 1),
                "stats": {
                    "avg_score": round(float(league_away_avg), 1),
                    "elo": round(a_elo, 0),
                    "score_low": intervals["away_low"],
                    "score_high": intervals["away_high"],
                },
            },
        },
        "prediction": {
            "favorite": favorite,
            "underdog": underdog,
            "spread": round(float(spread), 1),
            "total": round(float(home_score + away_score), 1),
            "margin_range": intervals["margin_range"],
        },
        "factors": {
            "elo_difference": round(h_elo - a_elo, 0),
            "home_field": round(float(bundle.get("league", {}).get("home_edge", 0.0)), 1),
            "same_conference": bool(same_conf),
        },
        # Repurposed: real model feature importances, normalized for display.
        "weights": bundle.get("feature_importances", {}),
        "low_confidence": low_confidence,
    }


def _intervals(bundle, X, home_score, away_score) -> Dict[str, Any]:
    """Translate margin/total quantiles into per-score ranges."""
    q = bundle.get("quantile_models")
    if not q:
        return {
            "home_low": None, "home_high": None,
            "away_low": None, "away_high": None,
            "margin_range": None,
        }
    m_lo = float(q["margin_lo"].predict(X)[0])
    m_hi = float(q["margin_hi"].predict(X)[0])
    t_lo = float(q["total_lo"].predict(X)[0])
    t_hi = float(q["total_hi"].predict(X)[0])
    return {
        "home_low": round(max(0.0, (t_lo + m_lo) / 2.0), 1),
        "home_high": round(max(0.0, (t_hi + m_hi) / 2.0), 1),
        "away_low": round(max(0.0, (t_lo - m_hi) / 2.0), 1),
        "away_high": round(max(0.0, (t_hi - m_lo) / 2.0), 1),
        "margin_range": [round(m_lo, 1), round(m_hi, 1)],
    }


def get_model_metrics() -> Dict[str, Any]:
    """Return validation metrics for the /api/model-info endpoint."""
    bundle = get_model_bundle()
    if bundle is None:
        return {"error": "Model not available."}
    return {
        "schema_version": bundle.get("schema_version"),
        "trained_at": bundle.get("trained_at"),
        "train_seasons": bundle.get("train_seasons"),
        "test_season": bundle.get("test_season"),
        "metrics": bundle.get("metrics"),
        "library_versions": bundle.get("library_versions"),
    }


def _error(msg: str) -> Dict[str, Any]:
    return {"scores": None, "prediction": None, "factors": None,
            "weights": None, "error": msg}


__all__ = [
    "FEATURES",
    "build_feature_frame",
    "compute_targets",
    "predict_score",
    "get_games_df",
    "get_model_bundle",
    "get_model_metrics",
    "MODEL_PATH",
    "SCHEMA_VERSION",
]
