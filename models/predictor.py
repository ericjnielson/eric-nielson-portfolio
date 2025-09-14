"""
College Football Score Prediction Module (cleaned & aligned to notebook)
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================
# Constants / defaults
# =========================
RANDOM_STATE = 42

# Globals populated by initialize_model()
games_df: pd.DataFrame | None = None
model_info: Dict[str, Any] | None = None
home_encoders: Dict[str, Any] | None = None


# =========================
# Utility + preprocessing
# =========================
def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light missing-data handling aligned with the notebook output.
    Uses home_/away_ conference names and weather columns if present.
    """
    df = df.copy()

    # Scores
    for col in ["homePoints", "awayPoints"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

    # Conferences
    for col in ["home_conference", "away_conference"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Team metrics (EPA etc.)
    team_metric_cols = [c for c in df.columns if any(x in c for x in ["epa", "success", "explosiveness"])]
    for col in team_metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if "home_conference" in df.columns:
            df[col] = df.groupby("home_conference")[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())

    # Weather
    for col in ["temperature", "humidity", "windSpeed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if set(["season", "week"]).issubset(df.columns):
                df[col] = df.groupby(["season", "week"])[col].transform(lambda x: x.fillna(x.mean()))
            df[col] = df[col].fillna(df[col].mean())

    # Ratings (expect legacy unsuffixed or normalized ones to be present from your notebook)
    for col in ["elo", "fpi", "spOverall", "srs", "home_elo", "away_elo", "home_fpi", "away_fpi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

    # Context
    for col in ["home_rest_days", "away_rest_days", "attendance"]:
        if col in df.columns:
            if col != "attendance":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Create rest_days (difference) if your UI code expects it
    if {"home_rest_days", "away_rest_days"}.issubset(df.columns) and "rest_days" not in df.columns:
        df["rest_days"] = df["home_rest_days"] - df["away_rest_days"]

    return df


def add_team_performance_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple rolling/context features that don't assume unavailable columns."""
    if df.empty:
        return df.copy()

    out = df.copy()
    for team_type in ["home", "away"]:
        team_id_col = f"{team_type}TeamId"
        points_col = f"{team_type}Points"
        if team_id_col not in out.columns or points_col not in out.columns:
            continue

        out[points_col] = pd.to_numeric(out[points_col], errors="coerce")

        # Expanding mean of scoring
        out[f"{team_type}_historical_scoring"] = (
            out.groupby(team_id_col)[points_col].transform(lambda x: x.expanding().mean())
        )

        # Last 3/5 game rolling mean
        for window in [3, 5]:
            out[f"{team_type}_last{window}_avg"] = out.groupby(team_id_col)[points_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        # Season avg/std
        if "season" in out.columns:
            out[f"{team_type}_season_avg"] = out.groupby([team_id_col, "season"])[points_col].transform("mean")
            out[f"{team_type}_season_std"] = out.groupby([team_id_col, "season"])[points_col].transform("std")

        # Win/loss indicator
        opp = "away" if team_type == "home" else "home"
        opp_points = f"{opp}Points"
        if opp_points in out.columns:
            out[f"{team_type}_won"] = (out[points_col] > out[opp_points]).astype(int)

        # Conference win %
        conf_col = f"{team_type}_conference"
        if conf_col in out.columns and f"{team_type}_won" in out.columns and "season" in out.columns:
            out[f"{team_type}_conf_win_pct"] = out.groupby(["season", conf_col])[f"{team_type}_won"].transform("mean")

    return out


def add_enhanced_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "startDate" in out.columns:
        out["startDate"] = pd.to_datetime(out["startDate"], errors="coerce")

    if {"season", "homeTeamId", "week"}.issubset(out.columns):
        out["games_into_season"] = out.groupby(["season", "homeTeamId"]).cumcount()
        max_week = out.groupby("season")["week"].transform("max")
        out["games_remaining"] = max_week - out["week"]
        out["season_progress"] = out["week"] / max_week.replace(0, np.nan)

    # rest_days created in handle_missing_data
    if {"home_rest_days", "away_rest_days"}.issubset(out.columns):
        out["rest_advantage"] = out["home_rest_days"] - out["away_rest_days"]
        out["total_rest"] = out["home_rest_days"] + out["away_rest_days"]
        out["home_short_rest"] = (out["home_rest_days"] < 6).astype(int)
        out["away_short_rest"] = (out["away_rest_days"] < 6).astype(int)
        out["home_long_rest"] = (out["home_rest_days"] > 8).astype(int)
        out["away_long_rest"] = (out["away_rest_days"] > 8).astype(int)

    # excitement-based flags if available
    if "excitement" in out.columns and {"homeTeamId", "awayTeamId"}.issubset(out.columns):
        grp = out.groupby(["homeTeamId", "awayTeamId"])["excitement"]
        thresh = grp.transform(lambda x: x.mean() + x.std())
        out["rivalry_game"] = (out["excitement"] > thresh).astype(int)
        out["big_game"] = out["excitement"] > out["excitement"].quantile(0.75)

    return out


def add_enhanced_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.sort_values(["startDate", "homeTeamId"]).copy()
    for team_type in ["home", "away"]:
        points_col = f"{team_type}Points"
        team_id_col = f"{team_type}TeamId"
        opp = "away" if team_type == "home" else "home"

        if not {points_col, team_id_col, f"{opp}Points"}.issubset(out.columns):
            continue

        out[points_col] = pd.to_numeric(out[points_col], errors="coerce")
        out[f"{opp}Points"] = pd.to_numeric(out[f"{opp}Points"], errors="coerce")

        # Short/long-term scoring
        for window in [3, 6]:
            out[f"{team_type}_points_last{window}"] = out.groupby([team_id_col, "season"])[points_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        out[f"{team_type}_points_momentum"] = (
            out.get(f"{team_type}_points_last3", 0) - out.get(f"{team_type}_points_last6", 0)
        )

        out[f"{team_type}_won"] = (out[points_col] > out[f"{opp}Points"]).astype(int)
        out[f"{team_type}_point_diff"] = out[points_col] - out[f"{opp}Points"]

        # scoring efficiency (vs season mean)
        season_mean = out.groupby([team_id_col, "season"])[points_col].transform("mean")
        out[f"{team_type}_scoring_efficiency"] = out[points_col] / season_mean.replace(0, np.nan)

        out[f"{team_type}_in_form"] = (out[f"{team_type}_points_momentum"] > 0).astype(int)

    return out


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering without undefined dependencies.
    (Removed add_sos_features / add_interaction_features which were undefined.)
    """
    enhanced = handle_missing_data(df)
    enhanced = add_team_performance_context(enhanced)
    enhanced = add_enhanced_timing_features(enhanced)
    enhanced = add_enhanced_momentum_features(enhanced)

    # Season period bucket by week if available
    if "week" in enhanced.columns and len(enhanced) > 0:
        try:
            enhanced["season_period"] = pd.qcut(enhanced["week"], q=3, labels=["Early", "Mid", "Late"])
        except Exception:
            enhanced["season_period"] = "Mid"

    # Expected scoring
    if "overUnder" in enhanced.columns:
        enhanced["expected_scoring"] = pd.to_numeric(enhanced["overUnder"], errors="coerce")
        enhanced["expected_scoring"] = enhanced["expected_scoring"].fillna(enhanced["expected_scoring"].mean())

        if "fpiOffensiveEfficiency" in enhanced.columns:
            # Use home team indicator if available; otherwise neutral factor
            home_indicator = np.where(enhanced.get("homeTeam").notna(), 1.1, 1.0) if "homeTeam" in enhanced.columns else 1.0
            enhanced["scoring_efficiency_interaction"] = (
                enhanced["expected_scoring"] * pd.to_numeric(enhanced["fpiOffensiveEfficiency"], errors="coerce").fillna(0)
                * home_indicator
            )

    return enhanced


# =========================
# Prediction core
# =========================
def calculate_dynamic_weights(
    home_team: str,
    away_team: str,
    home_recent: pd.DataFrame,
    away_recent: pd.DataFrame,
    df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate dynamic weights based on team characteristics and data quality.
    """
    weights = {
        "base_strength": 0.4,
        "recent_form": 0.15,
        "h2h": 0.1,
        "conference": 0.25,
        "efficiency": 0.30,
        "rankings": 0.25,
    }

    # Conference strength / diff
    home_conf = home_recent.get("home_conference").iloc[0] if "home_conference" in home_recent.columns and not home_recent.empty else "Unknown"
    away_conf = away_recent.get("away_conference").iloc[0] if "away_conference" in away_recent.columns and not away_recent.empty else "Unknown"

    conf_games = df[
        (df["home_conference"].isin([home_conf, away_conf])) | (df["away_conference"].isin([home_conf, away_conf]))
    ] if {"home_conference", "away_conference"}.issubset(df.columns) else pd.DataFrame()

    if len(conf_games) > 0 and {"homePoints", "awayPoints"}.issubset(conf_games.columns):
        # Careful with precedence: use parentheses around each side
        home_conf_wins = conf_games[
            ((conf_games["home_conference"] == home_conf) & (conf_games["homePoints"] > conf_games["awayPoints"])) |
            ((conf_games["away_conference"] == home_conf) & (conf_games["awayPoints"] > conf_games["homePoints"]))
        ].shape[0] / conf_games.shape[0]

        conf_diff = abs(home_conf_wins - 0.5)
        weights["conference"] *= (1 + conf_diff)
        weights["h2h"] *= (1 - conf_diff)

    # Rankings diff (if present)
    if {"fpi", "spOverall"}.issubset(df.columns):
        home_rank = pd.to_numeric(home_recent.get("fpi", pd.Series(dtype=float)), errors="coerce").mean()
        home_rank += pd.to_numeric(home_recent.get("spOverall", pd.Series(dtype=float)), errors="coerce").mean()
        away_rank = pd.to_numeric(away_recent.get("fpi", pd.Series(dtype=float)), errors="coerce").mean()
        away_rank += pd.to_numeric(away_recent.get("spOverall", pd.Series(dtype=float)), errors="coerce").mean()

        # scale by global max to stay stable
        global_max = pd.concat(
            [
                pd.to_numeric(df.get("fpi", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(df.get("spOverall", pd.Series(dtype=float)), errors="coerce"),
            ],
            axis=0,
        ).max()
        if pd.isna(global_max) or global_max == 0:
            global_max = 1.0
        rank_diff = abs(home_rank - away_rank) / max(global_max, 1)
        weights["rankings"] *= (1 + rank_diff)
        weights["recent_form"] *= (1 - rank_diff * 0.5)

    # Data quality (how much recent data we have)
    home_data_quality = min(len(home_recent) / 12.0, 1.0)
    away_data_quality = min(len(away_recent) / 12.0, 1.0)
    dq = (home_data_quality + away_data_quality) / 2.0
    weights["base_strength"] *= dq
    weights["recent_form"] *= dq

    # Normalize to sum 1
    total = sum(weights.values())
    if total <= 0:
        # fallback
        weights = {k: 1.0 / len(weights) for k in weights}
    else:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def predict_game_score(
    home_team: str, away_team: str, df: pd.DataFrame, season: int | None = None
) -> Tuple[float | None, float | None, Dict[str, Any]]:
    """
    Predict game score using dynamically calculated weights.
    Returns (home_score, away_score, details).
    """
    if df is None or df.empty:
        return None, None, {}

    # Narrow to the most recent rows for each side (based on team name columns you use on the site)
    home_recent = df[df.get("homeTeam") == home_team].sort_values("startDate", ascending=False)
    away_recent = df[df.get("awayTeam") == away_team].sort_values("startDate", ascending=False)
    if len(home_recent) == 0 or len(away_recent) == 0:
        return None, None, {}

    def smean(x, default=0.0) -> float:
        val = pd.to_numeric(x, errors="coerce").mean()
        return float(val) if pd.notna(val) else float(default)

    def sval(v, default=0.0) -> float:
        try:
            v = float(v)
            return v if pd.notna(v) else float(default)
        except Exception:
            return float(default)

    # Weights
    WEIGHTS = calculate_dynamic_weights(home_team, away_team, home_recent, away_recent, df)

    # Base metrics
    home_avg = smean(df[df.get("homeTeam") == home_team]["homePoints"])
    away_avg = smean(df[df.get("awayTeam") == away_team]["awayPoints"])
    home_advantage = sval(smean(df["homePoints"]) - smean(df["awayPoints"]), 0.0)

    # Strength vs league means
    home_strength = sval(smean(home_recent["homePoints"]) - smean(df["homePoints"]), 0.0)
    away_strength = sval(smean(away_recent["awayPoints"]) - smean(df["awayPoints"]), 0.0)

    # Recent form (weighted last 5)
    weights = [0.35, 0.25, 0.20, 0.15, 0.05]
    h_pts = pd.to_numeric(home_recent["homePoints"], errors="coerce").dropna().head(5).tolist()
    a_pts = pd.to_numeric(away_recent["awayPoints"], errors="coerce").dropna().head(5).tolist()
    home_form = sval(np.average(h_pts[: len(weights)], weights=weights[: len(h_pts)]) - home_avg, 0.0) if h_pts else 0.0
    away_form = sval(np.average(a_pts[: len(weights)], weights=weights[: len(a_pts)]) - away_avg, 0.0) if a_pts else 0.0

    # H2H
    h2h = df[
        ((df.get("homeTeam") == home_team) & (df.get("awayTeam") == away_team))
        | ((df.get("homeTeam") == away_team) & (df.get("awayTeam") == home_team))
    ].sort_values("startDate", ascending=False)

    h2h_factor = 0.0
    if len(h2h) >= 3 and {"homePoints", "awayPoints"}.issubset(h2h.columns):
        h2h_weights = [0.5, 0.3, 0.2]
        h2h_games = h2h.head(3)[["homeTeam", "homePoints", "awayPoints"]]
        deltas = (h2h_games["homePoints"] - h2h_games["awayPoints"]).tolist()
        w = h2h_weights[: len(deltas)]
        h2h_factor = sval(np.average(deltas, weights=w), 0.0)
        # If last meeting had teams flipped, invert sign
        if not h2h_games.empty and h2h_games.iloc[0]["homeTeam"] == away_team:
            h2h_factor *= -1

    # Conference stats
    conf_stats = {
        "home": {"points": 0.0, "offense": 0.0, "defense": 0.0},
        "away": {"points": 0.0, "offense": 0.0, "defense": 0.0},
    }
    epa_available = {"home_epa", "away_epa", "home_epaAllowed", "away_epaAllowed"}.issubset(df.columns)

    if {"home_conference", "away_conference"}.issubset(df.columns):
        home_conf = home_recent.get("home_conference").iloc[0] if "home_conference" in home_recent.columns and not home_recent.empty else "Unknown"
        away_conf = away_recent.get("away_conference").iloc[0] if "away_conference" in away_recent.columns and not away_recent.empty else "Unknown"

        home_conf_data = df[df["home_conference"] == home_conf]
        away_conf_data = df[df["away_conference"] == away_conf]

        conf_stats["home"]["points"] = smean(home_conf_data["homePoints"])
        conf_stats["away"]["points"] = smean(away_conf_data["awayPoints"])
        if epa_available:
            conf_stats["home"]["offense"] = smean(home_conf_data["home_epa"])
            conf_stats["home"]["defense"] = smean(home_conf_data["home_epaAllowed"])
            conf_stats["away"]["offense"] = smean(away_conf_data["away_epa"])
            conf_stats["away"]["defense"] = smean(away_conf_data["away_epaAllowed"])

    conf_diff = sval(
        (conf_stats["home"]["points"] - conf_stats["away"]["points"]) * WEIGHTS["conference"]
        + (conf_stats["home"]["offense"] - conf_stats["away"]["offense"]) * (WEIGHTS["conference"] * 0.5)
        - (conf_stats["home"]["defense"] - conf_stats["away"]["defense"]) * (WEIGHTS["conference"] * 0.5),
        0.0,
    )

    # Efficiency (EPA) diff
    efficiency_diff = 0.0
    if epa_available:
        home_off = smean(home_recent["home_epa"])
        home_def = smean(home_recent["home_epaAllowed"])
        away_off = smean(away_recent["away_epa"])
        away_def = smean(away_recent["away_epaAllowed"])
        efficiency_diff = sval((home_off - away_def) * WEIGHTS["efficiency"] - (away_off - home_def) * WEIGHTS["efficiency"])

    # Rankings
    rankings_impact = 0.0
    if {"fpi", "spOverall"}.issubset(df.columns):
        home_rank = smean(home_recent[["fpi", "spOverall"]].mean(axis=1))
        away_rank = smean(away_recent[["fpi", "spOverall"]].mean(axis=1))
        rankings_impact = sval((home_rank - away_rank) * WEIGHTS["rankings"])

    # Weather (very light touch; if missing, neutral)
    weather_impact = 0.0
    if {"temperature", "windSpeed"}.issubset(df.columns) and len(df) > 0:
        temp = sval(df["temperature"].iloc[0], 70)
        wind = sval(df["windSpeed"].iloc[0], 0)
        if temp < 40:
            weather_impact -= 1
        elif temp > 85:
            weather_impact -= 0.5
        if wind > 15:
            weather_impact -= wind * 0.1

    # Season progress
    season_factor = 0.0
    if season is not None and "week" in df.columns and "season" in df.columns:
        current_week = sval(df[df["season"] == season]["week"].max(), 7)
        season_progress = current_week / 14.0  # heuristic
        season_factor = sval(season_progress * 2.0)

    # Rest differential
    rest_advantage = 0.0
    if "rest_days" in df.columns:
        h_rest = sval(home_recent.get("rest_days").iloc[0] if not home_recent.empty else 0.0)
        a_rest = 0.0  # rest_days is already (home - away)
        rest_advantage = sval(h_rest - a_rest, 0.0) * 0.5
    elif {"home_rest_days", "away_rest_days"}.issubset(df.columns):
        h_rest = sval(home_recent.get("home_rest_days").iloc[0] if "home_rest_days" in home_recent.columns and not home_recent.empty else 7.0)
        a_rest = sval(away_recent.get("away_rest_days").iloc[0] if "away_rest_days" in away_recent.columns and not away_recent.empty else 7.0)
        rest_advantage = sval(h_rest - a_rest, 0.0) * 0.5

    # Final scores
    home_score = sval(
        home_avg
        + home_strength * WEIGHTS["base_strength"]
        + home_form * WEIGHTS["recent_form"]
        + h2h_factor * WEIGHTS["h2h"]
        + home_advantage
        + conf_diff
        + efficiency_diff
        + rankings_impact
        + weather_impact
        + season_factor
        + rest_advantage
    )
    away_score = sval(
        away_avg
        + away_strength * WEIGHTS["base_strength"]
        + away_form * WEIGHTS["recent_form"]
        - h2h_factor * WEIGHTS["h2h"]
        - conf_diff
        - efficiency_diff
        - rankings_impact
        + weather_impact
        + season_factor
        - rest_advantage
    )

    details = {
        "base_scores": {"home": sval(home_avg), "away": sval(away_avg)},
        "strength": {"home": sval(home_strength), "away": sval(away_strength)},
        "form": {"home": sval(home_form), "away": sval(away_form)},
        "h2h_factor": sval(h2h_factor),
        "conf_stats": conf_stats,
        "efficiency": {"diff": sval(efficiency_diff)},
        "rankings": {"impact": sval(rankings_impact)},
        "weather": {"impact": sval(weather_impact)},
        "season": {"factor": sval(season_factor)},
        "rest": {"advantage": sval(rest_advantage)},
        "weights": WEIGHTS,
    }
    return round(home_score, 1), round(away_score, 1), details


def predict_score(home_team: str, away_team: str, df: pd.DataFrame, season: int = 2024) -> Dict[str, Any]:
    """
    UI-friendly wrapper that returns a dict your site can render.
    """
    home_score, away_score, details = predict_game_score(home_team, away_team, df, season)
    if home_score is None or away_score is None:
        return {
            "scores": None,
            "prediction": None,
            "factors": None,
            "weights": None,
            "error": "Insufficient data to make a prediction for the specified teams.",
        }

    raw_spread = home_score - away_score
    if raw_spread > 0:
        favorite, underdog, spread = home_team, away_team, -raw_spread
    else:
        favorite, underdog, spread = away_team, home_team, raw_spread

    return {
        "scores": {
            "home": {
                "team": home_team,
                "score": float(home_score),
                "stats": {
                    "avg_score": float(details["base_scores"]["home"]),
                    "strength": float(details["strength"]["home"]),
                    "form": float(details["form"]["home"]),
                    "conference_avg": float(details["conf_stats"]["home"]["points"]),
                },
            },
            "away": {
                "team": away_team,
                "score": float(away_score),
                "stats": {
                    "avg_score": float(details["base_scores"]["away"]),
                    "strength": float(details["strength"]["away"]),
                    "form": float(details["form"]["away"]),
                    "conference_avg": float(details["conf_stats"]["away"]["points"]),
                },
            },
        },
        "prediction": {
            "favorite": favorite,
            "underdog": underdog,
            "spread": round(float(spread), 1),
            "total": round(float(home_score + away_score), 1),
        },
        "factors": {
            "rankings": round(float(details["rankings"]["impact"]), 1),
            "efficiency": round(float(details["efficiency"]["diff"]), 1),
            "h2h": round(float(details["h2h_factor"]), 1),
            "weather": round(float(details["weather"]["impact"]), 1),
            "rest": round(float(details["rest"]["advantage"]), 1),
        },
        "weights": {k: round(v * 100.0, 1) for k, v in details["weights"].items()},
    }


# =========================
# Model IO (for your saved training artifacts)
# =========================
def initialize_model() -> bool:
    """
    Load saved models/encoders/features and the processed games CSV.
    Your web app can import predictor.py and call predict_score with `games_df`.
    """
    global games_df, model_info, home_encoders
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, "saved")
        data_path = os.path.join(os.path.dirname(current_dir), "data")

        model_info = {
            "models": {
                "homePoints": {
                    "model": joblib.load(os.path.join(save_path, "home_points_model.joblib")),
                    "metrics": joblib.load(os.path.join(save_path, "home_points_metrics.joblib")),
                },
                "awayPoints": {
                    "model": joblib.load(os.path.join(save_path, "away_points_model.joblib")),
                    "metrics": joblib.load(os.path.join(save_path, "away_points_metrics.joblib")),
                },
            },
            "imputer": joblib.load(os.path.join(save_path, "imputer.joblib")),
            "encoders": joblib.load(os.path.join(save_path, "encoders.joblib")),
            "features": joblib.load(os.path.join(save_path, "features.joblib")),
        }

        games_df = pd.read_csv(os.path.join(data_path, "processed_games.csv"))
        if "startDate" in games_df.columns:
            games_df["startDate"] = pd.to_datetime(games_df["startDate"], errors="coerce")

        # keep a lightly cleaned copy for inference
        games_df = handle_missing_data(games_df)

        home_encoders = model_info["encoders"]
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Ensure the training script has created the model files in ./saved and data/processed_games.csv exists.")
        return False


def save_models(
    results: Dict[str, Any],
    best_home_model_name: str,
    best_away_model_name: str,
    home_encoders_in: Dict[str, Any],
    X_home_train: pd.DataFrame,
    games_df_in: pd.DataFrame,
) -> bool:
    """
    Persist trained models, metrics, preprocessors, selected features, and processed dataset.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, "saved")
        data_path = os.path.join(os.path.dirname(current_dir), "data")
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        home_results = results["home"][best_home_model_name]
        away_results = results["away"][best_away_model_name]

        joblib.dump(home_results["model"], os.path.join(save_path, "home_points_model.joblib"))
        joblib.dump(home_results["test_metrics"], os.path.join(save_path, "home_points_metrics.joblib"))

        joblib.dump(away_results["model"], os.path.join(save_path, "away_points_model.joblib"))
        joblib.dump(away_results["test_metrics"], os.path.join(save_path, "away_points_metrics.joblib"))

        joblib.dump(home_results["imputer"], os.path.join(save_path, "imputer.joblib"))
        joblib.dump(home_encoders_in, os.path.join(save_path, "encoders.joblib"))
        joblib.dump(list(X_home_train.columns), os.path.join(save_path, "features.joblib"))

        games_df_in.to_csv(os.path.join(data_path, "processed_games.csv"), index=False)
        print("All models and data saved successfully!")
        return True
    except Exception as e:
        print(f"Fatal error during save process: {e}")
        return False


# =========================
# Module init on import
# =========================
if initialize_model():
    print("Model initialized successfully")
else:
    # You can choose to not hard-fail on import for web apps:
    # raise RuntimeError("Failed to initialize model")
    print("Warning: model failed to initialize on import. You can still load it later by calling initialize_model().")


__all__ = [
    "games_df",
    "model_info",
    "predict_game_score",
    "predict_score",
    "add_enhanced_features",
    "add_team_performance_context",
    "add_enhanced_timing_features",
    "add_enhanced_momentum_features",
    "handle_missing_data",
    "save_models",
    "initialize_model",
]
