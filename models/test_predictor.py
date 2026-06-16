"""
Smoke + guardrail tests for the CFB predictor.

Run directly (``python models/test_predictor.py``) or via pytest.
Covers: model loads, prediction contract, sane ranges, sign consistency, and a
leakage guard that fails if a post-game column sneaks into the feature list.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predictor  # noqa: E402

# Substrings that indicate a feature is derived from the game's own outcome.
LEAKY_TOKENS = (
    "points", "won", "win_streak", "loss", "streak", "momentum", "end_elo",
    "endelo", "postgame", "season_avg", "historical", "scoring_efficiency",
    "point_diff", "excitement", "in_form", "win_quality", "rolling",
)


def test_no_leaky_features():
    for f in predictor.FEATURES:
        low = f.lower()
        assert not any(tok in low for tok in LEAKY_TOKENS), \
            f"Potentially leaky feature in FEATURES: {f}"


def test_model_loads():
    assert predictor.get_model_bundle() is not None, \
        "Model bundle missing — run `python models/train.py` first."


def _sample_teams():
    df = predictor.get_games_df()
    teams = df["homeTeam"].dropna().unique().tolist()
    assert len(teams) >= 2
    return teams[0], teams[1]


def test_prediction_contract_and_ranges():
    home, away = _sample_teams()
    r = predictor.predict_score(home, away)

    assert r.get("error") is None, r.get("error")
    for key in ("scores", "prediction", "factors", "weights"):
        assert key in r, f"missing key: {key}"

    hs = r["scores"]["home"]["score"]
    as_ = r["scores"]["away"]["score"]
    assert 0 <= hs <= 100 and 0 <= as_ <= 100, f"scores out of range: {hs}, {as_}"

    pred = r["prediction"]
    assert pred["favorite"] in (home, away)
    assert pred["total"] > 0
    # Favorite must be the higher-scoring side.
    if hs >= as_:
        assert pred["favorite"] == home
    else:
        assert pred["favorite"] == away


def test_unknown_team_is_handled():
    r = predictor.predict_score("__nonexistent_home__", "__nonexistent_away__")
    assert r.get("error") is None
    assert r["low_confidence"] is True


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
