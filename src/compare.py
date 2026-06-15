"""
Compare model predictions (xG) against actual WC 2026 goals.

Metrics:
  - MAE on goals per match-team (e.g. predicted 1.6 vs actual 2)
  - Direction accuracy (predicted winner = actual winner)
  - Brier score on match outcomes (home win / draw / away win)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .model import DixonColesModel
from .wc2026_teams import GROUPS, to_model_name

WC2026_PATH = Path(__file__).parent.parent / "data" / "wc2026_results.csv"

# martj42 display names (used in team_a/team_b columns of results)
# martj42 name → WC display name (for human-readable output only).
# Only non-identity entries are needed.
_MODEL_TO_DISPLAY: dict[str, str] = {
    "Czech Republic": "Czechia",
    "Turkey":         "Türkiye",
}


def _display(name: str) -> str:
    return _MODEL_TO_DISPLAY.get(name, name)


def load_actual_results() -> pd.DataFrame:
    """
    Return confirmed WC 2026 results using normalised martj42 team names.

    Loads June-11+ scores from the martj42 cached file, then fills/appends
    any confirmed results from the tracker (wc2026_results.csv).  The tracker
    names are translated to martj42 conventions before deduplication so that
    e.g. "Czechia" and "Czech Republic" don't produce duplicate rows.
    """
    from .data_loader import DATA_DIR, _martj42_name

    raw = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    wc = raw[
        raw["tournament"].str.contains("FIFA World Cup", na=False)
        & (raw["date"] >= pd.Timestamp("2026-06-11"))
    ].dropna(subset=["home_score", "away_score"]).copy()

    if WC2026_PATH.exists():
        extra = pd.read_csv(WC2026_PATH, parse_dates=["date"])
        for col in ("home_score", "away_score"):
            if extra[col].dtype == object:
                extra[col] = extra[col].astype(str).str.strip()
        extra["home_score"] = pd.to_numeric(extra["home_score"], errors="coerce")
        extra["away_score"] = pd.to_numeric(extra["away_score"], errors="coerce")
        extra = extra.dropna(subset=["home_score", "away_score"])

        if not extra.empty:
            # Normalise tracker names → martj42 names before merging
            extra["home_team"] = extra["home_team"].apply(_martj42_name)
            extra["away_team"] = extra["away_team"].apply(_martj42_name)
            extra["tournament"] = "FIFA World Cup"
            extra["neutral"]    = True
            wc = pd.concat([wc, extra], ignore_index=True)
            wc = wc.drop_duplicates(
                subset=["date", "home_team", "away_team"], keep="last"
            )

    wc["home_score"] = wc["home_score"].astype(int)
    wc["away_score"] = wc["away_score"].astype(int)
    return wc.sort_values("date").reset_index(drop=True)


def _outcome(home: int, away: int) -> str:
    if home > away:  return "H"
    if home < away:  return "A"
    return "D"


def _outcome_probs(lam: float, mu: float, max_goals: int = 10) -> tuple[float, float, float]:
    """Compute P(home win), P(draw), P(away win) from Poisson parameters."""
    ph = pd.DataFrame(
        [[poisson.pmf(i, lam) * poisson.pmf(j, mu) for j in range(max_goals)]
         for i in range(max_goals)]
    ).values
    p_home = float(np.sum(np.tril(ph, -1)))   # home > away
    p_draw = float(np.trace(ph))
    p_away = float(np.sum(np.triu(ph, 1)))
    return p_home, p_draw, p_away


def _build_pretournament_model(df_train: pd.DataFrame) -> DixonColesModel:
    """
    Fit a model that excludes all WC 2026 match results (June 11 onward).

    This is the only honest baseline for evaluating predictions: we want the
    xG values the model would have produced BEFORE seeing each result, not
    the values it produces after incorporating that result into its parameters.
    Including a match in training and then 'predicting' it inflates accuracy
    (data leakage / look-ahead bias).
    """
    df_pre = df_train[
        ~(
            df_train["tournament"].str.contains("FIFA World Cup", na=False)
            & (df_train["date"] >= pd.Timestamp("2026-06-11"))
        )
    ].copy()
    print(f"  Pre-tournament model: fitting on {len(df_pre):,} matches (WC 2026 excluded)")
    return DixonColesModel().fit(df_pre)


def analyse(model: DixonColesModel, df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Compare PRE-MATCH xG predictions against actual WC 2026 goals.

    Uses a pre-tournament model (trained without any WC 2026 results) so that
    every prediction is genuinely out-of-sample.  The `model` argument (which
    includes WC results) is only used to provide updated xG for upcoming
    matches; the pre-tournament model drives the accuracy metrics.
    """
    actuals = load_actual_results()
    if actuals.empty:
        print("  No completed WC 2026 matches found.")
        return pd.DataFrame()

    pre_model = _build_pretournament_model(df_train)

    rows = []
    for _, match in actuals.iterrows():
        ht = match["home_team"]
        at = match["away_team"]
        try:
            lam, mu = pre_model.predict_xg(ht, at, neutral=True)
        except (KeyError, ValueError):
            print(f"  SKIP: unknown team '{ht}' or '{at}'")
            continue

        actual_h = int(match["home_score"])
        actual_a = int(match["away_score"])
        predicted_outcome = _outcome(round(lam), round(mu))
        actual_outcome    = _outcome(actual_h, actual_a)

        p_h, p_d, p_a = _outcome_probs(lam, mu)
        outcome_probs = {"H": p_h, "D": p_d, "A": p_a}
        brier = sum(
            (1.0 if o == actual_outcome else 0.0 - outcome_probs[o]) ** 2
            for o in ("H", "D", "A")
        )

        rows.append({
            "date":            match["date"].strftime("%b %d"),
            "home":            _display(ht),
            "xg_home":         round(lam, 2),
            "goals_home":      actual_h,
            "xg_away":         round(mu, 2),
            "goals_away":      actual_a,
            "away":            _display(at),
            "pred_outcome":    predicted_outcome,
            "actual_outcome":  actual_outcome,
            "correct":         predicted_outcome == actual_outcome,
            "mae_home":        round(abs(lam - actual_h), 3),
            "mae_away":        round(abs(mu  - actual_a), 3),
            "brier":           round(brier, 4),
        })

    df = pd.DataFrame(rows)
    return df


def summary(df: pd.DataFrame) -> None:
    """Print a human-readable accuracy report."""
    if df.empty:
        return

    print("\n  Match-by-match comparison (xG predicted vs goals actual):")
    print("  " + "-" * 70)
    for _, r in df.iterrows():
        flag = "OK" if r["correct"] else "--"
        print(
            f"  [{flag}] {r['date']}  "
            f"{r['home']:<22} {r['xg_home']:.2f} ({r['goals_home']})  "
            f"vs  "
            f"({r['goals_away']}) {r['xg_away']:.2f}  {r['away']}"
        )

    n = len(df)
    mae = round((df["mae_home"].sum() + df["mae_away"].sum()) / (2 * n), 3)
    acc = round(df["correct"].mean() * 100, 1)
    avg_brier = round(df["brier"].mean(), 4)
    print(f"\n  Matches played: {n}")
    print(f"  Outcome accuracy: {acc}%  ({df['correct'].sum()}/{n} correct)")
    print(f"  Mean Absolute Error (goals): {mae}")
    print(f"  Avg Brier score (lower=better): {avg_brier}")
