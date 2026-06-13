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

# Map martj42 names -> WC display names (inverse of to_model_name)
_MODEL_TO_DISPLAY: dict[str, str] = {
    "Czech Republic":    "Czechia",
    "Turkey":            "Turkiye",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Curacao":           "Curacao",
    "Ivory Coast":       "Ivory Coast",
}


def _display(name: str) -> str:
    return _MODEL_TO_DISPLAY.get(name, name)


def load_actual_results(path: Path = WC2026_PATH) -> pd.DataFrame:
    """
    Merge confirmed WC 2026 results from the tracker file with matches
    already in the martj42 dataset (June 11 onward, non-NaN scores).
    """
    from .data_loader import DATA_DIR

    raw = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    wc = raw[
        raw["tournament"].str.contains("FIFA World Cup", na=False)
        & (raw["date"] >= pd.Timestamp("2026-06-11"))
    ].dropna(subset=["home_score", "away_score"]).copy()

    # Add confirmed results from tracker that may not be in martj42 yet
    if path.exists():
        extra = pd.read_csv(path, parse_dates=["date"])
        extra = extra.dropna(subset=["home_score", "away_score"])
        if not extra.empty:
            extra["tournament"] = "FIFA World Cup"
            extra["neutral"]    = True
            wc = pd.concat([wc, extra], ignore_index=True)
            # Deduplicate: keep tracker version (more up-to-date) on conflict
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


def analyse(model: DixonColesModel) -> pd.DataFrame:
    """
    Return a DataFrame comparing xG predictions vs actual goals for every
    completed WC 2026 group-stage match.
    """
    actuals = load_actual_results()
    if actuals.empty:
        print("  No completed WC 2026 matches found.")
        return pd.DataFrame()

    rows = []
    for _, match in actuals.iterrows():
        ht = match["home_team"]
        at = match["away_team"]
        lam, mu = model.predict_xg(ht, at, neutral=True)

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
