"""
Downloads and preprocesses international football results from
https://github.com/martj42/international_results

Key design choices:
- Time decay half-life = 180 days (focus on recent form, not 2014 results)
- WC 2026 actual results carry a 3x weight multiplier (strongest available signal)
- Matches with combined weight < 0.02 are dropped (noise from old friendlies)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
DATA_DIR        = Path(__file__).parent.parent / "data" / "raw"
WC2026_PATH     = Path(__file__).parent.parent / "data" / "wc2026_results.csv"

# Translates names the user might write in wc2026_results.csv → exact martj42 name.
# Verified against martj42 results.csv (June 2026 snapshot).
# Only entries that actually differ from the martj42 stored name are needed;
# tolerance aliases (e.g. "Curacao" without ç) are included for usability.
_WC_TO_MARTJ42: dict[str, str] = {
    # True renames
    "Czechia":          "Czech Republic",
    "Türkiye":          "Turkey",
    "Turkiye":          "Turkey",          # accent-free alias
    "Côte d'Ivoire":    "Ivory Coast",
    # Tolerance aliases — user may omit special characters
    "Curacao":          "Curaçao",         # without ç → with ç (martj42 uses ç)
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",  # hyphenated alias
    # Bosnia and Herzegovina: martj42 stores exactly "Bosnia and Herzegovina", no rename needed
    # Curaçao: martj42 stores exactly "Curaçao", no rename needed
}

# WC 2026 multiplier: each confirmed WC result is worth ~3 regular WC matches
# from last year, anchoring model parameters to actual tournament performance.
WC2026_WEIGHT_MULTIPLIER = 3.0

# Importance weights by tournament keyword (checked via 'in tournament.lower()')
_TOURNAMENT_WEIGHTS: list[tuple[str, float]] = [
    ("fifa world cup qualification",         0.60),
    ("fifa world cup",                       1.00),
    ("uefa euro qualification",              0.55),
    ("uefa euro",                            0.85),
    ("copa america",                         0.85),
    ("african cup of nations qualification", 0.50),
    ("african cup of nations",               0.80),
    ("afc asian cup qualification",          0.50),
    ("afc asian cup",                        0.80),
    ("concacaf gold cup qualification",      0.45),
    ("concacaf gold cup",                    0.75),
    ("concacaf nations league",              0.70),
    ("uefa nations league",                  0.70),
    ("copa centroamericana",                 0.65),
    ("conmebol",                             0.70),
    ("friendly",                             0.30),
]


def _tournament_weight(tournament: str) -> float:
    t = tournament.lower()
    for keyword, w in _TOURNAMENT_WEIGHTS:
        if keyword in t:
            return w
    if "qualification" in t or "qualifying" in t:
        return 0.55
    if "friendly" in t:
        return 0.30
    return 0.50


def download(force: bool = False) -> pd.DataFrame:
    """Download (or load cached) results.csv from martj42/international_results."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "results.csv"

    if not cache.exists() or force:
        print(f"Downloading {RESULTS_URL} ...")
        r = requests.get(RESULTS_URL, timeout=120)
        r.raise_for_status()
        cache.write_bytes(r.content)
        print(f"  Saved -> {cache}")
    else:
        print(f"  Using cached data: {cache}")

    df = pd.read_csv(cache, parse_dates=["date"])
    return df


def _martj42_name(name: str) -> str:
    """Translate WC display names to martj42 dataset names."""
    return _WC_TO_MARTJ42.get(name, name)


def fill_wc2026_scores(
    df: pd.DataFrame,
    path: Path = WC2026_PATH,
) -> pd.DataFrame:
    """
    Fill in confirmed WC 2026 scores that aren't yet in the martj42 dataset.

    Normalises team names from the tracker CSV to martj42 conventions
    (e.g. "Czechia" → "Czech Republic") before matching, preventing duplicate
    rows when the display name differs from the dataset name.
    """
    if not path.exists():
        return df

    updates = pd.read_csv(path, parse_dates=["date"])
    # Strip accidental whitespace in score columns (e.g. " 4" → "4")
    for col in ("home_score", "away_score"):
        if updates[col].dtype == object:
            updates[col] = updates[col].astype(str).str.strip()
    updates["home_score"] = pd.to_numeric(updates["home_score"], errors="coerce")
    updates["away_score"] = pd.to_numeric(updates["away_score"], errors="coerce")

    if updates.empty:
        return df

    df = df.copy()
    added = 0
    updated = 0

    for _, row in updates.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        # Translate display name → martj42 name before matching
        home = _martj42_name(str(row["home_team"]))
        away = _martj42_name(str(row["away_team"]))

        mask = (
            (df["date"] == row["date"])
            & (df["home_team"] == home)
            & (df["away_team"] == away)
        )
        if mask.any():
            df.loc[mask, "home_score"] = row["home_score"]
            df.loc[mask, "away_score"] = row["away_score"]
            updated += 1
        else:
            new = {
                "date":       row["date"],
                "home_team":  home,
                "away_team":  away,
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "tournament": "FIFA World Cup",
                "city":       "",
                "country":    "",
                "neutral":    True,
            }
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            added += 1

    if updated or added:
        print(f"  WC 2026 results: {updated} scores filled, {added} new rows added")

    return df


def preprocess(
    df: pd.DataFrame,
    min_year: int = 2014,
    reference_date: str = "2026-06-12",
    half_life_days: float = 180,
    min_weight: float = 0.02,
) -> pd.DataFrame:
    """
    Filter, weight, and clean the raw results DataFrame.

    Weight = tournament_importance * time_decay
    WC 2026 actual results receive an additional WC2026_WEIGHT_MULTIPLIER boost.

    half_life_days=180 focuses model on last ~6 months of form rather than
    a broad 2-year window, which better captures current team strength.
    """
    ref = pd.Timestamp(reference_date)

    # Drop future fixtures (NaN scores) and pre-min_year data
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df[df["date"].dt.year >= min_year]

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Tournament importance
    df["t_weight"] = df["tournament"].apply(_tournament_weight)

    # Time decay: exp(-ln2/half_life * days_ago)
    days_ago = (ref - df["date"]).dt.days.clip(lower=0)
    decay_rate = np.log(2) / half_life_days
    df["time_weight"] = np.exp(-decay_rate * days_ago)

    # Base combined weight
    df["weight"] = df["t_weight"] * df["time_weight"]

    # Boost for confirmed WC 2026 results (played on/after June 11)
    wc2026_mask = (
        df["tournament"].str.contains("FIFA World Cup", na=False)
        & (df["date"] >= pd.Timestamp("2026-06-11"))
    )
    df.loc[wc2026_mask, "weight"] *= WC2026_WEIGHT_MULTIPLIER

    # Drop low-weight matches (old friendlies, irrelevant noise)
    df = df[df["weight"] >= min_weight]

    # Normalise neutral column to bool
    df["neutral"] = df["neutral"].map(
        {True: True, False: False, "TRUE": True, "FALSE": False,
         "True": True, "False": False, 1: True, 0: False}
    ).fillna(False)

    return df.reset_index(drop=True)
