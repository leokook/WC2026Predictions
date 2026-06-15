"""
World Football Elo ratings computed directly from the martj42 dataset.

K-factor varies by tournament importance so that WC matches carry more
weight than friendlies — same philosophy as our Dixon-Coles weighting.
No external API needed; results cached to data/cache/elo_ratings.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_PATH = Path(__file__).parent.parent / "data" / "cache" / "elo_ratings.csv"
INITIAL_ELO = 1500.0

# K-factor by tournament keyword (checked via `in tournament.lower()`)
_K_FACTORS: list[tuple[str, float]] = [
    ("fifa world cup",          60),
    ("copa america",            50),
    ("uefa euro",               50),
    ("african cup",             45),
    ("afc asian cup",           45),
    ("gold cup",                40),
    ("nations league",          40),
    ("concacaf nations",        40),
    ("qualification",           30),
    ("qualifying",              30),
    ("friendly",                10),
]
_DEFAULT_K = 25.0


def _k_factor(tournament: str) -> float:
    t = tournament.lower()
    for keyword, k in _K_FACTORS:
        if keyword in t:
            return k
    return _DEFAULT_K


def compute_elo(
    df: pd.DataFrame,
    cutoff_date: str = "2026-06-10",
    force: bool = False,
) -> dict[str, float]:
    """
    Compute Elo ratings from full martj42 history up to cutoff_date.

    Uses the full dataset (from 1872) because early history still helps
    establish stable ratings for older nations; K decays naturally with
    the sparse data of those eras.

    Result is cached; set force=True to re-compute.
    """
    if CACHE_PATH.exists() and not force:
        cached = pd.read_csv(CACHE_PATH)
        return dict(zip(cached["team"], cached["elo"]))

    cutoff = pd.Timestamp(cutoff_date)
    played = df[df["date"] <= cutoff].dropna(
        subset=["home_score", "away_score"]
    ).sort_values("date")

    elos: dict[str, float] = {}

    for _, row in played.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        elo_h = elos.get(h, INITIAL_ELO)
        elo_a = elos.get(a, INITIAL_ELO)

        # Home-field adjustment: +100 Elo points if not neutral
        neutral = bool(row.get("neutral", False))
        adj_h = elo_h + (0 if neutral else 100)

        exp_h = 1.0 / (1.0 + 10 ** ((elo_a - adj_h) / 400))
        exp_a = 1.0 - exp_h

        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        if hs > as_:
            act_h, act_a = 1.0, 0.0
        elif hs < as_:
            act_h, act_a = 0.0, 1.0
        else:
            act_h = act_a = 0.5

        # Goal-difference margin multiplier (capped at 3): winning by more = bigger update
        gd = abs(hs - as_)
        margin = 1.0 + min(gd - 1, 2) * 0.25 if gd > 1 else 1.0

        k = _k_factor(str(row.get("tournament", "")))
        elos[h] = elo_h + k * margin * (act_h - exp_h)
        elos[a] = elo_a + k * margin * (act_a - exp_a)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        sorted(elos.items(), key=lambda x: -x[1]), columns=["team", "elo"]
    ).to_csv(CACHE_PATH, index=False)

    return elos


def elo_win_prob(elo_a: float, elo_b: float) -> float:
    """P(team_a wins) using standard Elo formula (neutral venue)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def elo_xg_adjustment(
    lam: float,
    mu: float,
    elo_a: float,
    elo_b: float,
    blend_weight: float = 0.20,
) -> tuple[float, float]:
    """
    Blend Dixon-Coles xG with an Elo-implied strength ratio.

    Preserves the total expected goals (λ + μ) but shifts the balance
    between the two teams according to their Elo difference.
    blend_weight=0.20 means 20% Elo, 80% Dixon-Coles.
    """
    total = lam + mu
    if total < 1e-6 or mu < 1e-6:
        return lam, mu

    dc_ratio = lam / mu                                     # DC implied odds
    p_a = elo_win_prob(elo_a, elo_b)
    elo_ratio = p_a / max(1 - p_a, 1e-6)                   # Elo implied odds

    blended_ratio = (1 - blend_weight) * dc_ratio + blend_weight * elo_ratio
    lam_new = total * blended_ratio / (1 + blended_ratio)
    mu_new  = total / (1 + blended_ratio)
    return float(lam_new), float(mu_new)
