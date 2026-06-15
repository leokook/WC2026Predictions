"""
Feature pipeline for WC 2026 match-level xG adjustments.

Orchestrates all five feature sources:
  1. Venue altitude  (venues.py)
  2. Match weather   (weather.py)
  3. Elo ratings     (elo.py)
  4. Shot quality    (shot_quality.py)
  5. Market value    (market_value.py)

Entry points
------------
build_all_features(schedule, raw_df)
    Builds a MatchFeatures object for every group-stage fixture.

adjust_xg(lam, mu, team_a, team_b, match_feat)
    Applies all adjustments to Dixon-Coles xG.  Returns (lam_adj, mu_adj).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .venues import (
    get_venue,
    altitude_multiplier,
    climate_multiplier,
    Venue,
)
from .weather import get_match_weather
from .elo import compute_elo, elo_xg_adjustment
from .shot_quality import load_shot_quality, shot_quality_multiplier
from .market_value import load_market_values, market_value_multiplier


@dataclass
class MatchFeatures:
    """All contextual features for a single WC 2026 fixture."""
    # Venue
    venue: Optional[Venue] = None
    # Weather
    temp_c: float         = 22.0
    humidity_pct: float   = 60.0
    precip_mm: float      = 0.0
    wind_kmh: float       = 15.0
    # Elo
    elo_a: float          = 1500.0
    elo_b: float          = 1500.0
    # Shot quality
    xg_per_shot_a: float  = 0.100
    xg_per_shot_b: float  = 0.100
    # Market value (EUR-M)
    mv_a: float           = 100.0
    mv_b: float           = 100.0


# ---------------------------------------------------------------------------
# Global feature stores (loaded once, reused for all fixtures)
# ---------------------------------------------------------------------------

_elo_ratings:    dict[str, float] = {}
_shot_quality:   dict[str, dict]  = {}
_market_values:  dict[str, float] = {}
_features_ready: bool = False


def load_all_feature_stores(
    raw_df: pd.DataFrame,
    force: bool = False,
) -> None:
    """
    Pre-load Elo, shot quality, and market value stores.
    Call once before build_all_features().
    """
    global _elo_ratings, _shot_quality, _market_values, _features_ready

    print("  Loading Elo ratings ...")
    _elo_ratings = compute_elo(raw_df, force=force)

    print("  Loading shot quality (FBref/fallback) ...")
    _shot_quality = load_shot_quality(force=force)

    print("  Loading market values (Transfermarkt/fallback) ...")
    _market_values = load_market_values(force=force)

    _features_ready = True
    print(f"  Feature stores ready: {len(_elo_ratings)} Elo ratings, "
          f"{len(_shot_quality)} shot-quality entries, "
          f"{len(_market_values)} market values")


def build_match_features(
    team_a_model: str,
    team_b_model: str,
    match_date: str,
    city: str,
    country: str = "",
) -> MatchFeatures:
    """
    Build a MatchFeatures object for one fixture.

    Parameters
    ----------
    team_a_model, team_b_model : martj42 team names
    match_date : "YYYY-MM-DD"
    city, country : as stored in martj42
    """
    feat = MatchFeatures()

    # ── Venue & weather ────────────────────────────────────────────────
    venue = get_venue(city, country)
    feat.venue = venue

    if venue:
        wx = get_match_weather(venue.lat, venue.lon, match_date)
        feat.temp_c       = wx.get("afternoon_temp_c",      22.0)
        feat.humidity_pct = wx.get("afternoon_humidity_pct", 60.0)
        feat.precip_mm    = wx.get("precip_mm",              0.0)
        feat.wind_kmh     = wx.get("wind_max_kmh",           15.0)

    # ── Elo ────────────────────────────────────────────────────────────
    feat.elo_a = _elo_ratings.get(team_a_model, 1500.0)
    feat.elo_b = _elo_ratings.get(team_b_model, 1500.0)

    # ── Shot quality ────────────────────────────────────────────────────
    sq_a = _shot_quality.get(team_a_model, {})
    sq_b = _shot_quality.get(team_b_model, {})
    feat.xg_per_shot_a = sq_a.get("xg_per_shot", 0.100)
    feat.xg_per_shot_b = sq_b.get("xg_per_shot", 0.100)

    # ── Market value ───────────────────────────────────────────────────
    feat.mv_a = _market_values.get(team_a_model, 100.0)
    feat.mv_b = _market_values.get(team_b_model, 100.0)

    return feat


def adjust_xg(
    lam: float,
    mu: float,
    team_a: str,
    team_b: str,
    feat: MatchFeatures,
) -> tuple[float, float]:
    """
    Apply all feature-based multipliers to Dixon-Coles xG.

    Adjustment order (each is multiplicative / independent):
      1. Altitude penalty  — reduces xG for sea-level teams at high venues
      2. Climate penalty   — reduces xG for cold teams in humid heat
      3. Elo blend         — 20% shift toward Elo-implied strength ratio
      4. Shot quality      — scales by team's xG/shot vs tournament average
      5. Market value      — gentle scaling by squad value ratio

    All multipliers are bounded to [0.60, 1.50] to prevent extreme values.
    """
    venue_alt = feat.venue.altitude_m if feat.venue else 0.0

    # 1. Altitude
    alt_a = altitude_multiplier(team_a, venue_alt)
    alt_b = altitude_multiplier(team_b, venue_alt)

    # 2. Climate
    clim_a = climate_multiplier(team_a, feat.temp_c, feat.humidity_pct)
    clim_b = climate_multiplier(team_b, feat.temp_c, feat.humidity_pct)

    # 3. Elo blend (modifies ratio between lam and mu)
    lam_elo, mu_elo = elo_xg_adjustment(
        lam * alt_a * clim_a,
        mu  * alt_b * clim_b,
        feat.elo_a, feat.elo_b,
        blend_weight=0.20,
    )

    # 4. Shot quality
    sq_mult_a = shot_quality_multiplier(team_a, _shot_quality)
    sq_mult_b = shot_quality_multiplier(team_b, _shot_quality)

    # 5. Market value
    mv_mult_a = market_value_multiplier(team_a, team_b, _market_values)
    mv_mult_b = market_value_multiplier(team_b, team_a, _market_values)

    lam_final = lam_elo * sq_mult_a * mv_mult_a
    mu_final  = mu_elo  * sq_mult_b * mv_mult_b

    # Safety bounds
    lam_final = max(0.05, min(lam_final, lam * 1.60))
    mu_final  = max(0.05, min(mu_final,  mu  * 1.60))

    return float(lam_final), float(mu_final)


def build_all_features(
    schedule: pd.DataFrame,
    raw_df: pd.DataFrame,
    force_reload: bool = False,
) -> dict[tuple[str, str], MatchFeatures]:
    """
    Build MatchFeatures for every row in schedule.

    Parameters
    ----------
    schedule : DataFrame with columns [home_team, away_team, date, city, country]
               using martj42 team names
    raw_df   : full martj42 DataFrame (used for Elo computation)

    Returns
    -------
    dict keyed by (home_team, away_team) -> MatchFeatures
    """
    if not _features_ready or force_reload:
        load_all_feature_stores(raw_df, force=force_reload)

    result: dict[tuple[str, str], MatchFeatures] = {}

    for _, row in schedule.iterrows():
        key = (str(row["home_team"]), str(row["away_team"]))
        result[key] = build_match_features(
            team_a_model=str(row["home_team"]),
            team_b_model=str(row["away_team"]),
            match_date=str(row["date"])[:10],
            city=str(row.get("city", "")),
            country=str(row.get("country", "")),
        )

    return result
