"""
WC 2026 Goal Prediction -- main entry point.

Usage:
    python main.py                   # full run with defaults
    python main.py --no-download     # use cached martj42 data
    python main.py --sims 50000      # more Monte Carlo iterations
    python main.py --compare         # show prediction vs actual accuracy
    python main.py --no-features     # skip contextual features (faster, less accurate)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import download, fill_wc2026_scores, preprocess
from src.model import DixonColesModel
from src.predict import group_stage_xg, simulate_group_stage
from src.compare import analyse, summary as compare_summary

RESULTS_DIR = Path("results")


def _build_actual_scores(wc26: pd.DataFrame) -> dict[tuple[str, str], tuple[int, int]]:
    """Extract played WC 2026 match scores as a dict keyed by (home, away) martj42 names."""
    scores = {}
    for _, row in wc26.iterrows():
        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            scores[(row["home_team"], row["away_team"])] = (
                int(row["home_score"]),
                int(row["away_score"]),
            )
    return scores


def main(
    n_sims: int = 10_000,
    force_download: bool = False,
    run_compare: bool = False,
    use_features: bool = True,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 -- Load & update data")
    print("=" * 60)
    raw = download(force=force_download)

    # Fill in confirmed WC 2026 scores not yet in the martj42 repo
    raw = fill_wc2026_scores(raw)

    df = preprocess(raw, min_year=2014, half_life_days=180)
    print(
        f"  {len(df):,} matches used  |  "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )
    wc26 = df[
        df["tournament"].str.contains("FIFA World Cup", na=False)
        & (df["date"] >= pd.Timestamp("2026-06-11"))
    ]
    print(f"  WC 2026 matches in training: {len(wc26)}")

    # ── 2. Fit model ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 -- Fit Dixon-Coles Poisson model")
    print("=" * 60)
    model = DixonColesModel().fit(df)

    # ── 3. Feature stores ─────────────────────────────────────────────────
    features_dict = None
    if use_features:
        print("\n" + "=" * 60)
        print("STEP 3 -- Load contextual feature stores")
        print("=" * 60)
        from src.features import load_all_feature_stores, build_all_features

        # WC 2026 group-stage schedule from martj42 (has city + country columns)
        wc26_schedule = raw[
            raw["tournament"].str.contains("FIFA World Cup", na=False)
            & (raw["date"] >= pd.Timestamp("2026-06-11"))
        ][["home_team", "away_team", "date", "city", "country"]].copy()

        load_all_feature_stores(raw)
        features_dict = build_all_features(wc26_schedule, raw)
        print(f"  Features built for {len(features_dict)} fixtures")

    # ── 4. Accuracy vs played matches ────────────────────────────────────
    if run_compare:
        step = 4 if use_features else 3
        print("\n" + "=" * 60)
        print(f"STEP {step} -- Model accuracy on played WC 2026 matches")
        print("=" * 60)
        cmp_df = analyse(model, df)
        compare_summary(cmp_df)
        cmp_df.to_csv(RESULTS_DIR / "model_accuracy.csv", index=False)

    # ── 5. Team ratings ───────────────────────────────────────────────────
    step = (4 if use_features else 3) + (1 if run_compare else 0)
    print(f"\n{'=' * 60}")
    print(f"STEP {step} -- WC 2026 team ratings")
    print("=" * 60)
    ratings = model.ratings()
    ratings.to_csv(RESULTS_DIR / "team_ratings.csv", index=False)

    from src.wc2026_teams import WC_TEAMS, to_model_name
    wc_model_names = {to_model_name(t) for t in WC_TEAMS}
    wc_ratings = ratings[ratings["team"].isin(wc_model_names)].copy()
    print(wc_ratings.to_string(index=False))

    # ── 6. Per-match xG ──────────────────────────────────────────────────
    step += 1
    print(f"\n{'=' * 60}")
    print(f"STEP {step} -- Expected goals per remaining group-stage match")
    print("=" * 60)
    xg_df = group_stage_xg(model, features_dict=features_dict)
    xg_df.to_csv(RESULTS_DIR / "match_xg.csv", index=False)

    # Only show unplayed matches.
    # wc26 uses martj42 names; xg_df uses WC display names.
    # _martj42_name translates display→martj42 (e.g. "Czechia"→"Czech Republic")
    from src.data_loader import _martj42_name
    played_pairs = set(
        zip(
            wc26["home_team"].tolist() + wc26["away_team"].tolist(),
            wc26["away_team"].tolist() + wc26["home_team"].tolist(),
        )
    )
    remaining = xg_df[
        ~xg_df.apply(
            lambda r: (_martj42_name(r["team_a"]), _martj42_name(r["team_b"])) in played_pairs
                      or (_martj42_name(r["team_b"]), _martj42_name(r["team_a"])) in played_pairs,
            axis=1,
        )
    ]
    print(f"  Remaining fixtures: {len(remaining)}")
    print(remaining.to_string(index=False))

    # ── 7. Monte Carlo simulation ─────────────────────────────────────────
    step += 1
    print(f"\n{'=' * 60}")
    print(f"STEP {step} -- Group-stage simulation ({n_sims:,} runs)")
    print("=" * 60)

    # Lock actual scores for played matches so simulation reflects reality
    actual_scores = _build_actual_scores(wc26)
    if actual_scores:
        print(f"  Locking {len(actual_scores)} played match score(s) in simulation")

    sim = simulate_group_stage(
        model,
        n_sims=n_sims,
        features_dict=features_dict,
        actual_scores=actual_scores,
    )
    sim.to_csv(RESULTS_DIR / "group_stage_predictions.csv", index=False)

    for grp in sorted(sim["group"].unique()):
        grp_df = sim[sim["group"] == grp]
        print(f"\nGroup {grp}")
        print(
            grp_df[["team", "avg_goals_scored", "avg_goals_against",
                     "avg_points", "prob_qualify"]].to_string(index=False)
        )

    # ── 8. Top scorers summary ────────────────────────────────────────────
    step += 1
    print(f"\n{'=' * 60}")
    print(f"STEP {step} -- Top 20 teams by expected goals scored")
    print("=" * 60)
    print(
        sim[["team", "group", "avg_goals_scored", "p10_goals",
             "p90_goals", "prob_qualify"]].head(20).to_string(index=False)
    )

    print(f"\nDone. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WC 2026 goal prediction model")
    parser.add_argument("--sims",        type=int,  default=10_000)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--compare",     action="store_true",
                        help="Show model accuracy vs played WC 2026 matches")
    parser.add_argument("--no-features", action="store_true",
                        help="Skip contextual features (faster, less accurate)")
    args = parser.parse_args()
    main(
        n_sims=args.sims,
        force_download=not args.no_download,
        run_compare=args.compare,
        use_features=not args.no_features,
    )
