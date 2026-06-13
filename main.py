"""
WC 2026 Goal Prediction -- main entry point.

Usage:
    python main.py                   # full run with defaults
    python main.py --no-download     # use cached martj42 data
    python main.py --sims 50000      # more Monte Carlo iterations
    python main.py --compare         # show prediction vs actual accuracy
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


def main(
    n_sims: int = 10_000,
    force_download: bool = False,
    run_compare: bool = False,
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

    # ── 3. Accuracy vs played matches ────────────────────────────────────
    if run_compare:
        print("\n" + "=" * 60)
        print("STEP 3 -- Model accuracy on played WC 2026 matches")
        print("=" * 60)
        cmp_df = analyse(model)
        compare_summary(cmp_df)
        cmp_df.to_csv(RESULTS_DIR / "model_accuracy.csv", index=False)

    # ── 4. Team ratings ───────────────────────────────────────────────────
    step = 4 if run_compare else 3
    print(f"\n{'=' * 60}")
    print(f"STEP {step} -- WC 2026 team ratings")
    print("=" * 60)
    ratings = model.ratings()
    ratings.to_csv(RESULTS_DIR / "team_ratings.csv", index=False)

    from src.wc2026_teams import WC_TEAMS, to_model_name
    wc_model_names = {to_model_name(t) for t in WC_TEAMS}
    wc_ratings = ratings[ratings["team"].isin(wc_model_names)].copy()
    print(wc_ratings.to_string(index=False))

    # ── 5. Per-match xG ──────────────────────────────────────────────────
    step += 1
    print(f"\n{'=' * 60}")
    print(f"STEP {step} -- Expected goals per remaining group-stage match")
    print("=" * 60)
    xg_df = group_stage_xg(model)
    xg_df.to_csv(RESULTS_DIR / "match_xg.csv", index=False)

    # Only show unplayed matches (those with NaN score in martj42 schedule)
    played_pairs = set(
        zip(
            wc26["home_team"].tolist() + wc26["away_team"].tolist(),
            wc26["away_team"].tolist() + wc26["home_team"].tolist(),
        )
    )
    from src.wc2026_teams import to_model_name as tmn
    remaining = xg_df[
        ~xg_df.apply(
            lambda r: (tmn(r["team_a"]), tmn(r["team_b"])) in played_pairs
                      or (tmn(r["team_b"]), tmn(r["team_a"])) in played_pairs,
            axis=1,
        )
    ]
    print(f"  Remaining fixtures: {len(remaining)}")
    print(remaining.to_string(index=False))

    # ── 6. Monte Carlo simulation ─────────────────────────────────────────
    step += 1
    print(f"\n{'=' * 60}")
    print(f"STEP {step} -- Group-stage simulation ({n_sims:,} runs)")
    print("=" * 60)
    sim = simulate_group_stage(model, n_sims=n_sims)
    sim.to_csv(RESULTS_DIR / "group_stage_predictions.csv", index=False)

    for grp in sorted(sim["group"].unique()):
        grp_df = sim[sim["group"] == grp]
        print(f"\nGroup {grp}")
        print(
            grp_df[["team", "avg_goals_scored", "avg_goals_against",
                     "avg_points", "prob_qualify"]].to_string(index=False)
        )

    # ── 7. Top scorers summary ────────────────────────────────────────────
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
    args = parser.parse_args()
    main(
        n_sims=args.sims,
        force_download=not args.no_download,
        run_compare=args.compare,
    )
