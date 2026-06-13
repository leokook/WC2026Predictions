"""
Dixon-Coles Poisson model for international football goal prediction.

Reference: Dixon & Coles (1997), Applied Statistics.

Parameters per team:
  alpha_i  -- attack strength   (higher -> scores more)
  beta_i   -- defense weakness  (higher -> concedes more; mean=1 by normalisation)

Global parameters:
  gamma    -- home-field advantage multiplier
  rho      -- low-score correlation correction (typically small, negative)

Expected goals:
  lam(home) = alpha_home * beta_away * gamma   (if not neutral venue)
  lam(away) = alpha_away * beta_home
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Low-score correction (vectorised)
# ---------------------------------------------------------------------------

def _dc_tau(
    hg: np.ndarray,
    ag: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    tau = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m11] = 1.0 - rho
    return np.maximum(tau, 1e-10)


# ---------------------------------------------------------------------------
# Negative log-likelihood
# ---------------------------------------------------------------------------

def _neg_ll(
    params: np.ndarray,
    n: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    is_neutral: np.ndarray,
) -> float:
    alpha = params[:n]
    beta  = params[n:2*n]
    gamma = params[2*n]
    rho   = params[2*n + 1]

    home_adv = np.where(is_neutral, 1.0, gamma)
    lam = np.maximum(alpha[home_idx] * beta[away_idx] * home_adv, 1e-10)
    mu  = np.maximum(alpha[away_idx] * beta[home_idx],             1e-10)

    tau = _dc_tau(hg, ag, lam, mu, rho)

    log_ll = weights * (
        np.log(tau)
        + hg * np.log(lam) - lam - gammaln(hg + 1)
        + ag * np.log(mu)  - mu  - gammaln(ag + 1)
    )
    return -np.sum(log_ll)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class DixonColesModel:
    """Dixon-Coles Poisson model for predicting international football scores."""

    def __init__(self) -> None:
        self.teams: list[str] = []
        self.alpha: dict[str, float] = {}
        self.beta:  dict[str, float] = {}
        self.gamma: float = 1.1
        self.rho:   float = -0.1
        self._avg_alpha: float = 1.0
        self._avg_beta:  float = 1.0
        self.fitted: bool = False

    def fit(self, df: pd.DataFrame, weight_col: str = "weight") -> "DixonColesModel":
        """
        Fit on a preprocessed DataFrame with columns:
          home_team, away_team, home_score, away_score, neutral, <weight_col>
        """
        all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams = all_teams
        n = len(all_teams)
        idx = {t: i for i, t in enumerate(all_teams)}

        home_idx   = np.array([idx[t] for t in df["home_team"]])
        away_idx   = np.array([idx[t] for t in df["away_team"]])
        hg         = df["home_score"].values.astype(int)
        ag         = df["away_score"].values.astype(int)
        weights    = df[weight_col].values.astype(float)
        is_neutral = df["neutral"].values.astype(bool)

        x0 = np.ones(2 * n + 2)
        x0[2*n]     = 1.15
        x0[2*n + 1] = -0.10

        bounds = (
            [(0.01, 10.0)] * n    # alpha
            + [(0.01, 10.0)] * n  # beta
            + [(1.0,  2.0)]       # gamma
            + [(-0.99, 0.0)]      # rho
        )

        print(f"  Fitting Dixon-Coles on {len(df):,} matches, {n} teams ...")

        result = minimize(
            _neg_ll,
            x0,
            args=(n, home_idx, away_idx, hg, ag, weights, is_neutral),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-8},
        )

        if not result.success:
            warnings.warn(f"Optimisation warning: {result.message}")

        alpha_vals = result.x[:n]
        beta_vals  = result.x[n:2*n]

        # Normalise identifiability: mean(beta) = 1
        scale = beta_vals.mean()
        alpha_vals = alpha_vals * scale
        beta_vals  = beta_vals  / scale

        self.gamma = float(result.x[2*n])
        self.rho   = float(result.x[2*n + 1])
        self.alpha = dict(zip(all_teams, alpha_vals))
        self.beta  = dict(zip(all_teams, beta_vals))
        self._avg_alpha = float(alpha_vals.mean())
        self._avg_beta  = float(beta_vals.mean())
        self.fitted = True

        print(
            f"  Done. home_adv={self.gamma:.3f}  "
            f"rho={self.rho:.4f}  "
            f"avg_attack={self._avg_alpha:.3f}"
        )
        return self

    def _lookup(self, team: str) -> tuple[float, float]:
        a = self.alpha.get(team, self._avg_alpha)
        b = self.beta.get(team,  self._avg_beta)
        return a, b

    def predict_xg(
        self, team_a: str, team_b: str, neutral: bool = True
    ) -> tuple[float, float]:
        """Expected goals: (team_a xG, team_b xG)."""
        if not self.fitted:
            raise RuntimeError("Call .fit() first.")
        a_a, b_a = self._lookup(team_a)
        a_b, b_b = self._lookup(team_b)
        home_adv = 1.0 if neutral else self.gamma
        lam = a_a * b_b * home_adv
        mu  = a_b * b_a
        return float(lam), float(mu)

    def simulate_match(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw n Poisson-sampled scorelines."""
        lam, mu = self.predict_xg(team_a, team_b, neutral)
        rng = rng or np.random.default_rng()
        return rng.poisson(lam, n), rng.poisson(mu, n)

    def ratings(self) -> pd.DataFrame:
        """Team ratings sorted by overall strength (attack / defense_weakness)."""
        rows = [
            {
                "team":    t,
                "attack":  round(a, 4),
                "defense": round(1.0 / b, 4),  # higher = better defence
                "overall": round(a / b, 4),
            }
            for t, (a, b) in [(t, self._lookup(t)) for t in self.teams]
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("overall", ascending=False)
            .reset_index(drop=True)
        )
