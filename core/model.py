"""
Comnegolf es Mundial — Forecast Model
======================================
Composite algorithm pulled fresh on every willful run (no caching of results).

Weight breakdown per match prediction
--------------------------------------
  40%  FIFA Ranking score   — inverse-normalized 0-1 across all 48 teams
  40%  H2H history          — home win rate from historical head-to-head table
  20%  Random walk          — independent Uniform[0,1] draws per iteration

Headstart rule (group stage only)
-----------------------------------
  +0.25 strength bonus for WC winners or top-12 seeds vs non-elite opponents
  in the same group, applied each Monte Carlo iteration.

Output
------
  Monte Carlo over N_ITER iterations → aggregated W / D / L probabilities,
  expected goals, most-likely scoreline, and model metadata.
"""
import math
import random

# ── Iteration budgets ────────────────────────────────────────────────────────
N_ITER_MATCH = 8_000   # per direct predict_match call
N_ITER_GROUP = 8_000   # per match in predict_group pre-computation

# ── Historical H2H table ─────────────────────────────────────────────────────
# (team_a, team_b) → (played, a_wins, draws, b_wins)
# Source: all-time records between World Cup-winning nations (approx.)
_H2H: dict[tuple, tuple] = {
    ("BRA", "ARG"): (109, 44, 25, 40),
    ("BRA", "URU"): (77,  37, 19, 21),
    ("ARG", "URU"): (56,  22, 15, 19),
    ("ITA", "ESP"): (38,  13, 13, 12),
    ("GER", "ENG"): (37,  15, 12, 10),
    ("ITA", "FRA"): (35,  14, 11, 10),
    ("GER", "ITA"): (35,  12, 11, 12),
    ("FRA", "ESP"): (33,  13, 11,  9),
    ("FRA", "ENG"): (32,  13, 10,  9),
    ("GER", "FRA"): (28,  13,  7,  8),
    ("ESP", "ENG"): (28,  12,  9,  7),
    ("BRA", "ENG"): (25,  12,  7,  6),
    ("GER", "ESP"): (24,  10,  7,  7),
    ("BRA", "ITA"): (27,  10,  9,  8),
    ("BRA", "GER"): (23,  12,  4,  7),
    ("BRA", "ESP"): (23,  11,  7,  5),
    ("ITA", "ENG"): (27,  11,  9,  7),
    ("ARG", "GER"): (20,   9,  4,  7),
    ("ARG", "ITA"): (17,   7,  5,  5),
    ("ARG", "ESP"): (18,   8,  5,  5),
    ("ARG", "ENG"): (16,   7,  4,  5),
    ("ARG", "FRA"): (12,   6,  3,  3),
    ("ITA", "URU"): (15,   7,  5,  3),
    ("BRA", "FRA"): (15,   7,  5,  3),
    ("URU", "ESP"): (12,   4,  4,  4),
    ("URU", "ENG"): (10,   4,  3,  3),
    ("FRA", "URU"): (10,   5,  3,  2),
    ("GER", "URU"): (10,   5,  3,  2),
}

# ── Normalised ranking scores ─────────────────────────────────────────────────

def _normalize_rankings(teams: dict) -> dict:
    """
    Return {team_id: score} where score ∈ [0, 1].
    Lower FIFA ranking number (= better team) maps to higher score.
    """
    rankings = {tid: t.get("fifa_ranking", 200) for tid, t in teams.items()}
    max_r = max(rankings.values())
    min_r = min(rankings.values())
    span = max(max_r - min_r, 1)
    return {tid: (max_r - r) / span for tid, r in rankings.items()}


# ── H2H win rate ──────────────────────────────────────────────────────────────

def _h2h_home_rate(home_id: str, away_id: str) -> float:
    """
    Historical home-team win rate (0–1) from the H2H table.
    Returns 0.5 (no information) when the pair is not found.
    """
    key = (home_id, away_id)
    rev = (away_id, home_id)
    if key in _H2H:
        played, aw, _d, _bw = _H2H[key]
        return aw / played if played else 0.5
    if rev in _H2H:
        played, aw, _d, bw = _H2H[rev]
        return bw / played if played else 0.5
    return 0.5


# ── Elite-team detection for headstart ───────────────────────────────────────

def _is_elite(team_id: str, teams: dict) -> bool:
    """
    Returns True if the team is a WC winner OR a top-12 seed by ELO.
    Used for the group-stage headstart rule.
    """
    if teams.get(team_id, {}).get("world_cup_titles", 0) > 0:
        return True
    top12 = sorted(teams, key=lambda t: teams[t].get("elo_rating", 0), reverse=True)[:12]
    return team_id in top12


# ── Poisson goal sampler ──────────────────────────────────────────────────────

def _poisson(lam: float) -> int:
    """Knuth algorithm: sample one Poisson-distributed integer."""
    L = math.exp(-max(0.05, lam))
    p, k = 1.0, 0
    while p > L:
        p *= random.random()
        k += 1
    return k - 1


# ── Core Monte Carlo engine ───────────────────────────────────────────────────

def compute_match_probs(
    home_id: str,
    away_id: str,
    teams: dict,
    neutral: bool = False,
    in_same_group: bool = False,
    n_iter: int = N_ITER_MATCH,
) -> dict:
    """
    Run the ComneGolf Monte Carlo forecast for a single match.

    Each iteration independently samples:
      - A random-walk variate for each team (Uniform[0,1])
      - Poisson goal counts from the resulting expected-goals rates

    The aggregate over n_iter iterations yields probabilistic W/D/L output.

    Parameters
    ----------
    home_id, away_id : FIFA team ID strings (e.g. "BRA", "ARG")
    teams            : full teams dict from FixtureManager
    neutral          : True → no home-field advantage applied
    in_same_group    : True → WC/seed headstart rule is eligible
    n_iter           : Monte Carlo iteration count (higher = smoother, slower)

    Returns
    -------
    dict matching the ForecastEngine.predict_match contract
    """
    # ── Input signals ───────────────────────────────────────────────────
    rank_scores = _normalize_rankings(teams)
    R_home = rank_scores.get(home_id, 0.5)
    R_away = rank_scores.get(away_id, 0.5)

    H_home = _h2h_home_rate(home_id, away_id)
    H_away = 1.0 - H_home

    # Home-field advantage (non-neutral only) — small additive boost
    home_adv = 0.0 if neutral else 0.04

    # Headstart: only applies when both teams compete in the same group
    hs_home = hs_away = 0.0
    if in_same_group:
        elite_home = _is_elite(home_id, teams)
        elite_away = _is_elite(away_id, teams)
        if elite_home and not elite_away:
            hs_home = 0.25
        elif elite_away and not elite_home:
            hs_away = 0.25

    # ── Monte Carlo loop ─────────────────────────────────────────────────
    home_wins = draws = away_wins = 0
    total_gh = total_ga = 0.0

    for _ in range(n_iter):
        # Independent random walk draws for each team (20% weight)
        rw_h = random.random()
        rw_a = random.random()

        # Composite strength scores (all components already in [0,1] range)
        S_home = (0.40 * R_home
                + 0.40 * H_home
                + 0.20 * rw_h
                + home_adv
                + hs_home)
        S_away = (0.40 * R_away
                + 0.40 * H_away
                + 0.20 * rw_a
                + hs_away)

        # Home dominance fraction [0, 1] — drives expected goals
        total_s = S_home + S_away
        p_dom = S_home / total_s if total_s > 0 else 0.5

        # Expected goals: centred at 1.35/1.10 (avg WC scoring rates),
        # scaled ±1.5 by dominance deviation from 0.5
        lam_h = max(0.15, 1.35 + (p_dom - 0.5) * 3.0)
        lam_a = max(0.15, 1.10 - (p_dom - 0.5) * 3.0)

        gh = _poisson(lam_h)
        ga = _poisson(lam_a)

        total_gh += gh
        total_ga += ga

        if gh > ga:
            home_wins += 1
        elif gh < ga:
            away_wins += 1
        else:
            draws += 1

    # ── Aggregate results ────────────────────────────────────────────────
    hw_prob = home_wins / n_iter
    d_prob  = draws     / n_iter
    aw_prob = away_wins / n_iter
    exp_h   = total_gh  / n_iter
    exp_a   = total_ga  / n_iter

    # Predicted scoreline: determine outcome first, then sample a Poisson-consistent score.
    # High-scoring results are naturally tail events via the Poisson distribution.
    if hw_prob >= d_prob and hw_prob >= aw_prob:
        pred_outcome = "home"
    elif d_prob >= hw_prob and d_prob >= aw_prob:
        pred_outcome = "draw"
    else:
        pred_outcome = "away"
    pred_gh = _poisson(exp_h)
    pred_ga = _poisson(exp_a)
    if pred_outcome == "home" and pred_gh <= pred_ga:
        pred_gh = pred_ga + 1
    elif pred_outcome == "away" and pred_ga <= pred_gh:
        pred_ga = pred_gh + 1
    elif pred_outcome == "draw":
        pred_ga = pred_gh

    # Confidence: higher when signals agree (rank gap + H2H signal both present)
    rank_gap = abs(R_home - R_away)
    h2h_signal = abs(H_home - 0.5) * 2.0   # 0 if unknown, 1 if perfect record
    confidence = min(0.95, 0.50 + rank_gap * 0.30 + h2h_signal * 0.15)

    headstart_label = ""
    if hs_home > 0:
        headstart_label = f" | +{int(hs_home*100)}% HS→{home_id}"
    elif hs_away > 0:
        headstart_label = f" | +{int(hs_away*100)}% HS→{away_id}"

    return {
        "home_win_prob":   round(hw_prob, 4),
        "draw_prob":       round(d_prob,  4),
        "away_win_prob":   round(aw_prob, 4),
        "expected_home":   round(exp_h, 2),
        "expected_away":   round(exp_a, 2),
        "predicted_home":  pred_gh,
        "predicted_away":  pred_ga,
        "confidence":      round(confidence, 2),
        "method": (
            f"ComneGolf MC · {n_iter:,} iters · "
            f"40% Rank + 40% H2H + 20% RW{headstart_label}"
        ),
        # Diagnostics (used by UI breakdowns)
        "_rank_home": round(R_home, 3),
        "_rank_away": round(R_away, 3),
        "_h2h_home":  round(H_home, 3),
        "_hs_home":   hs_home,
        "_hs_away":   hs_away,
    }
