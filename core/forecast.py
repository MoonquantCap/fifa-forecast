"""
Forecast Engine — pluggable prediction model for Comnegolf es Mundial.

To plug in your own logic, subclass ForecastEngine and override `predict_match`.
The rest of the app calls `engine.predict_match(home_id, away_id)` and expects
the dict shape defined below.

Architecture
------------
All heavy computation lives in core/model.py (the ComneGolf algorithm).
This file is a thin routing layer that:
  • holds team/group data loaded once at startup
  • auto-detects group membership so call sites need no extra context
  • delegates every prediction to model.compute_match_probs() — run fresh
    on each call (no caching of MC results)
"""
import math
import random

from core.model import (
    compute_match_probs,
    _normalize_rankings,
    _is_elite,
    _poisson,
    N_ITER_MATCH,
    N_ITER_GROUP,
)


class ForecastEngine:
    """
    Thin wrapper over core/model.py.

    Parameters
    ----------
    teams_data  : dict  — fm.teams
    groups_data : dict  — fm.groups  (optional; enables headstart detection)
    """

    def __init__(self, teams_data: dict, groups_data: dict | None = None):
        self.teams = teams_data
        self._groups = groups_data or {}

        # Build reverse lookup: team_id → group key
        self._team_group: dict[str, str] = {}
        for grp, members in self._groups.items():
            for tid in members:
                self._team_group[tid] = grp

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_match(
        self,
        home_id: str,
        away_id: str,
        neutral: bool = False,
        n_iter: int | None = None,
    ) -> dict:
        """
        Predict a single match via the ComneGolf Monte Carlo model.

        Group membership (for the headstart rule) is detected automatically
        from the loaded groups data — no extra argument needed.

        Returns
        -------
        dict with keys:
          home_win_prob  : float  0-1
          draw_prob      : float  0-1
          away_win_prob  : float  0-1
          expected_home  : float  expected goals (home)
          expected_away  : float  expected goals (away)
          predicted_home : int    most-likely score (home)
          predicted_away : int    most-likely score (away)
          confidence     : float  0-1
          method         : str    label shown in UI
        """
        in_sg = self._in_same_group(home_id, away_id)
        return compute_match_probs(
            home_id, away_id,
            teams=self.teams,
            neutral=neutral,
            in_same_group=in_sg,
            n_iter=n_iter or N_ITER_MATCH,
        )

    def predict_group(self, group_teams: list, fixture_manager) -> dict:
        """
        Simulate an entire group and return qualification probabilities.

        Strategy
        --------
        1. Pre-compute match probabilities once for every pair (MC, N_ITER_GROUP).
        2. Run 5 000 group-stage simulations reusing those pre-computed probs
           but sampling fresh Poisson goal counts each time — fast and accurate.
        """
        # Step 1: pre-compute all pairwise match probabilities
        match_probs: dict[tuple, dict] = {}
        for i, h in enumerate(group_teams):
            for a in group_teams[i + 1:]:
                match_probs[(h, a)] = compute_match_probs(
                    h, a,
                    teams=self.teams,
                    neutral=True,
                    in_same_group=True,
                    n_iter=N_ITER_GROUP,
                )

        # Step 2: group-stage Monte Carlo
        n_sim = 5_000
        advances:    dict[str, int] = {t: 0 for t in group_teams}
        wins_count:  dict[str, int] = {t: 0 for t in group_teams}

        for _ in range(n_sim):
            pts: dict[str, int]   = {t: 0 for t in group_teams}
            gd:  dict[str, int]   = {t: 0 for t in group_teams}
            gf:  dict[str, int]   = {t: 0 for t in group_teams}

            for (h, a), pred in match_probs.items():
                # Sample fresh Poisson goals from pre-computed expected rates
                gh = _poisson(pred["expected_home"])
                ga = _poisson(pred["expected_away"])

                # Use pre-computed outcome probabilities to decide result
                r = random.random()
                if r < pred["home_win_prob"]:
                    # Home win — fix scoreline if Poisson didn't give one
                    if gh <= ga:
                        gh, ga = ga + 1, gh
                    pts[h] += 3
                elif r < pred["home_win_prob"] + pred["draw_prob"]:
                    # Draw
                    ga = gh
                    pts[h] += 1
                    pts[a] += 1
                else:
                    # Away win
                    if ga <= gh:
                        gh, ga = ga, gh + 1
                    pts[a] += 3

                gf[h] += gh
                gf[a] += ga
                gd[h] += gh - ga
                gd[a] += ga - gh

            ranked = sorted(
                group_teams,
                key=lambda t: (pts[t], gd[t], gf[t]),
                reverse=True,
            )
            advances[ranked[0]] += 1
            advances[ranked[1]] += 1
            wins_count[ranked[0]] += 1

        return {
            t: {
                "qualify_prob": round(advances[t]   / n_sim, 3),
                "win_prob":     round(wins_count[t] / n_sim, 3),
            }
            for t in group_teams
        }

    def tournament_winner_probs(self, fixture_manager) -> dict:
        """
        Approximate tournament-winner probabilities using the ComneGolf
        composite strength signal (rank score + WC-titles bonus), exponentiated
        to spread the distribution.

        Not a full bracket MC — kept fast for the home-page chart.
        """
        teams = fixture_manager.teams
        rank_scores = _normalize_rankings(teams)

        strengths: dict[str, float] = {}
        for tid, t in teams.items():
            rs = rank_scores.get(tid, 0.5)
            # WC titles give a small logarithmic bonus (diminishing returns)
            titles = t.get("world_cup_titles", 0)
            titles_bonus = math.log1p(titles) * 0.12
            strengths[tid] = rs + titles_bonus

        # Softmax with temperature=4 to sharpen the distribution
        exp_s = {tid: math.exp(s * 4) for tid, s in strengths.items()}
        total = sum(exp_s.values())
        return {tid: round(v / total, 4) for tid, v in exp_s.items()}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _in_same_group(self, home_id: str, away_id: str) -> bool:
        g1 = self._team_group.get(home_id)
        g2 = self._team_group.get(away_id)
        return g1 is not None and g1 == g2
