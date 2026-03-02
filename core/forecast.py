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

    # ── Group simulation helper (returns full ranking) ──────────────────────

    def _simulate_group_once(self, group_teams, pair_probs):
        """Simulate one group stage using pre-computed pair probabilities.
        Returns the full ranking [1st, 2nd, 3rd, 4th] with stats."""
        pts  = {t: 0 for t in group_teams}
        gd   = {t: 0 for t in group_teams}
        gfor = {t: 0 for t in group_teams}
        for (h, a), pred in pair_probs.items():
            gh = _poisson(pred["expected_home"])
            ga = _poisson(pred["expected_away"])
            r = random.random()
            if r < pred["home_win_prob"]:
                if gh <= ga:
                    gh, ga = ga + 1, gh
                pts[h] += 3
            elif r < pred["home_win_prob"] + pred["draw_prob"]:
                ga = gh
                pts[h] += 1; pts[a] += 1
            else:
                if ga <= gh:
                    gh, ga = ga, gh + 1
                pts[a] += 3
            gfor[h] += gh; gfor[a] += ga
            gd[h] += gh - ga; gd[a] += ga - gh
        ranked = sorted(group_teams,
                        key=lambda t: (pts[t], gd[t], gfor[t]),
                        reverse=True)
        return ranked, pts, gd, gfor

    @staticmethod
    def _build_r32_pairings(group_keys, first, second, third_qual):
        """
        Build 16 R32 pairings for the 48-team format.
        24 teams from top-2 + 8 best 3rd-place teams = 32.

        Pairing scheme (FIFA-style):
          Match 1:  1A vs 3C/D/E/F    Match 9:  1G vs 3I/J/K/L
          Match 2:  2A vs 2C          Match 10: 2G vs 2I
          Match 3:  1B vs 3A/B/E/F    Match 11: 1H vs 3G/H/K/L
          Match 4:  2B vs 2D          Match 12: 2H vs 2J
          Match 5:  1C vs 3A/B/C/D    Match 13: 1I vs 3G/H/I/J
          Match 6:  2E vs 2F          Match 14: 2K vs 2L
          Match 7:  1D vs 3C/D/G/H    Match 15: 1J vs 3I/J/K/L
          Match 8:  1E vs 1F          Match 16: 1K vs 1L

        Simplified: 1st-place teams face 3rd-place or cross-group 2nd;
        we use a deterministic bracket seeding.
        """
        # Simple bracket: pair group winners with best-3rd, and runners-up cross
        # Slot the 8 qualified 3rd-place teams in order
        t3 = list(third_qual)  # already sorted best→worst

        pairs = [
            (first["A"], t3[0] if len(t3) > 0 else second["C"]),
            (second["A"], second["C"]),
            (first["B"], t3[1] if len(t3) > 1 else second["D"]),
            (second["B"], second["D"]),
            (first["C"], t3[2] if len(t3) > 2 else second["E"]),
            (second["E"], second["F"]),
            (first["D"], t3[3] if len(t3) > 3 else second["F"]),
            (first["E"], first["F"]),
            (first["G"], t3[4] if len(t3) > 4 else second["I"]),
            (second["G"], second["I"]),
            (first["H"], t3[5] if len(t3) > 5 else second["J"]),
            (second["H"], second["J"]),
            (first["I"], t3[6] if len(t3) > 6 else second["K"]),
            (second["K"], second["L"]),
            (first["J"], t3[7] if len(t3) > 7 else second["L"]),
            (first["K"], first["L"]),
        ]
        return pairs

    def tournament_winner_probs(self, fixture_manager, n_sim: int = 800) -> dict:
        """
        Full-bracket Monte Carlo tournament winner probabilities using the
        ComneGolf algorithm (40% Rank + 40% H2H + 20% RW + headstart).

        48-team format: top 2 per group (24) + 8 best 3rd-place = 32 in R32.
        Simulates R32→SF.  The Final is NOT decided — SF winners are tallied.
        """
        groups = fixture_manager.groups
        group_keys = sorted(groups.keys())
        rank_scores = _normalize_rankings(self.teams)

        # Pre-compute all group pairwise match probabilities
        group_pair_probs: dict[str, dict[tuple, dict]] = {}
        for gk in group_keys:
            g_teams = groups[gk]
            pair_probs: dict[tuple, dict] = {}
            for i, h in enumerate(g_teams):
                for a in g_teams[i + 1:]:
                    pair_probs[(h, a)] = compute_match_probs(
                        h, a, teams=self.teams,
                        neutral=True, in_same_group=True,
                        n_iter=N_ITER_GROUP,
                    )
            group_pair_probs[gk] = pair_probs

        win_count: dict[str, int] = {tid: 0 for tid in self.teams}

        for _ in range(n_sim):
            first: dict[str, str] = {}
            second: dict[str, str] = {}
            thirds: list[tuple] = []  # (team_id, pts, gd, gf)

            for gk in group_keys:
                g_teams = groups[gk]
                ranked, pts, gd, gfor = self._simulate_group_once(
                    g_teams, group_pair_probs[gk])
                first[gk] = ranked[0]
                second[gk] = ranked[1]
                thirds.append((ranked[2], pts[ranked[2]],
                               gd[ranked[2]], gfor[ranked[2]]))

            # Best 8 of 12 third-place teams
            thirds.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
            third_qual = [t[0] for t in thirds[:8]]

            # Build R32 bracket
            r32_pairs = self._build_r32_pairings(
                group_keys, first, second, third_qual)

            # Simulate KO rounds using pre-computed rank scores for speed
            def _quick_ko(hid, aid):
                rh = rank_scores.get(hid, 0.5)
                ra = rank_scores.get(aid, 0.5)
                p_home = rh / (rh + ra) if (rh + ra) > 0 else 0.5
                return hid if random.random() < p_home else aid

            r16 = [_quick_ko(h, a) for h, a in r32_pairs]
            qf = [_quick_ko(r16[i], r16[i+1]) for i in range(0, 16, 2)]
            sf = [_quick_ko(qf[i], qf[i+1]) for i in range(0, 8, 2)]
            finalists = [_quick_ko(sf[i], sf[i+1]) for i in range(0, 4, 2)]

            for w in finalists:
                win_count[w] += 1

        return {tid: round(cnt / (n_sim * 2), 4)
                for tid, cnt in win_count.items()}

    def simulate_knockout(self, fixture_manager) -> tuple:
        """
        Single deterministic knockout bracket using ComneGolf MC.

        48-team format: top 2 per group (24) + 8 best 3rd = 32 in R32.
        Returns (bracket, qualifiers) where bracket has stages R32→SF.
        The Final is NOT decided — left open.
        """
        groups = fixture_manager.groups
        group_keys = sorted(groups.keys())

        first: dict[str, str] = {}
        second: dict[str, str] = {}
        all_thirds: list[tuple] = []  # (team, group, qualify_prob)

        for gk in group_keys:
            g_teams = groups[gk]
            qp = self.predict_group(g_teams, fixture_manager)
            ranked = sorted(g_teams,
                            key=lambda t: (qp[t]["qualify_prob"], qp[t]["win_prob"]),
                            reverse=True)
            first[gk] = ranked[0]
            second[gk] = ranked[1]
            all_thirds.append((ranked[2], gk, qp[ranked[2]]["qualify_prob"]))

        # Best 8 third-place teams
        all_thirds.sort(key=lambda x: x[2], reverse=True)
        third_qual = [t[0] for t in all_thirds[:8]]

        qualifiers = {}
        for gk in group_keys:
            qualifiers[gk] = [first[gk], second[gk]]

        # Build R32 bracket
        r32_matches_pairs = self._build_r32_pairings(
            group_keys, first, second, third_qual)

        bracket: dict[str, list] = {}

        def _run_round(pairs, stage_name):
            results = []
            winners = []
            for h, a in pairs:
                pred = self.predict_match(h, a, neutral=True)
                winner = h if pred["home_win_prob"] >= pred["away_win_prob"] else a
                results.append({
                    "home": h, "away": a, "winner": winner, "pred": pred,
                })
                winners.append(winner)
            bracket[stage_name] = results
            return winners

        r32_winners = _run_round(r32_matches_pairs, "Round of 32")
        r16_pairs = [(r32_winners[i], r32_winners[i+1])
                     for i in range(0, len(r32_winners), 2)]
        r16_winners = _run_round(r16_pairs, "Round of 16")
        qf_pairs = [(r16_winners[i], r16_winners[i+1])
                    for i in range(0, len(r16_winners), 2)]
        qf_winners = _run_round(qf_pairs, "Quarter-finals")
        sf_pairs = [(qf_winners[i], qf_winners[i+1])
                    for i in range(0, len(qf_winners), 2)]
        _run_round(sf_pairs, "Semi-finals")

        return bracket, qualifiers

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _in_same_group(self, home_id: str, away_id: str) -> bool:
        g1 = self._team_group.get(home_id)
        g2 = self._team_group.get(away_id)
        return g1 is not None and g1 == g2
