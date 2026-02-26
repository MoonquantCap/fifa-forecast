"""
Forecast Engine — pluggable prediction model for FIFA 2026.

To plug in your own logic, subclass ForecastEngine and override `predict_match`.
The rest of the app calls `engine.predict_match(home_id, away_id)` and expects
the dict shape defined below.
"""
import math


class ForecastEngine:
    """
    Default engine: simple ELO-based model.

    Replace or extend `predict_match` with your custom model when ready.
    """

    def __init__(self, teams_data: dict):
        self.teams = teams_data

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_match(self, home_id: str, away_id: str,
                      neutral: bool = False) -> dict:
        """
        Predict a single match.

        Returns
        -------
        dict with keys:
          home_win_prob  : float  0-1
          draw_prob      : float  0-1
          away_win_prob  : float  0-1
          expected_home  : float  expected goals
          expected_away  : float  expected goals
          predicted_home : int    most-likely scoreline (home)
          predicted_away : int    most-likely scoreline (away)
          confidence     : float  0-1 (model confidence)
          method         : str    label shown in UI
        """
        home = self.teams.get(home_id, {})
        away = self.teams.get(away_id, {})

        home_elo = home.get("elo_rating", 1500)
        away_elo = away.get("elo_rating", 1500)

        # Neutral-venue adjustment
        advantage = 0 if neutral else 60

        # ELO win probability
        p_home = 1 / (1 + 10 ** ((away_elo - home_elo - advantage) / 400))

        # Map win probability → W/D/L split (Dixon-Coles inspired constants)
        draw_base = 0.26
        draw_adj = draw_base - abs(p_home - 0.5) * 0.18
        draw_prob = max(0.12, draw_adj)
        home_win_prob = p_home * (1 - draw_prob)
        away_win_prob = (1 - p_home) * (1 - draw_prob)

        # Expected goals: base ± ELO difference factor
        elo_diff = (home_elo + advantage - away_elo) / 400
        exp_home = max(0.4, 1.35 + elo_diff * 0.8)
        exp_away = max(0.4, 1.10 - elo_diff * 0.8)

        # Most-likely scoreline (Poisson mode)
        pred_home = max(0, round(exp_home - 0.5))
        pred_away = max(0, round(exp_away - 0.5))

        # Confidence: higher when ELO gap is larger
        confidence = min(0.95, 0.55 + abs(home_elo - away_elo) / 2000)

        return {
            "home_win_prob":  round(home_win_prob, 4),
            "draw_prob":      round(draw_prob, 4),
            "away_win_prob":  round(away_win_prob, 4),
            "expected_home":  round(exp_home, 2),
            "expected_away":  round(exp_away, 2),
            "predicted_home": pred_home,
            "predicted_away": pred_away,
            "confidence":     round(confidence, 2),
            "method":         "ELO (placeholder — replace with your model)",
        }

    def predict_group(self, group_teams: list, fixture_manager) -> dict:
        """
        Simulate an entire group and return qualification probabilities.
        Uses Monte-Carlo with 5 000 simulations.
        """
        import random

        n_sim = 5_000
        advances = {t: 0 for t in group_teams}
        wins_count = {t: 0 for t in group_teams}

        group_matches = []
        for i, h in enumerate(group_teams):
            for a in group_teams[i + 1:]:
                group_matches.append((h, a))

        for _ in range(n_sim):
            pts = {t: 0 for t in group_teams}
            gd  = {t: 0 for t in group_teams}
            gf  = {t: 0 for t in group_teams}

            for h, a in group_matches:
                pred = self.predict_match(h, a, neutral=True)
                r = random.random()
                if r < pred["home_win_prob"]:
                    pts[h] += 3
                    gh, ga = _poisson_score(pred["expected_home"],
                                            pred["expected_away"])
                elif r < pred["home_win_prob"] + pred["draw_prob"]:
                    pts[h] += 1; pts[a] += 1
                    gh, ga = _poisson_score(pred["expected_home"],
                                            pred["expected_away"])
                    ga = gh  # force draw scoreline
                else:
                    pts[a] += 3
                    gh, ga = _poisson_score(pred["expected_away"],
                                            pred["expected_home"])
                    gh, ga = ga, gh

                gf[h] += gh; gf[a] += ga
                gd[h] += gh - ga; gd[a] += ga - gh

            ranked = sorted(group_teams,
                            key=lambda t: (pts[t], gd[t], gf[t]),
                            reverse=True)
            advances[ranked[0]] += 1
            advances[ranked[1]] += 1
            wins_count[ranked[0]] += 1

        return {
            t: {
                "qualify_prob": round(advances[t] / n_sim, 3),
                "win_prob":     round(wins_count[t] / n_sim, 3),
            }
            for t in group_teams
        }

    def tournament_winner_probs(self, fixture_manager) -> dict:
        """
        Rough tournament winner probabilities based on average ELO strength.
        (Very simplified — replace with your proper model.)
        """
        teams = fixture_manager.teams
        elos = {tid: t.get("elo_rating", 1500) for tid, t in teams.items()}
        total = sum(math.exp(e / 400) for e in elos.values())
        return {
            tid: round(math.exp(elo / 400) / total, 4)
            for tid, elo in elos.items()
        }


# ── Utility ───────────────────────────────────────────────────────────────────

def _poisson_score(lam_h: float, lam_a: float) -> tuple:
    """Draw random Poisson goals from given rates."""
    import random
    def poisson(lam):
        L = math.exp(-lam)
        p, k = 1.0, 0
        while p > L:
            p *= random.random()
            k += 1
        return k - 1
    return poisson(lam_h), poisson(lam_a)
