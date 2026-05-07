"""
Fixture Manager — loads teams, groups and the official 2026 World Cup schedule.

The 104-match schedule lives in data/schedule.json with all kickoff times in
Argentina time (ART, UTC-3). Group-stage entries reference the four teams of
each group by slot index (0-3); knockout entries are TBD until the group
stage completes.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class FixtureManager:
    def __init__(self):
        self.teams = self._load_teams()
        self.groups = self._load_groups()
        self.tournament = self._load_tournament()
        self.venues_data = self._load_venues()
        self.schedule = self._load_schedule()
        self.fixtures = self._build_fixtures()

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load_teams(self):
        with open(os.path.join(DATA_DIR, "teams.json")) as f:
            return json.load(f)["teams"]

    def _load_groups(self):
        with open(os.path.join(DATA_DIR, "groups.json")) as f:
            return json.load(f)["groups"]

    def _load_tournament(self):
        with open(os.path.join(DATA_DIR, "groups.json")) as f:
            return json.load(f)["tournament"]

    def _load_venues(self):
        with open(os.path.join(DATA_DIR, "groups.json")) as f:
            return json.load(f)["venues"]

    def _load_schedule(self):
        with open(os.path.join(DATA_DIR, "schedule.json")) as f:
            return json.load(f)

    # ── Fixture building ─────────────────────────────────────────────────────

    def _build_fixtures(self):
        fixtures = []
        match_id = 1

        # Group stage — resolve slot indices to actual team IDs
        for entry in self.schedule["group_stage"]:
            grp = entry["group"]
            teams = self.groups[grp]
            home = teams[entry["home_slot"]]
            away = teams[entry["away_slot"]]
            fixtures.append({
                "id": f"GS{match_id:03d}",
                "stage": "Group Stage",
                "group": grp,
                "matchday": entry["matchday"],
                "home": home,
                "away": away,
                "date": entry["date"],
                "time": entry["time"],
                "venue": entry["venue"],
                "city": entry["city"],
                "country": entry["country"],
                "home_score": None,
                "away_score": None,
                "status": "upcoming",
            })
            match_id += 1

        # Knockout stage — teams TBD until group stage finishes
        ko_id = 1
        for entry in self.schedule["knockout"]:
            fixtures.append({
                "id": f"KO{ko_id:03d}",
                "stage": entry["stage"],
                "group": None,
                "matchday": None,
                "home": "TBD",
                "away": "TBD",
                "date": entry["date"],
                "time": entry["time"],
                "venue": entry["venue"],
                "city": entry["city"],
                "country": entry["country"],
                "home_score": None,
                "away_score": None,
                "status": "upcoming",
                "matchup": entry.get("matchup"),
            })
            ko_id += 1

        return fixtures

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_team(self, team_id: str) -> dict:
        return self.teams.get(team_id, {"name": team_id, "flag": "🏳️"})

    def get_group_fixtures(self, group: str) -> list:
        return [m for m in self.fixtures if m["group"] == group]

    def get_stage_fixtures(self, stage: str) -> list:
        return [m for m in self.fixtures if m["stage"] == stage]

    def get_team_fixtures(self, team_id: str) -> list:
        return [m for m in self.fixtures
                if m["home"] == team_id or m["away"] == team_id]

    def update_result(self, match_id: str, home_score: int, away_score: int):
        for m in self.fixtures:
            if m["id"] == match_id:
                m["home_score"] = home_score
                m["away_score"] = away_score
                m["status"] = "completed"
                break

    def get_group_standings(self, group: str) -> list:
        """Return sorted standings list for the group."""
        teams = self.groups[group]
        table = {t: {"team": t, "P": 0, "W": 0, "D": 0, "L": 0,
                     "GF": 0, "GA": 0, "GD": 0, "Pts": 0} for t in teams}

        for m in self.get_group_fixtures(group):
            if m["status"] != "completed":
                continue
            h, a = m["home"], m["away"]
            hg, ag = m["home_score"], m["away_score"]
            table[h]["P"] += 1
            table[a]["P"] += 1
            table[h]["GF"] += hg;  table[h]["GA"] += ag
            table[a]["GF"] += ag;  table[a]["GA"] += hg
            table[h]["GD"] = table[h]["GF"] - table[h]["GA"]
            table[a]["GD"] = table[a]["GF"] - table[a]["GA"]
            if hg > ag:
                table[h]["W"] += 1; table[h]["Pts"] += 3
                table[a]["L"] += 1
            elif ag > hg:
                table[a]["W"] += 1; table[a]["Pts"] += 3
                table[h]["L"] += 1
            else:
                table[h]["D"] += 1; table[h]["Pts"] += 1
                table[a]["D"] += 1; table[a]["Pts"] += 1

        return sorted(table.values(),
                      key=lambda x: (x["Pts"], x["GD"], x["GF"]),
                      reverse=True)

    @property
    def total_matches(self):
        return len(self.fixtures)

    @property
    def completed_matches(self):
        return sum(1 for m in self.fixtures if m["status"] == "completed")

    @property
    def upcoming_matches(self):
        return [m for m in self.fixtures if m["status"] == "upcoming"]
