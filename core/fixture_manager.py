"""
Fixture Manager — loads teams/groups and generates the full 104-match schedule.
Results are stored in session state so every page stays in sync.
"""
import json
import os
from datetime import date, timedelta
from itertools import combinations

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Venue pool (cycled round-robin for group stage) ──────────────────────────
VENUES = [
    ("MetLife Stadium", "New York/NJ", "USA"),
    ("SoFi Stadium", "Inglewood, CA", "USA"),
    ("AT&T Stadium", "Arlington, TX", "USA"),
    ("Levi's Stadium", "Santa Clara, CA", "USA"),
    ("Mercedes-Benz Stadium", "Atlanta, GA", "USA"),
    ("Hard Rock Stadium", "Miami, FL", "USA"),
    ("Arrowhead Stadium", "Kansas City, MO", "USA"),
    ("Gillette Stadium", "Foxborough, MA", "USA"),
    ("Lincoln Financial Field", "Philadelphia, PA", "USA"),
    ("Lumen Field", "Seattle, WA", "USA"),
    ("BC Place", "Vancouver, BC", "Canada"),
    ("BMO Field", "Toronto, ON", "Canada"),
    ("Estadio Azteca", "Mexico City", "Mexico"),
    ("Estadio BBVA", "Monterrey", "Mexico"),
    ("Estadio Akron", "Guadalajara", "Mexico"),
]

# Knockout stage venues (larger stadiums for bigger games)
KO_VENUES = [
    ("MetLife Stadium", "New York/NJ", "USA"),
    ("SoFi Stadium", "Inglewood, CA", "USA"),
    ("AT&T Stadium", "Arlington, TX", "USA"),
    ("Mercedes-Benz Stadium", "Atlanta, GA", "USA"),
    ("Hard Rock Stadium", "Miami, FL", "USA"),
    ("Estadio Azteca", "Mexico City", "Mexico"),
]


class FixtureManager:
    def __init__(self):
        self.teams = self._load_teams()
        self.groups = self._load_groups()
        self.tournament = self._load_tournament()
        self.venues_data = self._load_venues()
        self.fixtures = self._generate_fixtures()

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

    # ── Fixture generation ────────────────────────────────────────────────────

    def _generate_fixtures(self):
        fixtures = []
        match_id = 1

        # ── Group Stage ──────────────────────────────────────────────────────
        # Matchday base dates
        md_bases = {
            1: date(2026, 6, 11),
            2: date(2026, 6, 19),
            3: date(2026, 6, 27),
        }
        groups_list = list(self.groups.keys())  # A-L

        venue_idx = 0
        for g_idx, group in enumerate(groups_list):
            teams = self.groups[group]
            # 3 matchday pairs per group: (0,1)+(2,3), (0,2)+(1,3), (0,3)+(1,2)
            matchdays = [
                [(teams[0], teams[1]), (teams[2], teams[3])],
                [(teams[0], teams[2]), (teams[1], teams[3])],
                [(teams[0], teams[3]), (teams[1], teams[2])],
            ]
            for md_num, pairs in enumerate(matchdays, start=1):
                match_date = md_bases[md_num] + timedelta(days=g_idx % 6)
                kick_offs = ["15:00", "18:00", "21:00"]
                for p_idx, (home, away) in enumerate(pairs):
                    v = VENUES[venue_idx % len(VENUES)]
                    fixtures.append({
                        "id": f"GS{match_id:03d}",
                        "stage": "Group Stage",
                        "group": group,
                        "matchday": md_num,
                        "home": home,
                        "away": away,
                        "date": str(match_date + timedelta(days=p_idx)),
                        "time": kick_offs[p_idx % 3],
                        "venue": v[0],
                        "city": v[1],
                        "country": v[2],
                        "home_score": None,
                        "away_score": None,
                        "status": "upcoming",
                    })
                    match_id += 1
                    venue_idx += 1

        # ── Knockout Stage slots (TBD teams) ─────────────────────────────────
        ko_stages = [
            ("Round of 32", 16, date(2026, 7, 4)),
            ("Round of 16", 8,  date(2026, 7, 8)),
            ("Quarter-finals", 4, date(2026, 7, 12)),
            ("Semi-finals", 2, date(2026, 7, 15)),
            ("3rd Place Play-off", 1, date(2026, 7, 18)),
            ("Final", 1, date(2026, 7, 19)),
        ]
        ko_id = 1
        kv_idx = 0
        for stage_name, num_matches, base_date in ko_stages:
            for i in range(num_matches):
                v = KO_VENUES[kv_idx % len(KO_VENUES)]
                fixtures.append({
                    "id": f"KO{ko_id:03d}",
                    "stage": stage_name,
                    "group": None,
                    "matchday": None,
                    "home": "TBD",
                    "away": "TBD",
                    "date": str(base_date + timedelta(days=i % 4)),
                    "time": "20:00",
                    "venue": v[0],
                    "city": v[1],
                    "country": v[2],
                    "home_score": None,
                    "away_score": None,
                    "status": "upcoming",
                })
                ko_id += 1
                kv_idx += 1

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
