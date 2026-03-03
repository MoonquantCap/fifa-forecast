"""
Comnegolf es Mundial
====================
Run with:  streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime

from core.fixture_manager import FixtureManager
from core.forecast import ForecastEngine

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Comnegolf es Mundial",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a2a1a 0%, #0d3322 100%);
    border-right: 2px solid #FFD700;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem; padding: 4px 0;
}

/* ── Cards ── */
.card {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.card:hover { border-color: #FFD700; }

.match-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4e;
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 14px;
    text-align: center;
}
.match-card:hover { border-color: #FFD700; box-shadow: 0 4px 20px rgba(255,215,0,0.15); }

/* ── Team badges ── */
.team-name { font-weight: 700; font-size: 1.05rem; }
.team-flag { font-size: 2rem; }
.vs-text   { color: #FFD700; font-weight: 900; font-size: 1.3rem; }
.score-box { font-size: 1.8rem; font-weight: 900; color: #FFD700; }

/* ── Metrics ── */
.metric-card {
    background: linear-gradient(135deg, #0a2a1a, #0d3322);
    border: 1px solid #1a5e3a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-size: 2.4rem; font-weight: 900; color: #FFD700; }
.metric-label { font-size: 0.8rem; color: #aaa; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

/* ── Headers ── */
.page-header {
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #FFD700, #FFA500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.page-sub { color: #888; font-size: 0.9rem; margin-bottom: 24px; }

/* ── Probability bars ── */
.prob-bar-wrap { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.prob-label    { min-width: 80px; font-size: 0.8rem; color: #aaa; }
.prob-bar      { height: 10px; border-radius: 5px; }

/* ── Table ── */
.standings-table th { color: #FFD700 !important; }

/* ── Bracket ── */
.bracket-match {
    background: #16213e;
    border: 1px solid #2a2a4e;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.85rem;
    min-width: 160px;
}
.bracket-match .winner { color: #FFD700; font-weight: 700; }
.bracket-match .loser  { color: #666; }

/* ── Countdown ── */
.countdown-wrap { display: flex; gap: 16px; justify-content: center; margin: 16px 0; }
.countdown-block {
    background: #0a2a1a;
    border: 2px solid #FFD700;
    border-radius: 12px;
    padding: 14px 20px;
    text-align: center;
    min-width: 80px;
}
.countdown-num  { font-size: 2.2rem; font-weight: 900; color: #FFD700; }
.countdown-unit { font-size: 0.7rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }

/* ── Prediction form ── */
.pred-result-btn {
    border: 2px solid #2a2a4e; border-radius: 8px;
    padding: 8px 16px; cursor: pointer; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA & ENGINE (cached for performance)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_app():
    fm = FixtureManager()
    engine = ForecastEngine(fm.teams, fm.groups)
    return fm, engine


# ═══════════════════════════════════════════════════════════════
# TRANSLATIONS
# ═══════════════════════════════════════════════════════════════
STAGE_TR = {
    "Group Stage": "Fase de Grupos",
    "Round of 32": "Dieciseisavos de Final",
    "Round of 16": "Octavos de Final",
    "Quarter-finals": "Cuartos de Final",
    "Semi-finals": "Semifinales",
    "3rd Place Play-off": "Partido por el 3er Puesto",
    "Final": "Final",
}

_MONTH_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
             "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def team_badge(team_id: str, fm: FixtureManager, size: str = "md") -> str:
    t = fm.get_team(team_id)
    flag = t.get("flag", "🏳️")
    name = t.get("name", team_id)
    fs = "1.8rem" if size == "md" else "1.2rem"
    return f'<span style="font-size:{fs}">{flag}</span> <span class="team-name">{name}</span>'


def prob_bar_html(home_prob: float, draw_prob: float, away_prob: float,
                  home_name: str, away_name: str) -> str:
    hw = round(home_prob * 100, 1)
    dw = round(draw_prob * 100, 1)
    aw = round(away_prob * 100, 1)
    return f"""
    <div style="display:flex; border-radius:8px; overflow:hidden; height:28px; margin:12px 0;">
        <div style="width:{hw}%; background:#1a8a4a; display:flex; align-items:center;
                    justify-content:center; font-size:0.8rem; font-weight:700; color:white;">
            {hw}%
        </div>
        <div style="width:{dw}%; background:#555; display:flex; align-items:center;
                    justify-content:center; font-size:0.8rem; color:white;">
            {dw}%
        </div>
        <div style="width:{aw}%; background:#8a1a1a; display:flex; align-items:center;
                    justify-content:center; font-size:0.8rem; font-weight:700; color:white;">
            {aw}%
        </div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#888;">
        <span>{home_name}</span><span>Empate</span><span>{away_name}</span>
    </div>""".strip()


def days_until(target: date) -> int:
    return max(0, (target - date.today()).days)


def format_date(d: str) -> str:
    try:
        dt = datetime.strptime(d, "%Y-%m-%d")
        return f"{dt.day} {_MONTH_ES[dt.month - 1]} {dt.year}"
    except Exception:
        return d


# ═══════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════
def page_home(fm: FixtureManager, engine: ForecastEngine):
    st.markdown('<div class="page-header">⚽ Comnegolf es Mundial</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">El pronosticador definitivo del torneo — EE.UU. · Canadá · México</div>',
                unsafe_allow_html=True)

    # ── Countdown ────────────────────────────────────────────────
    wc_start = date(2026, 6, 11)
    d = days_until(wc_start)
    hours   = d * 24
    minutes = hours * 60
    st.markdown(f"""
    <div class="countdown-wrap">
        <div class="countdown-block">
            <div class="countdown-num">{d}</div>
            <div class="countdown-unit">Días</div>
        </div>
        <div class="countdown-block">
            <div class="countdown-num">{hours}</div>
            <div class="countdown-unit">Horas</div>
        </div>
        <div class="countdown-block">
            <div class="countdown-num">{minutes}</div>
            <div class="countdown-unit">Minutos</div>
        </div>
    </div>
    <p style="text-align:center; color:#888; font-size:0.85rem;">Hasta el inicio · 11 de junio de 2026</p>
    """, unsafe_allow_html=True)

    # ── Key Stats ────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        ("48", "Equipos"),
        ("12", "Grupos"),
        ("104", "Partidos"),
        ("3", "Países Anfitriones"),
        ("16", "Sedes"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4, c5], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tournament Winner Odds ───────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("🏆 Pronóstico del Ganador del Torneo")
        win_probs = engine.tournament_winner_probs(fm)
        # Top 12 favourites
        top = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)[:12]

        names, flags, probs = [], [], []
        for tid, prob in top:
            t = fm.get_team(tid)
            names.append(t["name"])
            flags.append(t.get("flag", ""))
            probs.append(round(prob * 100, 1))

        display_names = [f"{f} {n}" for f, n in zip(flags, names)]
        fig = go.Figure(go.Bar(
            x=probs, y=display_names,
            orientation="h",
            marker=dict(
                color=probs,
                colorscale=[[0, "#1a5e3a"], [0.5, "#FFD700"], [1, "#FF6B00"]],
            ),
            text=[f"{p}%" for p in probs],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
            margin=dict(l=10, r=60, t=10, b=10),
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📅 Calendario del Torneo")
        timeline = [
            ("11 Jun – 3 Jul", "Fase de Grupos", "🟩"),
            ("4 – 7 Jul",       "Dieciseisavos de Final", "🟨"),
            ("8 – 11 Jul",      "Octavos de Final", "🟧"),
            ("12 – 13 Jul",     "Cuartos de Final", "🟥"),
            ("15 – 16 Jul",     "Semifinales",   "🔵"),
            ("18 Jul",          "3er Puesto",     "⚪"),
            ("19 Jul",          "⭐ Final",       "🏆"),
        ]
        for dates, stage, icon in timeline:
            st.markdown(f"""
            <div class="card" style="padding:10px 14px; margin-bottom:6px;">
                <span style="color:#888; font-size:0.75rem;">{dates}</span><br>
                <span style="font-weight:600;">{icon} {stage}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Confederation breakdown ──────────────────────────────────
    st.subheader("🌍 Equipos por Confederación")
    confs = {}
    for t in fm.teams.values():
        c = t["confederation"]
        confs[c] = confs.get(c, 0) + 1
    conf_colors = {
        "UEFA": "#1f77b4", "CONMEBOL": "#2ca02c",
        "CONCACAF": "#ff7f0e", "CAF": "#d62728",
        "AFC": "#9467bd", "OFC": "#8c564b",
    }
    fig2 = go.Figure(go.Pie(
        labels=list(confs.keys()),
        values=list(confs.values()),
        hole=0.5,
        marker=dict(colors=[conf_colors.get(c, "#999") for c in confs.keys()]),
        textinfo="label+value",
        textfont=dict(color="white"),
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        height=280,
        margin=dict(t=0, b=0, l=0, r=0),
    )
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: FIXTURES
# ═══════════════════════════════════════════════════════════════
def page_fixtures(fm: FixtureManager, engine: ForecastEngine):
    st.markdown('<div class="page-header">📅 Partidos</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Los 104 partidos — filtra por fase, grupo o equipo</div>',
                unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        stage_opts = ["All Stages", "Group Stage", "Round of 32", "Round of 16",
                      "Quarter-finals", "Semi-finals", "3rd Place Play-off", "Final"]
        selected_stage = st.selectbox("Fase", stage_opts,
                                      format_func=lambda s: "Todas las Fases" if s == "All Stages" else STAGE_TR.get(s, s))
    with col2:
        group_opts = ["All Groups"] + list(fm.groups.keys())
        selected_group = st.selectbox("Grupo", group_opts,
                                      format_func=lambda g: "Todos los Grupos" if g == "All Groups" else f"Grupo {g}")
    with col3:
        team_names = {tid: f"{t['flag']} {t['name']}" for tid, t in fm.teams.items()}
        team_opts = ["All Teams"] + sorted(team_names.values())
        selected_team_display = st.selectbox("Equipo", team_opts,
                                             format_func=lambda t: "Todos los Equipos" if t == "All Teams" else t)

    selected_team_id = None
    if selected_team_display != "All Teams":
        for tid, display in team_names.items():
            if display == selected_team_display:
                selected_team_id = tid
                break

    # ── Filter fixtures ───────────────────────────────────────────
    matches = fm.fixtures
    if selected_stage != "All Stages":
        matches = [m for m in matches if m["stage"] == selected_stage]
    if selected_group != "All Groups":
        matches = [m for m in matches if m.get("group") == selected_group]
    if selected_team_id:
        matches = [m for m in matches
                   if m["home"] == selected_team_id or m["away"] == selected_team_id]

    st.markdown(f"**{len(matches)} partidos** encontrados")

    # ── Match cards ───────────────────────────────────────────────
    show_predictions = st.toggle("Mostrar Predicciones IA", value=True)

    for m in matches:
        ht = fm.get_team(m["home"])
        at = fm.get_team(m["away"])
        h_flag = ht.get("flag", "🏳️") if m["home"] != "TBD" else "❓"
        a_flag = at.get("flag", "🏳️") if m["away"] != "TBD" else "❓"
        h_name = ht.get("name", m["home"]) if m["home"] != "TBD" else "PD"
        a_name = at.get("name", m["away"]) if m["away"] != "TBD" else "PD"

        group_tag = f"Grupo {m['group']}" if m["group"] else ""
        stage_tag = STAGE_TR.get(m["stage"], m["stage"])

        score_html = ""
        if m["status"] == "completed":
            score_html = f'<span class="score-box">{m["home_score"]} – {m["away_score"]}</span>'
        elif m["home"] != "TBD" and show_predictions:
            pred = engine.predict_match(m["home"], m["away"])
            score_html = (f'<span style="color:#888; font-size:0.9rem;">'
                          f'Pred: {pred["predicted_home"]}–{pred["predicted_away"]}</span>')

        pred_bar = ""
        if show_predictions and m["home"] != "TBD" and m["status"] != "completed":
            pred = engine.predict_match(m["home"], m["away"])
            pred_bar = prob_bar_html(
                pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"],
                h_name, a_name,
            )

        st.markdown(f"""
        <div class="match-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="color:#888; font-size:0.75rem;">{format_date(m['date'])} · {m['time']}</span>
                <span style="background:#0a2a1a; color:#FFD700; border-radius:20px;
                             padding:2px 10px; font-size:0.72rem; font-weight:700;">
                    {stage_tag}{f' · {group_tag}' if group_tag else ''}
                </span>
                <span style="color:#888; font-size:0.75rem;">📍 {m['city']}</span>
            </div>
            <div style="display:flex; align-items:center; justify-content:center; gap:24px;">
                <div style="text-align:right; min-width:140px;">
                    <div style="font-size:2rem;">{h_flag}</div>
                    <div style="font-weight:700;">{h_name}</div>
                </div>
                <div style="text-align:center; min-width:80px;">
                    {score_html if score_html else '<span class="vs-text">VS</span>'}
                </div>
                <div style="text-align:left; min-width:140px;">
                    <div style="font-size:2rem;">{a_flag}</div>
                    <div style="font-weight:700;">{a_name}</div>
                </div>
            </div>
            {pred_bar}
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: GROUPS
# ═══════════════════════════════════════════════════════════════
def page_groups(fm: FixtureManager, engine: ForecastEngine):
    st.markdown('<div class="page-header">👥 Fase de Grupos</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Clasificación, probabilidades de clasificación y partidos del grupo</div>',
                unsafe_allow_html=True)

    selected_group = st.selectbox("Seleccionar Grupo", list(fm.groups.keys()),
                                  format_func=lambda g: f"Grupo {g}")

    group_teams = fm.groups[selected_group]

    # ── Simulate qualification probabilities ─────────────────────
    with st.spinner("Ejecutando simulación Monte-Carlo…"):
        qual_probs = engine.predict_group(group_teams, fm)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # ── Standings table ─────────────────────────────────────
        st.subheader(f"Clasificación del Grupo {selected_group}")
        standings = fm.get_group_standings(selected_group)

        if all(row["P"] == 0 for row in standings):
            st.info("Sin resultados aún — mostrando clasificación previa al torneo por puntuación ELO.")
            standings = sorted(
                [{"team": t, "P": 0, "W": 0, "D": 0, "L": 0,
                  "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
                 for t in group_teams],
                key=lambda x: fm.teams.get(x["team"], {}).get("elo_rating", 0),
                reverse=True,
            )

        rows = []
        for i, row in enumerate(standings):
            t = fm.get_team(row["team"])
            qual = qual_probs.get(row["team"], {})
            q_pct = round(qual.get("qualify_prob", 0) * 100)
            w_pct = round(qual.get("win_prob", 0) * 100)
            pos_emoji = ["🥇", "🥈", "🥉", "4️⃣"][i]
            rows.append({
                "Pos": pos_emoji,
                "Equipo": f"{t.get('flag','🏳️')} {t.get('name', row['team'])}",
                "PJ": row["P"], "G": row["W"], "E": row["D"], "P": row["L"],
                "GF": row["GF"], "GC": row["GA"], "DG": row["GD"],
                "Pts": row["Pts"],
                "Clasif. %": f"{q_pct}%",
                "Ganar Grupo %": f"{w_pct}%",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "Clasif. %":   st.column_config.TextColumn("Clasif. %"),
                         "Ganar Grupo %": st.column_config.TextColumn("Ganar Grupo %"),
                     })

    with col_r:
        # ── Qualification probability donut chart ────────────────
        st.subheader("Probabilidad de Clasificación")
        team_labels, team_probs, team_colors = [], [], []
        colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#666666"]
        for i, tid in enumerate(
            sorted(group_teams, key=lambda t: qual_probs.get(t, {}).get("qualify_prob", 0),
                   reverse=True)
        ):
            t = fm.get_team(tid)
            team_labels.append(f"{t.get('flag','🏳️')} {t.get('name', tid)}")
            team_probs.append(qual_probs.get(tid, {}).get("qualify_prob", 0))
            team_colors.append(colors[i])

        fig = go.Figure(go.Bar(
            x=team_labels,
            y=[p * 100 for p in team_probs],
            marker_color=team_colors,
            text=[f"{p*100:.0f}%" for p in team_probs],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(title="Probabilidad (%)", showgrid=True, gridcolor="#2a2a4e"),
            xaxis=dict(showgrid=False),
            margin=dict(t=30, b=10),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Group Fixtures ────────────────────────────────────────────
    st.subheader(f"Partidos del Grupo {selected_group}")
    for m in fm.get_group_fixtures(selected_group):
        ht = fm.get_team(m["home"])
        at = fm.get_team(m["away"])
        pred = engine.predict_match(m["home"], m["away"])
        score = (f'{m["home_score"]} – {m["away_score"]}'
                 if m["status"] == "completed"
                 else f'Pred: {pred["predicted_home"]}–{pred["predicted_away"]}')

        st.markdown(f"""
        <div class="match-card" style="padding:12px;">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:16px;">
                <div style="flex:1; text-align:right;">
                    <span style="font-size:1.4rem;">{ht.get('flag','🏳️')}</span>
                    <span style="font-weight:700; margin-left:6px;">{ht.get('name', m['home'])}</span>
                </div>
                <div style="text-align:center; min-width:120px;">
                    <div style="color:#FFD700; font-weight:900;">{score}</div>
                    <div style="color:#888; font-size:0.72rem;">JN{m['matchday']} · {format_date(m['date'])}</div>
                </div>
                <div style="flex:1;">
                    <span style="font-size:1.4rem;">{at.get('flag','🏳️')}</span>
                    <span style="font-weight:700; margin-left:6px;">{at.get('name', m['away'])}</span>
                </div>
            </div>
            {prob_bar_html(pred['home_win_prob'], pred['draw_prob'], pred['away_win_prob'],
                           ht.get('name', m['home']), at.get('name', m['away']))}
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: BRACKET
# ═══════════════════════════════════════════════════════════════
def page_bracket(fm: FixtureManager, engine: ForecastEngine):
    st.markdown('<div class="page-header">🏆 Cuadro Eliminatorio</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Cuadro visual — los equipos avanzan cuando se registran los resultados de la fase de grupos</div>',
                unsafe_allow_html=True)

    ko_stages = ["Round of 32", "Round of 16", "Quarter-finals",
                 "Semi-finals", "Final"]

    cols = st.columns(len(ko_stages))
    for col, stage in zip(cols, ko_stages):
        matches = fm.get_stage_fixtures(stage)
        with col:
            st.markdown(f"**{STAGE_TR.get(stage, stage)}**")
            for m in matches:
                ht = fm.get_team(m["home"])
                at = fm.get_team(m["away"])
                h_name = ht.get("name", m["home"]) if m["home"] != "TBD" else "PD"
                a_name = at.get("name", m["away"]) if m["away"] != "TBD" else "PD"
                h_flag = ht.get("flag", "❓") if m["home"] != "TBD" else "❓"
                a_flag = at.get("flag", "❓") if m["away"] != "TBD" else "❓"

                # Determine winner display
                winner_home = m["status"] == "completed" and m.get("home_score", 0) > m.get("away_score", 0)
                winner_away = m["status"] == "completed" and m.get("away_score", 0) > m.get("home_score", 0)

                score_display = ""
                if m["status"] == "completed":
                    score_display = f'{m["home_score"]}–{m["away_score"]}'
                elif m["home"] != "TBD":
                    pred = engine.predict_match(m["home"], m["away"])
                    score_display = f'~{pred["predicted_home"]}–{pred["predicted_away"]}'

                h_style = "color:#FFD700; font-weight:700;" if winner_home else ""
                a_style = "color:#FFD700; font-weight:700;" if winner_away else ""

                st.markdown(f"""
                <div class="bracket-match">
                    <div style="{h_style}">{h_flag} {h_name}</div>
                    <div style="color:#FFD700; font-size:0.75rem; text-align:center;
                                padding:2px 0;">{score_display if score_display else "–"}</div>
                    <div style="{a_style}">{a_flag} {a_name}</div>
                    <div style="color:#555; font-size:0.68rem; margin-top:4px;">
                        📍 {m['city']} · {format_date(m['date'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── 3rd Place match ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("🥉 Partido por el 3er Puesto · 18 de julio")
    third = fm.get_stage_fixtures("3rd Place Play-off")
    if third:
        m = third[0]
        c1, c2, c3 = st.columns([2, 1, 2])
        ht = fm.get_team(m["home"])
        at = fm.get_team(m["away"])
        with c1:
            st.markdown(f"""
            <div style="text-align:right; padding:10px;">
                <div style="font-size:2rem;">{ht.get('flag','❓')}</div>
                <div style="font-weight:700;">{ht.get('name', m['home'])}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown('<div style="text-align:center; padding:20px 0; font-size:1.5rem; color:#FFD700;">VS</div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style="padding:10px;">
                <div style="font-size:2rem;">{at.get('flag','❓')}</div>
                <div style="font-weight:700;">{at.get('name', m['away'])}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ═══════════════════════════════════════════════════════════════
def page_predictions(fm: FixtureManager, engine: ForecastEngine):
    st.markdown('<div class="page-header">🎯 Mis Predicciones</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Envía tus predicciones de partidos y sigue tu puntuación</div>',
                unsafe_allow_html=True)

    if "user_predictions" not in st.session_state:
        st.session_state.user_predictions = {}  # match_id → {"home": int, "away": int}

    # ── Filter: only group stage for now ─────────────────────────
    stage_keys = ["Group Stage", "Round of 32",
                  "Round of 16", "Quarter-finals",
                  "Semi-finals", "Final"]
    stage_filter = st.selectbox("Fase a predecir", stage_keys,
                                format_func=lambda s: STAGE_TR.get(s, s))

    group_filter = None
    if stage_filter == "Group Stage":
        group_filter = st.selectbox("Grupo", list(fm.groups.keys()),
                                    format_func=lambda g: f"Grupo {g}")

    matches = [m for m in fm.fixtures if m["stage"] == stage_filter
               and m["home"] != "TBD"]
    if group_filter:
        matches = [m for m in matches if m.get("group") == group_filter]

    st.markdown(f"**{len(matches)} partidos disponibles para predecir**")
    st.markdown("---")

    pred_count = 0
    for m in matches:
        ht = fm.get_team(m["home"])
        at = fm.get_team(m["away"])
        ai_pred = engine.predict_match(m["home"], m["away"])
        existing = st.session_state.user_predictions.get(m["id"], {})

        with st.container():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        margin-bottom:6px;">
                <span style="color:#888; font-size:0.8rem;">{format_date(m['date'])} · {m['time']}</span>
                <span style="background:#0a2a1a; color:#FFD700; border-radius:20px;
                             padding:2px 10px; font-size:0.72rem;">
                    {STAGE_TR.get(m['stage'], m['stage'])}{f" · Grupo {m['group']}" if m['group'] else ""}
                </span>
            </div>
            <div style="display:flex; align-items:center; justify-content:center; gap:12px; margin-bottom:8px;">
                <span style="font-size:1.5rem;">{ht.get('flag','🏳️')}</span>
                <span style="font-weight:700;">{ht.get('name', m['home'])}</span>
                <span style="color:#FFD700; font-weight:900; margin:0 8px;">VS</span>
                <span style="font-weight:700;">{at.get('name', m['away'])}</span>
                <span style="font-size:1.5rem;">{at.get('flag','🏳️')}</span>
            </div>
            <div style="color:#666; font-size:0.78rem; text-align:center; margin-bottom:8px;">
                🤖 IA: {ai_pred['predicted_home']}–{ai_pred['predicted_away']}
                (Conf: {ai_pred['confidence']*100:.0f}%)
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                h_goals = st.number_input(
                    f"Goles de {ht.get('name', m['home'])}",
                    min_value=0, max_value=20,
                    value=existing.get("home", ai_pred["predicted_home"]),
                    key=f"home_{m['id']}",
                    label_visibility="collapsed",
                )
            with c2:
                st.markdown("<div style='text-align:center; padding-top:8px; color:#888;'>–</div>",
                            unsafe_allow_html=True)
            with c3:
                a_goals = st.number_input(
                    f"Goles de {at.get('name', m['away'])}",
                    min_value=0, max_value=20,
                    value=existing.get("away", ai_pred["predicted_away"]),
                    key=f"away_{m['id']}",
                    label_visibility="collapsed",
                )

            if st.button(f"💾 Guardar predicción", key=f"save_{m['id']}"):
                st.session_state.user_predictions[m["id"]] = {
                    "home": h_goals, "away": a_goals,
                    "match_id": m["id"],
                }
                st.success(f"Guardado: {ht.get('name')} {h_goals} – {a_goals} {at.get('name')}")
                pred_count += 1

            st.markdown("---")

    # ── Prediction summary ────────────────────────────────────────
    total_preds = len(st.session_state.user_predictions)
    if total_preds:
        st.subheader(f"📊 Tus Predicciones ({total_preds} guardadas)")
        rows = []
        for mid, pred in st.session_state.user_predictions.items():
            match = next((m for m in fm.fixtures if m["id"] == mid), None)
            if match:
                ht = fm.get_team(match["home"])
                at = fm.get_team(match["away"])
                rows.append({
                    "Partido": f"{ht.get('flag','')} {ht.get('name','')} vs {at.get('flag','')} {at.get('name','')}",
                    "Tu Marcador": f"{pred['home']} – {pred['away']}",
                    "Fecha": format_date(match["date"]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.button("🗑️ Borrar Todas las Predicciones", type="secondary"):
            st.session_state.user_predictions = {}
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════
def page_analytics(fm: FixtureManager, engine: ForecastEngine):
    st.markdown('<div class="page-header">📊 Análisis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Comparaciones de equipos, puntuaciones ELO, rendimiento histórico</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🆚 Comparador Directo", "🌍 Ranking ELO", "🏅 Rendimiento Histórico"])

    # ── Tab 1: Head-to-Head ─────────────────────────────────────
    with tab1:
        st.subheader("Simulador de Enfrentamientos Directos")
        team_options = {tid: f"{t['flag']} {t['name']}" for tid, t in fm.teams.items()}

        col_a, col_b = st.columns(2)
        with col_a:
            home_team = st.selectbox("Equipo Local", sorted(team_options.values()),
                                     index=0, key="h2h_home")
        with col_b:
            away_team = st.selectbox("Equipo Visitante", sorted(team_options.values()),
                                     index=5, key="h2h_away")

        home_id = next(tid for tid, v in team_options.items() if v == home_team)
        away_id = next(tid for tid, v in team_options.items() if v == away_team)

        neutral = st.checkbox("Sede neutral", value=False)

        if home_id != away_id:
            pred = engine.predict_match(home_id, away_id, neutral=neutral)
            ht = fm.get_team(home_id)
            at = fm.get_team(away_id)

            c1, c2, c3 = st.columns([3, 2, 3])
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2.5rem;">{ht.get('flag','🏳️')}</div>
                    <div class="metric-value">{round(pred['home_win_prob']*100)}%</div>
                    <div class="metric-label">{ht.get('name','')}</div>
                    <div style="color:#888; font-size:0.8rem; margin-top:8px;">
                        ELO: {ht.get('elo_rating','?')}<br>
                        Ranking: #{ht.get('fifa_ranking','?')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{round(pred['draw_prob']*100)}%</div>
                    <div class="metric-label">Empate</div>
                    <div style="color:#888; font-size:0.8rem; margin-top:8px;">
                        Pred: {pred['predicted_home']}–{pred['predicted_away']}<br>
                        Conf: {round(pred['confidence']*100)}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2.5rem;">{at.get('flag','🏳️')}</div>
                    <div class="metric-value">{round(pred['away_win_prob']*100)}%</div>
                    <div class="metric-label">{at.get('name','')}</div>
                    <div style="color:#888; font-size:0.8rem; margin-top:8px;">
                        ELO: {at.get('elo_rating','?')}<br>
                        Ranking: #{at.get('fifa_ranking','?')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(prob_bar_html(pred["home_win_prob"], pred["draw_prob"],
                                      pred["away_win_prob"], ht["name"], at["name"]),
                        unsafe_allow_html=True)

            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                xg_fig = go.Figure(go.Bar(
                    x=["Goles Esperados"],
                    y=[pred["expected_home"]],
                    name=ht["name"],
                    marker_color="#1a8a4a",
                ))
                xg_fig.add_trace(go.Bar(
                    x=["Goles Esperados"],
                    y=[pred["expected_away"]],
                    name=at["name"],
                    marker_color="#8a1a1a",
                ))
                xg_fig.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    title="Goles Esperados (xG)",
                    height=280,
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(xg_fig, use_container_width=True)

            with col_stat2:
                elo_fig = go.Figure(go.Bar(
                    x=[ht["name"], at["name"]],
                    y=[ht.get("elo_rating", 0), at.get("elo_rating", 0)],
                    marker_color=["#1a8a4a", "#8a1a1a"],
                    text=[ht.get("elo_rating", 0), at.get("elo_rating", 0)],
                    textposition="outside",
                ))
                elo_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    title="Comparación de Puntuación ELO",
                    yaxis=dict(showgrid=True, gridcolor="#2a2a4e"),
                    height=280,
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(elo_fig, use_container_width=True)

            st.info(f"⚙️ Modelo: {pred['method']}")
        else:
            st.warning("Por favor selecciona dos equipos diferentes.")

    # ── Tab 2: ELO Rankings ─────────────────────────────────────
    with tab2:
        st.subheader("Puntuaciones ELO — Los 48 Equipos")
        conf_filter = st.multiselect("Filtrar por Confederación",
                                     ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"],
                                     default=["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"])

        elo_rows = []
        for tid, t in fm.teams.items():
            if t["confederation"] not in conf_filter:
                continue
            elo_rows.append({
                "Equipo": f"{t['flag']} {t['name']}",
                "Confederación": t["confederation"],
                "ELO": t.get("elo_rating", 0),
                "Ranking FIFA": t.get("fifa_ranking", 99),
                "Títulos Mundiales": t.get("world_cup_titles", 0),
            })
        elo_df = pd.DataFrame(elo_rows).sort_values("ELO", ascending=False).reset_index(drop=True)
        elo_df.index += 1

        conf_color_map = {
            "UEFA": "#1f77b4", "CONMEBOL": "#2ca02c",
            "CONCACAF": "#ff7f0e", "CAF": "#d62728",
            "AFC": "#9467bd", "OFC": "#8c564b",
        }
        fig = px.bar(elo_df, x="ELO", y="Equipo", orientation="h",
                     color="Confederación",
                     color_discrete_map=conf_color_map,
                     height=max(400, len(elo_df) * 22))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(autorange="reversed", showgrid=False),
            xaxis=dict(showgrid=True, gridcolor="#2a2a4e"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=20, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Historical ────────────────────────────────────────
    with tab3:
        st.subheader("Enfrentamientos Directos: Campeones del Mundo")
        st.caption("Historial completo entre selecciones ganadoras de la Copa Mundial FIFA (partidos competitivos y amistosos).")

        # Hard-coded H2H data for WC-winning nations
        # (Team A, Team B, Played, A Wins, Draws, B Wins)
        wc_winner_ids = {
            "BRA": ("🇧🇷", "Brasil"),
            "GER": ("🇩🇪", "Alemania"),
            "ITA": ("🇮🇹", "Italia"),
            "ARG": ("🇦🇷", "Argentina"),
            "FRA": ("🇫🇷", "Francia"),
            "URU": ("🇺🇾", "Uruguay"),
            "ESP": ("🇪🇸", "España"),
            "ENG": ("🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Inglaterra"),
        }

        h2h_data = [
            ("BRA", "ARG", 109, 44, 25, 40),
            ("BRA", "URU",  77, 37, 19, 21),
            ("ARG", "URU",  56, 22, 15, 19),
            ("ITA", "ESP",  38, 13, 13, 12),
            ("GER", "ENG",  37, 15, 12, 10),
            ("ITA", "FRA",  35, 14, 11, 10),
            ("GER", "ITA",  35, 12, 11, 12),
            ("FRA", "ENG",  32, 13, 10,  9),
            ("FRA", "ESP",  33, 13, 11,  9),
            ("GER", "FRA",  28, 13,  7,  8),
            ("ITA", "ENG",  27, 11,  9,  7),
            ("BRA", "ITA",  27, 10,  9,  8),
            ("BRA", "GER",  23, 12,  4,  7),
            ("BRA", "ESP",  23, 11,  7,  5),
            ("GER", "ESP",  24, 10,  7,  7),
            ("BRA", "ENG",  25, 12,  7,  6),
            ("ARG", "GER",  20,  9,  4,  7),
            ("ARG", "ESP",  18,  8,  5,  5),
            ("ITA", "URU",  15,  7,  5,  3),
            ("BRA", "FRA",  15,  7,  5,  3),
            ("ARG", "ITA",  17,  7,  5,  5),
            ("ARG", "FRA",  12,  6,  3,  3),
            ("URU", "ESP",  12,  4,  4,  4),
            ("URU", "ENG",  10,  4,  3,  3),
            ("FRA", "URU",  10,  5,  3,  2),
            ("GER", "URU",  10,  5,  3,  2),
            ("ARG", "ENG",  16,  7,  4,  5),
            ("ESP", "ENG",  28, 12,  9,  7),
        ]

        h2h_rows = []
        for a, b, played, aw, d, bw in h2h_data:
            fa, na = wc_winner_ids[a]
            fb, nb = wc_winner_ids[b]
            h2h_rows.append({
                "Rivalidad": f"{fa} {na} vs {fb} {nb}",
                "Jugados": played,
                f"{na} V": aw,
                "Empates": d,
                f"{nb} V": bw,
                "_sort": played,
            })

        h2h_df = pd.DataFrame(h2h_rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])

        # Horizontal bar: top rivalries by matches played
        top_rivals = h2h_df.head(15).copy()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=top_rivals["Rivalidad"],
            x=top_rivals["Jugados"],
            orientation="h",
            marker_color="#FFD700",
            text=top_rivals["Jugados"],
            textposition="inside",
            name="Partidos Jugados",
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title="Rivalidades Más Jugadas — Campeones del Mundo",
            yaxis=dict(autorange="reversed", showgrid=False),
            xaxis=dict(showgrid=True, gridcolor="#2a2a4e", title="Total de Partidos"),
            margin=dict(l=10, r=20, t=40, b=10),
            height=440,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(h2h_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR + ROUTING
# ═══════════════════════════════════════════════════════════════
def main():
    fm, engine = load_app()

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:12px 0 8px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="140" height="140">
              <!-- Shield background -->
              <path d="M100 8 L185 40 L185 120 Q185 170 100 195 Q15 170 15 120 L15 40 Z"
                    fill="#1a1a1a" stroke="#FFD700" stroke-width="4"/>
              <!-- Yellow vertical stripes -->
              <clipPath id="shield-clip">
                <path d="M100 8 L185 40 L185 120 Q185 170 100 195 Q15 170 15 120 L15 40 Z"/>
              </clipPath>
              <g clip-path="url(#shield-clip)">
                <rect x="37" y="0" width="14" height="200" fill="#FFD700" opacity="0.35"/>
                <rect x="65" y="0" width="14" height="200" fill="#FFD700" opacity="0.35"/>
                <rect x="121" y="0" width="14" height="200" fill="#FFD700" opacity="0.35"/>
                <rect x="149" y="0" width="14" height="200" fill="#FFD700" opacity="0.35"/>
              </g>
              <!-- Soccer ball -->
              <circle cx="100" cy="52" r="26" fill="white" stroke="#1a1a1a" stroke-width="2"/>
              <polygon points="100,34 112,43 108,57 92,57 88,43" fill="#1a1a1a"/>
              <polygon points="100,70 112,61 123,68 119,82 81,82 77,68 88,61" fill="none"/>
              <circle cx="100" cy="52" r="26" fill="none" stroke="#1a1a1a" stroke-width="1.5"/>
              <!-- ConmeGolf text -->
              <text x="100" y="98" text-anchor="middle" font-family="Arial Black, sans-serif"
                    font-size="14" font-weight="900" fill="#FFD700" letter-spacing="1">ConmeGolf</text>
              <!-- es Mundial box -->
              <rect x="22" y="105" width="156" height="40" rx="4" fill="#FFD700"/>
              <text x="100" y="133" text-anchor="middle" font-family="Arial Black, sans-serif"
                    font-size="22" font-weight="900" fill="#1a1a1a" letter-spacing="1">es Mundial</text>
              <!-- Year -->
              <text x="100" y="170" text-anchor="middle" font-family="Arial, sans-serif"
                    font-size="13" font-weight="700" fill="#FFD700" letter-spacing="2">2026</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio("", [
            "🏠 Inicio",
            "📅 Partidos",
            "👥 Grupos",
            "🏆 Eliminatorias",
            "🎯 Predicciones",
            "📊 Análisis",
        ], label_visibility="collapsed")

        st.markdown("---")
        # Quick stats
        total = fm.total_matches
        done  = fm.completed_matches
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#aaa; padding:8px 0;">
            <div>✅ {done} / {total} partidos jugados</div>
            <div>📝 {len(st.session_state.get('user_predictions', {}))} predicciones guardadas</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.7rem; color:#555; text-align:center;">
            🔬 Pronóstico impulsado por modelo ELO<br>
            <span style="color:#FFD700;">Conecta tu propio modelo → core/forecast.py</span>
        </div>
        """, unsafe_allow_html=True)

    if page == "🏠 Inicio":
        page_home(fm, engine)
    elif page == "📅 Partidos":
        page_fixtures(fm, engine)
    elif page == "👥 Grupos":
        page_groups(fm, engine)
    elif page == "🏆 Eliminatorias":
        page_bracket(fm, engine)
    elif page == "🎯 Predicciones":
        page_predictions(fm, engine)
    elif page == "📊 Análisis":
        page_analytics(fm, engine)


if __name__ == "__main__":
    main()
