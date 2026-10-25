"""
Mohalla Mic 🎙️🏏
Voice-first gully-cricket scorekeeper powered by Sarvam AI.
"""

import os
import io
import json
import base64
import math
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ─── Constants ──────────────────────────────────────────────────────────────────

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"

VALID_EXTRAS = {"none", "wide", "no_ball", "bye", "leg_bye"}
VALID_REGIONS = {
    "covers", "point", "third_man", "fine_leg", "square_leg",
    "mid_wicket", "long_on", "long_off", "straight", "none",
}
WAGON_XY: dict[str, tuple[float, float]] = {
    "straight":   ( 0.0,  1.0),
    "long_on":    (-0.55, 0.85),
    "long_off":   ( 0.55, 0.85),
    "mid_wicket": (-0.80, 0.45),
    "covers":     ( 0.80, 0.45),
    "square_leg": (-1.0,  0.0),
    "point":      ( 1.0,  0.0),
    "fine_leg":   (-0.55,-0.85),
    "third_man":  ( 0.55,-0.85),
}
DEMO_CALLS: list[dict] = [
    {
        "text": "Arjun ne covers ke upar four maara",
        "event": {
            "batsman": "Arjun", "runs_off_bat": 4, "extra_runs": 0,
            "extra_type": "none", "is_wicket": False, "dismissal": "",
            "wagon_region": "covers",
            "commentary": "Arjun ka cover drive! Boundary line ke bahar, classic shot! 💥",
        },
    },
    {
        "text": "Wide ball, keeper ke paas se nikal gayi",
        "event": {
            "batsman": "", "runs_off_bat": 0, "extra_runs": 1,
            "extra_type": "wide", "is_wicket": False, "dismissal": "",
            "wagon_region": "none",
            "commentary": "Wide! Keeper ke haathon se nikal gayi, ek run aur batting side ko! 😬",
        },
    },
    {
        "text": "Sana caught at long on, kya pressure catch tha!",
        "event": {
            "batsman": "Sana", "runs_off_bat": 0, "extra_runs": 0,
            "extra_type": "none", "is_wicket": True, "dismissal": "caught at long on",
            "wagon_region": "long_on",
            "commentary": "Sana OUT! Long on pe pressure catch, crowd paagal ho gaya! 🔥",
        },
    },
]

# ─── Helpers ────────────────────────────────────────────────────────────────────

def get_api_key() -> str | None:
    """Read key from env or Streamlit secrets. Never hard-code."""
    key = os.getenv("SARVAM_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("SARVAM_API_KEY", "")
        except Exception:
            key = ""
    return key if key else None


def overs_display(legal_balls: int) -> str:
    return f"{legal_balls // 6}.{legal_balls % 6}"


def run_rate(runs: int, legal_balls: int) -> str:
    if legal_balls == 0:
        return "0.00"
    return f"{runs / (legal_balls / 6):.2f}"


def ball_pill_label(ev: dict) -> str:
    """Tiny label for the current-over display."""
    if ev["is_wicket"]:
        return "W"
    if ev["extra_type"] == "wide":
        extra = ev.get("extra_runs", 1)
        return f"Wd" if extra <= 1 else f"Wd+{extra-1}"
    if ev["extra_type"] == "no_ball":
        bat = ev.get("runs_off_bat", 0)
        return f"Nb" if bat == 0 else f"Nb+{bat}"
    return str(ev["runs_off_bat"])


# ─── State init ─────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "runs": 0,
        "wickets": 0,
        "legal_balls": 0,
        "events": [],
        "current_over_pills": [],
        "striker": "Arjun",
        "non_striker": "Sana",
        "bowler": "Rohan",
        "batters": {
            "Arjun": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0},
            "Sana":  {"runs": 0, "balls": 0, "fours": 0, "sixes": 0},
        },
        "bowlers": {
            "Rohan": {"legal_balls": 0, "runs": 0, "wickets": 0},
        },
        "local_rules": "One bounce over the boundary is out. Two runs if the ball goes into the neighbour's garden.",
        "last_transcript": "",
        "match_memory": "",
        "live_mode": True,
        "demo_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Sarvam API calls ──────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """Call Saaras v4 STT and return transcript text."""
    headers = {"api-subscription-key": api_key}
    files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
    data = {
        "model": "saaras:v4",
        "language_code": "hi-IN",
        "with_timestamps": "false",
    }
    resp = requests.post(SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json().get("transcript", "")


def extract_ball_data(transcript: str, match_state: dict, api_key: str) -> dict:
    """Send transcript + state to Sarvam-105B and get structured JSON event."""
    system_prompt = (
        "You are a precise local-cricket scorer. Interpret the voice call using the supplied local rules. "
        "Return only valid JSON matching this schema: "
        '{"batsman":"string","runs_off_bat":int,"extra_runs":int,'
        '"extra_type":"none|wide|no_ball|bye|leg_bye","is_wicket":bool,'
        '"dismissal":"string","wagon_region":"covers|point|third_man|fine_leg|square_leg|mid_wicket|long_on|long_off|straight|none",'
        '"commentary":"one short dramatic Hinglish sentence"}. '
        "Wides and no-balls are not legal deliveries. Do not invent an event if the spoken call is ambiguous. "
        "Keep commentary energetic, short, and Hinglish."
    )
    user_msg = json.dumps({
        "transcript": transcript,
        "striker": match_state["striker"],
        "non_striker": match_state["non_striker"],
        "bowler": match_state["bowler"],
        "score": f"{match_state['runs']}/{match_state['wickets']}",
        "overs": overs_display(match_state["legal_balls"]),
        "local_rules": match_state["local_rules"],
    }, ensure_ascii=False)
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sarvam-105b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(SARVAM_CHAT_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    # Strip accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    return json.loads(raw)


def generate_match_memory(api_key: str | None) -> str:
    """Ask Sarvam-105B to create a WhatsApp-ready match recap."""
    s = st.session_state
    standout = ""
    if s["events"]:
        last = s["events"][-1]
        standout = last.get("commentary", "")
    top_batter = max(s["batters"], key=lambda b: s["batters"][b]["runs"])
    prompt = (
        f"Score: {s['runs']}/{s['wickets']} in {overs_display(s['legal_balls'])} overs. "
        f"Top batter: {top_batter} ({s['batters'][top_batter]['runs']} runs). "
        f"Standout moment: {standout}. "
        "Write a warm, dramatic Hinglish WhatsApp message of 45-50 words for Mohalla Match Memory. "
        "Start with 🏏 Mohalla Match Memory:. Make it nostalgic and exciting."
    )
    if api_key:
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": "sarvam-105b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 120,
        }
        try:
            resp = requests.post(SARVAM_CHAT_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"⚠️ API error: {e}"
    # Demo fallback
    return (
        f"🏏 Mohalla Match Memory: Gully XI are {s['runs']}/{s['wickets']} after "
        f"{overs_display(s['legal_balls'])} overs. {top_batter} ka kamaal aur "
        f"har ball pe drama! Yeh match yaad rahega. Ab asli chase shuru hai! 🔥"
    )


# ─── Normalise & validate event ────────────────────────────────────────────────

def normalise_event(raw: dict) -> dict:
    ev = dict(raw)
    ev["runs_off_bat"] = max(0, min(6, int(ev.get("runs_off_bat", 0))))
    ev["extra_runs"] = max(0, int(ev.get("extra_runs", 0)))
    ev["extra_type"] = ev.get("extra_type", "none")
    if ev["extra_type"] not in VALID_EXTRAS:
        ev["extra_type"] = "none"
    ev["wagon_region"] = ev.get("wagon_region", "none")
    if ev["wagon_region"] not in VALID_REGIONS:
        ev["wagon_region"] = "none"
    ev["is_wicket"] = bool(ev.get("is_wicket", False))
    ev["dismissal"] = str(ev.get("dismissal", ""))
    ev["commentary"] = str(ev.get("commentary", ""))
    ev["batsman"] = str(ev.get("batsman", ""))
    return ev


# ─── Apply event to match state ────────────────────────────────────────────────

def apply_event(ev: dict):
    s = st.session_state
    extra = ev["extra_type"]
    is_legal = extra not in ("wide", "no_ball")

    # Resolve batsman — default to striker if name doesn't match
    batsman = ev["batsman"]
    if batsman not in s["batters"]:
        batsman = s["striker"]

    # Ensure batter exists in stats
    if batsman not in s["batters"]:
        s["batters"][batsman] = {"runs": 0, "balls": 0, "fours": 0, "sixes": 0}

    # Runs
    bat_runs = ev["runs_off_bat"]
    extra_runs = ev["extra_runs"]

    # For wide/no_ball minimum 1 extra run
    if extra in ("wide", "no_ball") and extra_runs == 0:
        extra_runs = 1

    total_added = bat_runs + extra_runs
    s["runs"] += total_added

    # Batter stats (byes/leg_byes don't credit batsman)
    if extra not in ("bye", "leg_bye"):
        s["batters"][batsman]["runs"] += bat_runs
    if is_legal:
        s["batters"][batsman]["balls"] += 1
    if bat_runs == 4:
        s["batters"][batsman]["fours"] += 1
    if bat_runs == 6:
        s["batters"][batsman]["sixes"] += 1

    # Legal ball
    if is_legal:
        s["legal_balls"] += 1

    # Bowler stats
    bowler = s["bowler"]
    if bowler not in s["bowlers"]:
        s["bowlers"][bowler] = {"legal_balls": 0, "runs": 0, "wickets": 0}
    s["bowlers"][bowler]["runs"] += total_added
    if is_legal:
        s["bowlers"][bowler]["legal_balls"] += 1

    # Wicket
    if ev["is_wicket"]:
        s["wickets"] += 1
        s["bowlers"][bowler]["wickets"] += 1

    # Strike rotation: odd runs rotate
    runs_for_rotation = bat_runs if extra not in ("bye", "leg_bye") else extra_runs
    if runs_for_rotation % 2 == 1:
        s["striker"], s["non_striker"] = s["non_striker"], s["striker"]

    # Over pills
    s["current_over_pills"].append(ball_pill_label(ev))

    # Over complete? rotate strike again
    if is_legal and s["legal_balls"] % 6 == 0:
        s["striker"], s["non_striker"] = s["non_striker"], s["striker"]
        s["current_over_pills"] = []

    # Record event
    ev["over_display"] = overs_display(s["legal_balls"])
    ev["total_score"] = f"{s['runs']}/{s['wickets']}"
    s["events"].append(ev)


# ─── Wagon Wheel chart ─────────────────────────────────────────────────────────

def wagon_wheel() -> go.Figure:
    fig = go.Figure()
    # Outfield circle
    theta = [i * (360 / 100) for i in range(101)]
    cx = [1.15 * math.cos(math.radians(t)) for t in theta]
    cy = [1.15 * math.sin(math.radians(t)) for t in theta]
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="lines",
        line=dict(color="#2d6a4f", width=2),
        showlegend=False, hoverinfo="skip",
    ))
    # Pitch
    fig.add_shape(type="rect", x0=-0.04, x1=0.04, y0=-0.12, y1=0.12,
                  fillcolor="#d4a373", line=dict(color="#bc6c25", width=1))
    # Shot dots
    for ev in st.session_state["events"]:
        region = ev.get("wagon_region", "none")
        if region == "none" or region not in WAGON_XY:
            continue
        bx, by = WAGON_XY[region]
        runs = ev.get("runs_off_bat", 0)
        size = 10 + runs * 4
        color = "#e63946" if ev.get("is_wicket") else ("#fca311" if runs >= 4 else "#219ebc")
        fig.add_trace(go.Scatter(
            x=[bx], y=[by], mode="markers+text",
            marker=dict(size=size, color=color, opacity=0.85,
                        line=dict(color="#fff", width=1)),
            text=[str(runs)], textposition="top center",
            textfont=dict(size=10, color="#1a1a2e"),
            showlegend=False,
            hovertext=ev.get("commentary", ""),
            hoverinfo="text",
        ))
    fig.update_layout(
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="x"),
        plot_bgcolor="#e8f5e9",
        paper_bgcolor="#faf7f2",
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
    )
    return fig


# ─── Page config & custom CSS ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Mohalla Mic 🎙️🏏",
    page_icon="🏏",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');

/* ── Global ──────────────────────── */
html, body, [data-testid="stApp"] {
    font-family: 'Outfit', sans-serif;
    background-color: #faf7f2;
    color: #1a1a2e;
}
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; color: #1a1a2e; }

/* ── Scoreboard card ─────────────── */
.score-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
    border: 1.5px solid #d0d7de;
    border-radius: 18px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    margin-bottom: 0.6rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.score-big {
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: #d4700a;
    line-height: 1.1;
}
.score-meta {
    font-size: 1rem;
    color: #0e7490;
    margin-top: 0.2rem;
}
.over-pills {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-top: 0.7rem;
    flex-wrap: wrap;
}
.pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 30px;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #fff;
}
.pill-run   { background: #0e7490; }
.pill-four  { background: #d4700a; }
.pill-six   { background: #e63946; color: #fff; }
.pill-wkt   { background: #e63946; color: #fff; }
.pill-extra { background: #9d6b8e; }
.pill-dot   { background: #b0b8c1; color: #fff; }

/* ── Player strip ────────────────── */
.player-strip {
    background: #ffffff;
    border-radius: 14px;
    padding: 0.8rem 0.6rem;
    border: 1px solid #d0d7de;
    text-align: center;
    margin-bottom: 0.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.player-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1a2e;
}
.player-stat {
    font-size: 0.85rem;
    color: #0e7490;
}
.on-strike { border-left: 3px solid #d4700a; }

/* ── Header tag ──────────────────── */
.sarvam-tag {
    display: inline-block;
    background: linear-gradient(90deg, #0e7490, #38bdf8);
    color: #fff;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 10px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Mode badge ──────────────────── */
.mode-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-left: 6px;
}
.mode-live { background: #d1fae5; color: #065f46; }
.mode-demo { background: #fef3c7; color: #92400e; }

/* ── Feed event ──────────────────── */
.feed-event {
    background: #ffffff;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid #0e7490;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.feed-over {
    font-size: 0.75rem;
    color: #0e7490;
    font-weight: 600;
}
.feed-commentary {
    font-size: 0.95rem;
    color: #1a1a2e;
    margin-top: 0.15rem;
}
.feed-transcript {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.2rem;
    font-style: italic;
}

/* ── Buttons ─────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #0e7490, #0369a1);
    color: #fff;
    border: none;
    border-radius: 12px;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.6rem 1.6rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 4px 18px rgba(14,116,144,0.35);
}

/* ── Tabs ────────────────────────── */
button[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: #0e7490 !important;
}

/* ── Dataframes ──────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Audio input ─────────────────── */
[data-testid="stAudioInput"] label {
    color: #1a1a2e !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─── Init ───────────────────────────────────────────────────────────────────────

init_state()
api_key = get_api_key()

# ─── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; margin-bottom:0.3rem;">
    <div style="font-size:2.4rem; font-weight:800; letter-spacing:1px; color:#1a1a2e;">
        Mohalla Mic 🎙️🏏
    </div>
    <div style="font-size:1rem; color:#0e7490; font-style:italic; margin-top:0.1rem;">
        Bol ke score karo. Yaad ban ke ghar le jao.
    </div>
    <span class="sarvam-tag">SARVAM-POWERED</span>
</div>
""", unsafe_allow_html=True)

# Mode toggle
col_mode_l, col_mode_r = st.columns([3, 1])
with col_mode_r:
    st.session_state["live_mode"] = st.toggle("Live Sarvam mode", value=st.session_state["live_mode"])
mode_label = (
    '<span class="mode-badge mode-live">🟢 LIVE SARVAM</span>'
    if st.session_state["live_mode"]
    else '<span class="mode-badge mode-demo">🟠 DEMO MODE</span>'
)
st.markdown(f"<div style='text-align:center;margin-bottom:0.8rem;'>{mode_label}</div>", unsafe_allow_html=True)

# ─── Scoreboard ─────────────────────────────────────────────────────────────────

s = st.session_state
pill_html = ""
for p in s["current_over_pills"]:
    if p == "W":
        cls = "pill pill-wkt"
    elif p.startswith("Wd") or p.startswith("Nb"):
        cls = "pill pill-extra"
    elif p in ("4",):
        cls = "pill pill-four"
    elif p in ("6",):
        cls = "pill pill-six"
    elif p == "0":
        cls = "pill pill-dot"
    else:
        cls = "pill pill-run"
    pill_html += f'<span class="{cls}">{p}</span>'

st.markdown(f"""
<div class="score-card">
    <div class="score-big">{s['runs']}/{s['wickets']}</div>
    <div class="score-meta">
        Overs: {overs_display(s['legal_balls'])} &nbsp;|&nbsp;
        RR: {run_rate(s['runs'], s['legal_balls'])}
    </div>
    <div class="over-pills">{pill_html if pill_html else '<span style="color:#b0b8c1;">New over</span>'}</div>
</div>
""", unsafe_allow_html=True)

# ─── Player strip ───────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
striker_data = s["batters"].get(s["striker"], {"runs": 0, "balls": 0})
non_striker_data = s["batters"].get(s["non_striker"], {"runs": 0, "balls": 0})
bowler_data = s["bowlers"].get(s["bowler"], {"legal_balls": 0, "runs": 0, "wickets": 0})

with col1:
    st.markdown(f"""
    <div class="player-strip on-strike">
        <div class="player-name">🏏 {s['striker']}</div>
        <div class="player-stat">{striker_data['runs']} ({striker_data['balls']})</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="player-strip">
        <div class="player-name">{s['non_striker']}</div>
        <div class="player-stat">{non_striker_data['runs']} ({non_striker_data['balls']})</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="player-strip">
        <div class="player-name">🎾 {s['bowler']}</div>
        <div class="player-stat">{overs_display(bowler_data['legal_balls'])} ov · {bowler_data['runs']}r · {bowler_data['wickets']}w</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Settings expander ──────────────────────────────────────────────────────────

with st.expander("⚙️ Match settings & local rules"):
    s["local_rules"] = st.text_area(
        "Local ground rules",
        value=s["local_rules"],
        height=80,
    )
    new_striker = st.text_input("Change striker name", value="", placeholder="e.g. Priya")
    new_non_striker = st.text_input("Change non-striker name", value="", placeholder="e.g. Dhruv")
    new_bowler = st.text_input("Change bowler name", value="", placeholder="e.g. Virat")
    if st.button("Update players"):
        if new_striker.strip():
            name = new_striker.strip()
            if name not in s["batters"]:
                s["batters"][name] = {"runs": 0, "balls": 0, "fours": 0, "sixes": 0}
            s["striker"] = name
        if new_non_striker.strip():
            name = new_non_striker.strip()
            if name not in s["batters"]:
                s["batters"][name] = {"runs": 0, "balls": 0, "fours": 0, "sixes": 0}
            s["non_striker"] = name
        if new_bowler.strip():
            name = new_bowler.strip()
            if name not in s["bowlers"]:
                s["bowlers"][name] = {"legal_balls": 0, "runs": 0, "wickets": 0}
            s["bowler"] = name
        st.rerun()

# ─── Input zone ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### 🎤 Score a ball")

audio_bytes = st.audio_input("Record your call (under 30 sec)", key="mic_input")
typed_call = st.text_input(
    "Or type a demo call",
    placeholder="e.g. Arjun ne covers ke upar four maara",
)

if st.button("🏏 Record this ball", use_container_width=True):
    live = s["live_mode"]
    transcript = ""
    event_data = None
    error_msg = None

    # --- Transcription ---
    if live and audio_bytes:
        with st.spinner("🎧 Saaras is transcribing your call…"):
            if not api_key:
                error_msg = "⚠️ SARVAM_API_KEY not found. Paste it into .env and restart."
            else:
                try:
                    transcript = transcribe_audio(audio_bytes.getvalue(), api_key)
                except Exception as e:
                    error_msg = f"❌ Saaras STT failed: {e}"
    elif live and typed_call.strip():
        transcript = typed_call.strip()
    elif not live and typed_call.strip():
        # Demo mode typed input — match against known demos
        typed_lower = typed_call.strip().lower()
        matched = None
        for demo in DEMO_CALLS:
            if typed_lower in demo["text"].lower() or demo["text"].lower() in typed_lower:
                matched = demo
                break
        if matched:
            transcript = matched["text"]
            event_data = dict(matched["event"])
        else:
            # Default demo event for unrecognised input
            transcript = typed_call.strip()
            event_data = {
                "batsman": s["striker"], "runs_off_bat": 1, "extra_runs": 0,
                "extra_type": "none", "is_wicket": False, "dismissal": "",
                "wagon_region": "straight",
                "commentary": f"{s['striker']} ne ek run le liya, smart running! 🏃",
            }
    elif not live and not typed_call.strip():
        # Demo mode without text — cycle through demos
        idx = s["demo_index"] % len(DEMO_CALLS)
        demo = DEMO_CALLS[idx]
        transcript = demo["text"]
        event_data = dict(demo["event"])
        s["demo_index"] = idx + 1
    else:
        error_msg = "⚠️ Record audio or type a call first."

    if error_msg:
        st.error(error_msg)
    elif transcript:
        s["last_transcript"] = transcript

        # --- Event extraction ---
        if event_data is None:
            with st.spinner("🧠 Sarvam-105B is interpreting the ball and local rules…"):
                if not api_key:
                    error_msg = "⚠️ SARVAM_API_KEY not found. Paste it into .env and restart."
                else:
                    try:
                        match_state = {
                            "striker": s["striker"],
                            "non_striker": s["non_striker"],
                            "bowler": s["bowler"],
                            "runs": s["runs"],
                            "wickets": s["wickets"],
                            "legal_balls": s["legal_balls"],
                            "local_rules": s["local_rules"],
                        }
                        event_data = extract_ball_data(transcript, match_state, api_key)
                    except json.JSONDecodeError as e:
                        error_msg = f"❌ Sarvam-105B returned invalid JSON. Retry? Error: {e}"
                    except Exception as e:
                        error_msg = f"❌ Sarvam-105B call failed: {e}"

        if error_msg:
            st.error(error_msg)
        elif event_data:
            ev = normalise_event(event_data)
            ev["transcript"] = transcript
            apply_event(ev)
            st.success("✅ Score updated!")
            st.rerun()

# ─── Tabs ───────────────────────────────────────────────────────────────────────

tab_feed, tab_card, tab_wagon, tab_memory = st.tabs(
    ["📋 Match feed", "📊 Scorecard", "🎯 Wagon wheel", "💬 Match memory"]
)

# ── Match feed ──────────────────────────────────────────────────────────────────
with tab_feed:
    events_rev = list(reversed(s["events"]))
    if not events_rev:
        st.markdown("<div style='text-align:center;color:#6b7280;padding:2rem;'>No balls yet. Bol ke shuru karo! 🎤</div>", unsafe_allow_html=True)
    for ev in events_rev:
        pill = ball_pill_label(ev)
        st.markdown(f"""
        <div class="feed-event">
            <span class="feed-over">{ev.get('over_display','')}</span>
            &nbsp;
            <span style="background:#d4700a;color:#fff;padding:2px 8px;border-radius:8px;font-weight:700;font-size:0.85rem;">{pill}</span>
            &nbsp;
            <span style="font-size:0.85rem;color:#0e7490;">{ev.get('total_score','')}</span>
            <div class="feed-commentary">{ev.get('commentary','')}</div>
            <div class="feed-transcript">Heard: "{ev.get('transcript','')}"</div>
        </div>
        """, unsafe_allow_html=True)

# ── Scorecard ───────────────────────────────────────────────────────────────────
with tab_card:
    st.markdown("##### 🏏 Batting")
    bat_rows = []
    for name, stats in s["batters"].items():
        bat_rows.append({
            "Batter": name,
            "Runs": stats["runs"],
            "Balls": stats["balls"],
            "4s": stats["fours"],
            "6s": stats["sixes"],
        })
    if bat_rows:
        st.dataframe(pd.DataFrame(bat_rows), use_container_width=True, hide_index=True)

    st.markdown("##### 🎾 Bowling")
    bowl_rows = []
    for name, stats in s["bowlers"].items():
        bowl_rows.append({
            "Bowler": name,
            "Overs": overs_display(stats["legal_balls"]),
            "Runs": stats["runs"],
            "Wickets": stats["wickets"],
        })
    if bowl_rows:
        st.dataframe(pd.DataFrame(bowl_rows), use_container_width=True, hide_index=True)

# ── Wagon wheel ─────────────────────────────────────────────────────────────────
with tab_wagon:
    st.plotly_chart(wagon_wheel(), use_container_width=True, config={"displayModeBar": False})

# ── Match memory ────────────────────────────────────────────────────────────────
with tab_memory:
    st.markdown("##### 💬 Create your match memory")
    st.caption("A dramatic Hinglish recap you can share on WhatsApp.")
    if st.button("✨ Create match memory"):
        with st.spinner("Sarvam-105B is writing your match story…"):
            mem = generate_match_memory(api_key if s["live_mode"] else None)
            s["match_memory"] = mem
    if s["match_memory"]:
        st.text_area("Your match memory — copy & share! 📋", value=s["match_memory"], height=130)
