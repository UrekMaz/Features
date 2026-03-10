"""
streamlit_app.py
════════════════════════════════════════════════════════════════════════
Readability Classification App
Run: streamlit run streamlit_app.py
Requires api.py running on http://localhost:8000
════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from urllib.parse import urlparse

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
SARVAM_API_URL = os.getenv("SARVAM_API_URL", "https://api.sarvam.ai/v1/chat/completions")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-m")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


API_CONNECT_TIMEOUT_SEC = _env_int("API_CONNECT_TIMEOUT_SEC", 10)
API_CLASSIFY_TIMEOUT_SEC = _env_int("API_CLASSIFY_TIMEOUT_SEC", 120)
API_NUDGE_TIMEOUT_SEC = _env_int("API_NUDGE_TIMEOUT_SEC", 120)
API_GRADE_MEANS_TIMEOUT_SEC = _env_int("API_GRADE_MEANS_TIMEOUT_SEC", 60)


def _candidate_api_urls(base_url: str):
    urls = [base_url.rstrip("/")]
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host == "localhost":
        urls.append(base_url.replace("localhost", "127.0.0.1").rstrip("/"))
    elif host == "127.0.0.1":
        urls.append(base_url.replace("127.0.0.1", "localhost").rstrip("/"))
    seen = set()
    ordered = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


API_URL_CANDIDATES = _candidate_api_urls(API_URL)


@st.cache_resource(ttl=600)
def _pick_preferred_api_url():
    for base_url in API_URL_CANDIDATES:
        try:
            r = requests.get(
                f"{base_url}/health",
                timeout=(2, 5),
            )
            if r.ok:
                return base_url
        except requests.RequestException:
            continue
    return API_URL_CANDIDATES[0]


def _ordered_api_urls():
    preferred = _pick_preferred_api_url()
    ordered = [preferred]
    for url in API_URL_CANDIDATES:
        if url != preferred:
            ordered.append(url)
    return ordered

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReadabilityLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --ink:       #0f0e0d;
    --paper:     #f5f0e8;
    --cream:     #ede8dc;
    --red:       #c0392b;
    --blue:      #1a3a5c;
    --gold:      #c8930a;
    --teal:      #1a6b6b;
    --grade-2:   #2ecc71;
    --grade-3:   #27ae60;
    --grade-4:   #f1c40f;
    --grade-5:   #e67e22;
    --grade-6:   #e74c3c;
    --grade-7:   #9b59b6;
    --grade-8:   #2c3e50;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--paper);
    color: var(--ink);
}

.stApp { background-color: var(--paper); }

/* Hero header */
.hero {
    background: var(--blue);
    padding: 2.5rem 3rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 4px solid var(--gold);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 300px; height: 300px;
    border: 40px solid rgba(200,147,10,0.15);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 200px;
    width: 200px; height: 200px;
    border: 30px solid rgba(200,147,10,0.1);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #f5f0e8;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.1;
}
.hero-title span { color: var(--gold); font-style: italic; }
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: rgba(245,240,232,0.6);
    margin-top: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Section labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, var(--gold), transparent);
}

/* Grade badge */
.grade-badge {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 110px; height: 110px;
    border-radius: 50%;
    border: 4px solid var(--gold);
    background: var(--blue);
    color: #f5f0e8;
    font-family: 'DM Serif Display', serif;
}
.grade-badge .grade-num {
    font-size: 3rem;
    line-height: 1;
    color: var(--gold);
}
.grade-badge .grade-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.7;
}

/* Stat card */
.stat-card {
    background: white;
    border: 1px solid rgba(0,0,0,0.08);
    border-left: 4px solid var(--gold);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.stat-card .stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: var(--blue);
    line-height: 1;
}
.stat-card .stat-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    margin-top: 0.2rem;
}

/* Nudge card */
.nudge-card {
    background: white;
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.nudge-arrow {
    font-size: 1.4rem;
    min-width: 28px;
    padding-top: 2px;
}
.nudge-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--blue);
    margin-bottom: 0.2rem;
}
.nudge-advice { font-size: 0.82rem; color: #555; }
.nudge-gap {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--gold);
    margin-top: 0.3rem;
}

/* Text input area */
.stTextArea textarea {
    background: white !important;
    border: 2px solid #ddd !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(26,58,92,0.1) !important;
}

/* Button */
.stButton > button {
    background: var(--blue) !important;
    color: #f5f0e8 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: var(--ink) !important;
    transform: translateY(-1px);
}

/* Divider */
.ruled { border: none; border-top: 2px solid var(--cream); margin: 1.5rem 0; }

/* Probability bar */
.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.prob-grade {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    width: 56px;
    color: var(--ink);
}
.prob-bar-wrap {
    flex: 1;
    height: 12px;
    background: var(--cream);
    border-radius: 2px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 2px; transition: width 0.8s ease; }
.prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    width: 38px;
    text-align: right;
    color: #888;
}

/* Watermark */
.watermark {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #bbb;
    text-align: center;
    margin-top: 3rem;
    letter-spacing: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Colour map ────────────────────────────────────────────────────────────────
GRADE_COLORS = {
    2: "#2ecc71", 3: "#27ae60", 4: "#f1c40f",
    5: "#e67e22", 6: "#e74c3c", 7: "#9b59b6", 8: "#2c3e50"
}
GRADE_LABELS = {
    2: "Grade 2", 3: "Grade 3", 4: "Grade 4",
    5: "Grade 5", 6: "Grade 6", 7: "Grade 7", 8: "Grade 8"
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _request_api(method: str, path: str, read_timeout_sec: int, **kwargs):
    last_error = ""
    for base_url in _ordered_api_urls():
        url = f"{base_url}{path}"
        try:
            response = requests.request(
                method,
                url,
                timeout=(API_CONNECT_TIMEOUT_SEC, read_timeout_sec),
                **kwargs,
            )
        except requests.RequestException as exc:
            last_error = f"{url} unreachable: {exc}"
            continue

        if response.ok:
            return response

        detail = response.text.strip()
        try:
            payload = response.json()
            detail = payload.get("detail") or detail
        except Exception:
            pass

        last_error = f"{response.status_code} {response.reason} ({url}): {detail}"
        if response.status_code < 500:
            break

    raise RuntimeError(last_error or "API request failed")


def classify(text):
    payload = {"text": (text or "").strip()}
    r = _request_api("POST", "/classify", read_timeout_sec=API_CLASSIFY_TIMEOUT_SEC, json=payload)
    return r.json()


def nudge(text, target):
    payload = {"text": (text or "").strip(), "target_grade": target}
    r = _request_api("POST", "/nudge", read_timeout_sec=API_NUDGE_TIMEOUT_SEC, json=payload)
    return r.json()


def grade_means():
    r = _request_api("GET", "/grade-means", read_timeout_sec=API_GRADE_MEANS_TIMEOUT_SEC)
    return r.json()

@st.cache_data(ttl=3600)
def get_grade_means():
    return grade_means()


def get_sarvam_api_key() -> str:
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        secret_key = str(st.secrets.get("SARVAM_API_KEY", "")).strip()
        if secret_key:
            return secret_key
    except Exception:
        pass

    return ""

def rewrite_text(text: str, current_grade: int, target_grade: int) -> str:
    """Call Sarvam AI to rewrite text at target grade level."""
    api_key = get_sarvam_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing SARVAM_API_KEY. Set it in environment or .streamlit/secrets.toml and restart Streamlit."
        )

    direction = "simpler and easier" if target_grade < current_grade else "more complex and advanced"
    prompt = f"""Rewrite the following text to be appropriate for Grade {target_grade} reading level ({direction} than Grade {current_grade}).

Rules:
- Keep the same topic, facts, and meaning
- Adjust vocabulary complexity, sentence structure, and clause complexity to match Grade {target_grade}
- For lower grades: use shorter sentences, simpler words, fewer subordinate clauses
- For higher grades: use more sophisticated vocabulary, longer sentences, more complex syntax
- Return ONLY the rewritten text, no explanation or preamble

Original text (Grade {current_grade}):
{text}

Rewritten text (Grade {target_grade}):"""

    payload = {
        "model": SARVAM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(SARVAM_API_URL, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Sarvam response did not include choices.")

    message = choices[0].get("message") or {}
    content = message.get("content", "")

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text_part = item.get("text")
                if text_part:
                    parts.append(text_part)
        content = "\n".join(parts)

    rewritten = str(content).strip()
    if not rewritten:
        raise RuntimeError("Sarvam returned empty rewritten text.")
    return rewritten


def prob_bar_html(grade_probs, predicted_grade):
    rows = ""
    for g in sorted(grade_probs.keys()):
        p    = grade_probs[g]
        col  = GRADE_COLORS.get(g, "#888")
        bold = "font-weight:700;" if g == predicted_grade else ""
        rows += f"""
        <div class="prob-row">
            <div class="prob-grade" style="{bold}">Grade {g}</div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{p*100:.1f}%;background:{col};"></div>
            </div>
            <div class="prob-pct">{p*100:.1f}%</div>
        </div>"""
    return rows


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analyzed_text" not in st.session_state:
    st.session_state.analyzed_text = ""

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Readability<span>Lens</span></div>
    <div class="hero-sub">Text Complexity Analysis · Grades 2–8 · ML-Powered</div>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">① Input Text</div>', unsafe_allow_html=True)
text_input = st.text_area(
    label="Input text",
    placeholder="Paste your text here. Aim for at least 50 words for best results.",
    height=160,
    label_visibility="collapsed"
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    run = st.button("Analyse →")
with col_info:
    if text_input:
        wc = len(text_input.split())
        st.markdown(
            f'<p style="font-family:Space Mono,monospace;font-size:0.65rem;'
            f'color:#888;padding-top:0.7rem;">{wc} words</p>',
            unsafe_allow_html=True
        )

st.markdown('<hr class="ruled">', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if run:
    if not text_input or len(text_input.split()) < 10:
        st.warning("Please enter at least 10 words.")
        st.stop()

    with st.spinner("Extracting features and classifying…"):
        try:
            result = classify(text_input)
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.stop()

    st.session_state.analysis_result = result
    st.session_state.analyzed_text = text_input
    st.session_state.rewritten_text = None
    st.session_state.rewritten_grade = None
    st.session_state.rewritten_confidence = None
    st.session_state.rewrite_source_text = None
    st.session_state.rewrite_target_grade = None
    st.session_state.nudge_result = None
    st.session_state.nudge_source_text = None
    st.session_state.nudge_target_grade = None

if st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result
    source_text = st.session_state.analyzed_text

    grade      = int(result["grade"])
    confidence = result["confidence"]
    gp         = {int(k): v for k, v in result["grade_probs"].items()}
    top_feats  = result["top_features"]
    wc         = result["word_count"]
    gc         = GRADE_COLORS.get(grade, "#555")

    # ── Section 1: Prediction ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">② Prediction</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 2, 2])

    with c1:
        st.markdown(f"""
        <div style="display:flex;justify-content:center;padding-top:0.5rem;">
          <div class="grade-badge" style="border-color:{gc};">
            <div class="grade-num" style="color:{gc};">{grade}</div>
            <div class="grade-lbl">grade</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val" style="color:{gc};">{confidence*100:.1f}%</div>
            <div class="stat-lbl">Model Confidence</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{wc}</div>
            <div class="stat-lbl">Word Count</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:0.6rem;'
            'letter-spacing:2px;color:#aaa;text-transform:uppercase;'
            'margin-bottom:0.5rem;">Grade Distribution</div>',
            unsafe_allow_html=True
        )
        st.markdown(prob_bar_html(gp, grade), unsafe_allow_html=True)

    st.markdown('<hr class="ruled">', unsafe_allow_html=True)

    # ── Section 2: Feature Contributions ─────────────────────────────────────
    st.markdown('<div class="section-label">③ Top Feature Contributions</div>', unsafe_allow_html=True)

    df_feat = pd.DataFrame(top_feats)
    df_feat["abs"] = df_feat["contribution"].abs()
    df_feat = df_feat.sort_values("abs", ascending=True).tail(10)
    df_feat["color"] = df_feat["contribution"].apply(
        lambda x: "#1a3a5c" if x > 0 else "#c0392b"
    )

    fig_feat = go.Figure(go.Bar(
        x=df_feat["contribution"],
        y=df_feat["feature"],
        orientation="h",
        marker=dict(
            color=df_feat["color"].tolist(),
            line=dict(width=0)
        ),
        text=df_feat["contribution"].apply(lambda x: f"{x:+.2f}"),
        textposition="outside",
        textfont=dict(family="Space Mono", size=9, color="#555"),
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>"
    ))
    fig_feat.update_layout(
        plot_bgcolor="#f5f0e8",
        paper_bgcolor="#f5f0e8",
        margin=dict(l=10, r=60, t=10, b=10),
        height=320,
        xaxis=dict(
            showgrid=True,
            gridcolor="#ede8dc",
            zeroline=True,
            zerolinecolor="#1a3a5c",
            zerolinewidth=2,
            tickfont=dict(family="Space Mono", size=8),
        ),
        yaxis=dict(
            tickfont=dict(family="Space Mono", size=8),
            showgrid=False,
        ),
        font=dict(family="Inter"),
        showlegend=False,
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    # legend
    st.markdown("""
    <div style="display:flex;gap:1.5rem;font-family:Space Mono,monospace;
    font-size:0.6rem;letter-spacing:1px;color:#888;margin-top:-0.5rem;
    margin-bottom:1rem;">
        <span><span style="color:#1a3a5c;font-weight:700;">■</span> Pushes toward predicted grade</span>
        <span><span style="color:#c0392b;font-weight:700;">■</span> Pulls away from predicted grade</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="ruled">', unsafe_allow_html=True)

    # ── Section 3: Radar Chart ────────────────────────────────────────────────
    st.markdown('<div class="section-label">④ Feature Profile vs Grade Averages</div>', unsafe_allow_html=True)

    try:
        means_data = get_grade_means()
        means      = means_data["means"]

        # Pick 8 interpretable radar features
        radar_feats = [
            "at_AAKuW_C", "SquaNoV_S", "ra_SuVeT_C",
            "coref.avg_inference_distance", "at_ContW_C",
            "WRDPOLc", "WRDHYPnv", "const.np_density"
        ]
        radar_labels = [
            "Word Difficulty", "Noun Variety", "Clause Complexity",
            "Reference Distance", "Content Density",
            "Word Ambiguity", "Abstraction", "NP Density"
        ]

        all_feats    = result["all_features"]
        text_vals    = [all_feats.get(f, 0.0) for f in radar_feats]
        grade_m_vals = {
            g: [float(means.get(str(g), {}).get(f, 0.0)) for f in radar_feats]
            for g in [2, 5, 8]
        }

        # Normalise 0–1 across all grades for radar
        all_vals_matrix = np.array(
            list(grade_m_vals.values()) + [text_vals]
        )
        col_min = all_vals_matrix.min(axis=0)
        col_max = all_vals_matrix.max(axis=0)
        col_rng = np.where(col_max - col_min == 0, 1, col_max - col_min)

        def norm(vals):
            return ((np.array(vals) - col_min) / col_rng).tolist()

        fig_radar = go.Figure()

        # Grade reference lines (faint)
        grade_styles = {
            2: dict(color="#2ecc71", dash="dot",  width=1.5, name="Grade 2 avg"),
            5: dict(color="#e67e22", dash="dash", width=1.5, name="Grade 5 avg"),
            8: dict(color="#2c3e50", dash="dot",  width=1.5, name="Grade 8 avg"),
        }
        for g, style in grade_styles.items():
            nv = norm(grade_m_vals[g]) + [norm(grade_m_vals[g])[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=nv, theta=radar_labels + [radar_labels[0]],
                mode="lines",
                line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
                opacity=0.45,
                name=style["name"],
                fill="none",
            ))

        # Text trace (bold)
        nt = norm(text_vals) + [norm(text_vals)[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=nt, theta=radar_labels + [radar_labels[0]],
            mode="lines+markers",
            line=dict(color=gc, width=3),
            marker=dict(size=7, color=gc),
            fill="toself",
            fillcolor=gc,
            opacity=0.15,
            name=f"Your text (Grade {grade})",
        ))
        # Second trace for fill opacity layering
        fig_radar.add_trace(go.Scatterpolar(
            r=nt, theta=radar_labels + [radar_labels[0]],
            mode="lines",
            line=dict(color=gc, width=3),
            fill="toself",
            fillcolor=gc,
            opacity=0.1,
            showlegend=False,
            name="",
        ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="#ede8dc",
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=7, family="Space Mono"),
                    gridcolor="rgba(0,0,0,0.08)",
                    linecolor="rgba(0,0,0,0.1)",
                    tickvals=[0.25, 0.5, 0.75],
                ),
                angularaxis=dict(
                    tickfont=dict(size=9, family="Space Mono"),
                    gridcolor="rgba(0,0,0,0.08)",
                    linecolor="rgba(0,0,0,0.1)",
                ),
            ),
            plot_bgcolor="#f5f0e8",
            paper_bgcolor="#f5f0e8",
            legend=dict(
                font=dict(family="Space Mono", size=9),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="#ddd",
                borderwidth=1,
            ),
            margin=dict(l=40, r=40, t=30, b=30),
            height=360,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    except Exception as e:
        st.caption(f"Radar chart unavailable: {e}")

    st.markdown('<hr class="ruled">', unsafe_allow_html=True)

    # ── Section 4: Grade Nudge + Live Rewrite ────────────────────────────────
    st.markdown('<div class="section-label">⑤ Grade Nudge & Rewrite</div>', unsafe_allow_html=True)

    valid_grades = sorted(gp.keys())
    other_grades = [g for g in valid_grades if g != grade]

    # Session state for caching rewrite + nudge results
    if "rewritten_text" not in st.session_state:
        st.session_state.rewritten_text = None
    if "rewritten_grade" not in st.session_state:
        st.session_state.rewritten_grade = None
    if "rewritten_confidence" not in st.session_state:
        st.session_state.rewritten_confidence = None
    if "rewrite_source_text" not in st.session_state:
        st.session_state.rewrite_source_text = None
    if "rewrite_target_grade" not in st.session_state:
        st.session_state.rewrite_target_grade = None
    if "nudge_result" not in st.session_state:
        st.session_state.nudge_result = None
    if "nudge_source_text" not in st.session_state:
        st.session_state.nudge_source_text = None
    if "nudge_target_grade" not in st.session_state:
        st.session_state.nudge_target_grade = None

    sel_col, dir_col = st.columns([1, 3])
    with sel_col:
        target_grade = st.selectbox(
            "Rewrite for grade:",
            options=other_grades,
            format_func=lambda g: f"Grade {g}",
            key="target_grade_select",
        )
        target_grade = int(target_grade)
    with dir_col:
        direction_label = "▲ harder" if target_grade > grade else "▼ easier"
        gc_target = GRADE_COLORS.get(target_grade, "#555")
        st.markdown(
            f'<p style="font-family:Space Mono,monospace;font-size:0.7rem;color:#888;padding-top:2rem;">'
            f'Grade {grade} → Grade {target_grade} '
            f'<span style="color:{gc_target};">{direction_label}</span></p>',
            unsafe_allow_html=True
        )

    rewrite_col, nudge_col = st.columns([1, 1])
    with rewrite_col:
        run_rewrite = st.button("Rewrite now", key="run_rewrite_btn")
    with nudge_col:
        run_nudge = st.button("Generate nudges", key="run_nudge_btn")

    rewrite_cache_miss = (
        st.session_state.rewrite_source_text != source_text
        or st.session_state.rewrite_target_grade != target_grade
    )
    if rewrite_cache_miss:
        st.session_state.rewritten_text = None
        st.session_state.rewritten_grade = None
        st.session_state.rewritten_confidence = None

    if run_rewrite:
        with st.spinner(f"Rewriting at Grade {target_grade}…"):
            try:
                rewritten_text = rewrite_text(source_text, grade, target_grade)
                st.session_state.rewritten_text = rewritten_text
                st.session_state.rewrite_source_text = source_text
                st.session_state.rewrite_target_grade = target_grade
            except Exception as e:
                st.error(f"Rewrite error: {e}")
                st.caption(
                    "PowerShell: $env:SARVAM_API_KEY='your_key_here'  |  or add SARVAM_API_KEY to .streamlit/secrets.toml"
                )

        if st.session_state.rewritten_text:
            with st.spinner("Classifying rewritten text…"):
                try:
                    rr = classify(st.session_state.rewritten_text)
                    st.session_state.rewritten_grade = rr["grade"]
                    st.session_state.rewritten_confidence = rr["confidence"]
                except Exception as e:
                    st.warning(f"Could not classify rewritten text: {e}")

    # ── Side-by-side original vs rewritten ───────────────────────────────────
    if st.session_state.rewritten_text:
        orig_col, rew_col = st.columns(2)

        with orig_col:
            st.markdown(f"""
            <div style="background:white;border:1px solid rgba(0,0,0,0.08);
            border-left:4px solid {GRADE_COLORS.get(grade,'#555')};
            border-radius:4px;padding:1.2rem 1.5rem;">
                <div style="font-family:Space Mono,monospace;font-size:0.55rem;
                letter-spacing:2px;color:{GRADE_COLORS.get(grade,'#555')};
                text-transform:uppercase;margin-bottom:0.8rem;">
                    Original — Grade {grade}
                </div>
                <div style="font-family:Inter,sans-serif;font-size:0.85rem;
                line-height:1.7;color:#333;">
                    {source_text.replace(chr(10), "<br>")}
                </div>
            </div>""", unsafe_allow_html=True)

        with rew_col:
            rg  = st.session_state.rewritten_grade
            rc  = st.session_state.rewritten_confidence
            rgc = GRADE_COLORS.get(rg, gc_target) if rg else gc_target
            badge = (
                f'<span style="font-family:DM Serif Display,serif;font-size:1.1rem;color:{rgc};">'
                f'Grade {rg}</span> '
                f'<span style="font-family:Space Mono,monospace;font-size:0.6rem;color:#aaa;">{rc*100:.0f}% conf</span>'
            ) if rg else ""
            st.markdown(f"""
            <div style="background:white;border:1px solid rgba(0,0,0,0.08);
            border-left:4px solid {gc_target};
            border-radius:4px;padding:1.2rem 1.5rem;">
                <div style="display:flex;justify-content:space-between;align-items:baseline;
                margin-bottom:0.8rem;">
                    <div style="font-family:Space Mono,monospace;font-size:0.55rem;
                    letter-spacing:2px;color:{gc_target};text-transform:uppercase;">
                        Rewritten — Grade {target_grade}
                    </div>
                    {badge}
                </div>
                <div style="font-family:Inter,sans-serif;font-size:0.85rem;
                line-height:1.7;color:#333;">
                    {st.session_state.rewritten_text.replace(chr(10), "<br>")}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    # ── Nudge advice ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:2px;'
        'color:#aaa;text-transform:uppercase;margin-bottom:0.6rem;">Feature-Level Advice</div>',
        unsafe_allow_html=True
    )
    nudge_cache_miss = (
        st.session_state.nudge_source_text != source_text
        or st.session_state.nudge_target_grade != target_grade
    )
    if nudge_cache_miss:
        st.session_state.nudge_result = None

    if run_nudge:
        with st.spinner("Generating nudge suggestions…"):
            try:
                st.session_state.nudge_result = nudge(source_text, target_grade)
                st.session_state.nudge_source_text = source_text
                st.session_state.nudge_target_grade = target_grade
            except Exception as e:
                st.error(f"Nudge API error: {e}")

    if st.session_state.nudge_result is None:
        st.caption("Click 'Generate nudges' to fetch feature-level advice.")
    else:
        nudges = st.session_state.nudge_result.get("nudges", [])
        if not nudges:
            st.info("No significant feature changes needed — the text is already close to the target grade.")
        else:
            for n in nudges:
                arrow       = "↑" if n["direction"] == "increase" else "↓"
                arrow_color = "#1a3a5c" if n["direction"] == "increase" else "#c0392b"
                gap_str     = f"{n['current_value']:.3f} → {n['target_value']:.3f}"
                st.markdown(f"""
                <div class="nudge-card">
                    <div class="nudge-arrow" style="color:{arrow_color};">{arrow}</div>
                    <div>
                        <div class="nudge-title">{n['title']}</div>
                        <div class="nudge-advice">{n['advice']}</div>
                        <div class="nudge-gap">
                            <span style="font-family:Space Mono,monospace;font-size:0.6rem;
                            color:#aaa;">{n['feature']}</span>
                            &nbsp;·&nbsp; {gap_str}
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#bbb;">
        <div style="font-size:3rem;margin-bottom:1rem;">🔬</div>
        <div style="font-family:Space Mono,monospace;font-size:0.7rem;
        letter-spacing:2px;text-transform:uppercase;">
            Paste text above and click Analyse
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="watermark">READABILITYLENS · ML READABILITY CLASSIFIER · GRADES 2–8</div>',
    unsafe_allow_html=True
)