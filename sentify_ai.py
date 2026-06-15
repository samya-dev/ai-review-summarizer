"""
Sentify AI v2 — Strategic Customer Intelligence Suite
=====================================================
Major upgrade with three strategic pillars:

1. STRUCTURED AI OUTPUT   → Pydantic schemas guarantee exact, type-safe data
2. KEY-PHRASE EXTRACTION  → AI auto-detects trending topics as interactive tags
3. REVIEW BATCHING        → Intelligent pre-cleaning saves token spend & latency

Additional upgrades:
- Premium dark UI with glowing gradient borders & emerald/coral/indigo theming
- Full sidebar layout for setup controls (personas, languages, upload)
- Corporate-grade export report with sentiment summaries & strategic roadmaps
- Industry persona presets tailor AI analysis context per vertical

Install dependencies:
    pip install streamlit google-genai plotly pandas pydantic python-dotenv

Run:
    streamlit run sentify_ai_v2.py
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  PYDANTIC SCHEMAS  →  Structured AI Output Guarantee
# ═══════════════════════════════════════════════════════════════════════════════


class SentimentResult(BaseModel):
    """Exact, predictable sentiment breakdown from the AI."""

    positive: float = Field(description="Percentage of positive sentiment (0-100)", ge=0, le=100)
    negative: float = Field(description="Percentage of negative sentiment (0-100)", ge=0, le=100)
    neutral: float = Field(description="Percentage of neutral sentiment (0-100)", ge=0, le=100)
    dominant: Literal["positive", "negative", "neutral"] = Field(
        description="The dominant sentiment category"
    )
    confidence: float = Field(
        description="Model confidence score 0.0-1.0", ge=0.0, le=1.0
    )


class TopicKeyword(BaseModel):
    """A high-frequency keyword or trending topic extracted from reviews."""

    keyword: str = Field(description="The keyword or topic phrase (1-3 words)")
    frequency: Literal["high", "medium", "low"] = Field(description="Frequency tier")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment polarity of this topic"
    )
    review_count: int = Field(description="Approximate number of reviews mentioning this", ge=0)


class StrategicAnalysis(BaseModel):
    """Complete structured report — no more regex parsing free text."""

    compliments: list[str] = Field(description="List of what customers praise")
    complaints: list[str] = Field(description="List of pain points and friction areas")
    advice: list[str] = Field(description="Actionable strategic recommendations")
    key_phrases: list[TopicKeyword] = Field(
        description="Top trending keywords/topics across all reviews",
        max_length=6,
    )
    executive_summary: str = Field(
        description="2-3 sentence executive summary of the feedback landscape"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ENVIRONMENT & CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sentify AI v2 | Strategic Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PREMIUM CUSTOM CSS  →  Dark aesthetic with glow & gradients
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* ── Hero ── */
    .hero-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 40%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        text-align: center;
        color: #64748B;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, #10B98120, #10B98110);
        border: 1px solid #10B98140;
        color: #10B981;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 20px;
        margin-top: 0.5rem;
    }

    /* ── Divider ── */
    .styled-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 1.8rem 0;
    }

    /* ── Glow Cards ── */
    .glow-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .glow-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        border-radius: 14px 14px 0 0;
    }
    .glow-card:hover {
        border-color: #475569;
        transform: translateY(-1px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* ── Stat Cards ── */
    .stat-card-emerald::before { background: linear-gradient(90deg, #10B981, #34D399); }
    .stat-card-coral::before   { background: linear-gradient(90deg, #EF4444, #F87171); }
    .stat-card-amber::before   { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .stat-card-indigo::before  { background: linear-gradient(90deg, #6366F1, #818CF8); }

    .stat-value {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .stat-label {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-delta {
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* ── Topic Tag Cloud ── */
    .topic-cloud {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 0.5rem 0;
    }
    .topic-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        cursor: default;
        transition: all 0.2s ease;
        border: 1px solid;
    }
    .topic-tag:hover {
        transform: scale(1.05);
    }
    .topic-tag-positive {
        background: #10B98118;
        border-color: #10B98140;
        color: #34D399;
    }
    .topic-tag-positive:hover {
        background: #10B98130;
        box-shadow: 0 0 12px #10B98140;
    }
    .topic-tag-negative {
        background: #EF444418;
        border-color: #EF444440;
        color: #F87171;
    }
    .topic-tag-negative:hover {
        background: #EF444430;
        box-shadow: 0 0 12px #EF444440;
    }
    .topic-tag-neutral {
        background: #F59E0B18;
        border-color: #F59E0B40;
        color: #FBBF24;
    }
    .topic-tag-neutral:hover {
        background: #F59E0B30;
        box-shadow: 0 0 12px #F59E0B40;
    }
    .topic-freq {
        font-size: 0.65rem;
        padding: 2px 7px;
        border-radius: 10px;
        font-weight: 700;
    }
    .topic-freq-high { background: #EF444430; color: #F87171; }
    .topic-freq-med  { background: #F59E0B30; color: #FBBF24; }
    .topic-freq-low  { background: #38BDF830; color: #7DD3FC; }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-header span {
        font-size: 1.3rem;
    }

    /* ── Insight Cards ── */
    .insight-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid;
    }
    .insight-card-compliment { border-left-color: #10B981; }
    .insight-card-complaint  { border-left-color: #EF4444; }
    .insight-card-advice     { border-left-color: #6366F1; }

    .insight-text {
        color: #CBD5E1;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .insight-bullet {
        color: #64748B;
        font-weight: 700;
        margin-right: 6px;
    }

    /* ── RTL block ── */
    .rtl-block {
        direction: rtl;
        text-align: right;
        font-size: 16px;
        line-height: 1.8;
        padding: 1.2rem 1.5rem;
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        color: #E2E8F0;
    }

    /* ── Executive Summary ── */
    .exec-summary {
        background: linear-gradient(135deg, #1E1B4B20, #312E8120);
        border: 1px solid #6366F130;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #C7D2FE;
    }

    /* ── Tab overrides ── */
    [data-testid="stTab"] button {
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* ── Input labels ── */
    label { color: #CBD5E1 !important; }

    /* ── Export Section ── */
    .export-section {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.5rem;
    }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.5s ease forwards;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  API KEY GUARD
# ═══════════════════════════════════════════════════════════════════════════════

if not api_key:
    st.error(
        "🔒 **Gemini API key not found.**\n\n"
        "Create a `.env` file in your project root:\n```\nGEMINI_API_KEY=your_key_here\n```",
        icon="🔒",
    )
    st.stop()

client = genai.Client(api_key=api_key)

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SIDEBAR  →  All Setup Controls (Personas, Languages, Upload)
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Brand ──
    st.markdown(
        "<div style='text-align:center; margin-bottom:1rem;'>"
        "<div style='font-size:1.6rem; font-weight:800; background:linear-gradient(135deg,#38BDF8,#818CF8);"
        "-webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🧠 Sentify AI</div>"
        "<div style='font-size:0.7rem; color:#64748B; margin-top:2px;'>STRATEGIC INTELLIGENCE v2.0</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border:none; height:1px; background:#1E293B; margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Industry Persona ──
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;'>🎯 Industry Persona</div>", unsafe_allow_html=True)
    persona = st.selectbox(
        "Persona",
        [
            "General / Cross-Industry",
            "E-Commerce & Retail",
            "SaaS & Technology",
            "Hospitality & Travel",
            "Healthcare & Wellness",
            "Financial Services",
            "Food & Beverage",
            "Education & E-Learning",
        ],
        label_visibility="collapsed",
        help="Tailors AI analysis context to your industry vertical",
    )

    # ── Languages ──
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:1px; margin:1rem 0 0.4rem 0;'>🌐 Languages</div>", unsafe_allow_html=True)
    input_lang = st.selectbox(
        "Source Dialect",
        ["Detect Automatically", "English", "Arabic / Darija", "French", "Spanish", "German", "Portuguese", "Hindi"],
        label_visibility="collapsed",
    )
    output_lang = st.selectbox(
        "Report Language",
        ["English", "Arabic (العربية)", "French (Français)", "Spanish", "German", "Portuguese"],
        label_visibility="collapsed",
    )

    # ── File Upload ──
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:1px; margin:1rem 0 0.4rem 0;'>📁 Data Source</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload .txt or .csv",
        type=["txt", "csv"],
        label_visibility="collapsed",
        help="CSV will auto-detect the text column",
    )

    st.markdown("<hr style='border:none; height:1px; background:#1E293B; margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Advanced Options ──
    with st.expander("⚙️ Advanced", expanded=False):
        model_choice = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            help="Flash = max quality, Lite = faster & cheaper",
        )
        batch_size = st.slider(
            "Batch Size (reviews)",
            min_value=10,
            max_value=500,
            value=200,
            step=10,
            help="Max reviews per API chunk. Lower = faster, Higher = more context",
        )
        st.toggle("Strip emojis", value=True, key="strip_emojis")
        st.toggle("Remove blank lines", value=True, key="remove_blanks")

    # ── Footer ──
    st.markdown("<hr style='border:none; height:1px; background:#1E293B; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.7rem; color:#475569; text-align:center;'>"
        "Powered by <b style='color:#64748B;'>Google Gemini 2.5</b><br>"
        "Data never stored server-side</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN WORKSPACE  →  Hero + Input Area
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='hero-title'>🧠 Sentify AI</div>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-sub'>Transform raw customer commentary into structured strategic intelligence</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center;'><span class='hero-badge'>✦ Structured Output  ✦ Key-Phrase Extraction  ✦ Token Optimizer</span></div>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

# ── Text Input (always visible in main area) ──
st.markdown("<div class='section-header'><span>✏️</span> Customer Feedback Payload</div>", unsafe_allow_html=True)
pasted = st.text_area(
    "Paste reviews here",
    height=180,
    placeholder="Paste customer reviews, support tickets, social media threads, or survey responses here...",
    label_visibility="collapsed",
)

# ── Resolve raw_reviews from either upload or paste ──
raw_reviews: str = ""

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(io.BytesIO(file_bytes))
            text_cols = [c for c in df_upload.columns if df_upload[c].dtype == object]
            if text_cols:
                raw_reviews = "\n".join(
                    df_upload[text_cols[0]].dropna().astype(str).tolist()
                )
                st.success(f"✅ Loaded **{len(df_upload)}** rows from `{text_cols[0]}`", icon="📊")
            else:
                st.warning("⚠️ No text columns found in CSV.")
        else:
            raw_reviews = file_bytes.decode("utf-8", errors="replace").strip()
            st.success(f"✅ Loaded **{len(raw_reviews.splitlines())}** lines from file", icon="📄")
    except Exception as file_err:
        st.error(f"❌ File error: {file_err}")
elif pasted.strip():
    raw_reviews = pasted.strip()

# ── Analysis Button ──
st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
run_analysis = st.button(
    "🚀 Execute Strategic Analysis",
    use_container_width=True,
    type="primary",
    disabled=(not raw_reviews),
)

if not raw_reviews and not run_analysis:
    st.info("ℹ️ Paste feedback above **or** upload a file via the sidebar, then click **Execute Strategic Analysis**.")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  REVIEW BATCHING & CLEANUP  →  Token Optimization Engine
# ═══════════════════════════════════════════════════════════════════════════════


def clean_reviews(text: str) -> str:
    """
    Intelligently pre-process raw review text before sending to Gemini.
    Saves tokens and improves response quality.
    """
    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove blank lines if enabled
    if st.session_state.get("remove_blanks", True):
        lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in lines if ln]
        text = "\n".join(lines)

    # 3. Strip emojis if enabled
    if st.session_state.get("strip_emojis", True):
        # Wide emoji pattern covering most Unicode emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0000200D"              # ZWJ
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

    # 4. Collapse excessive whitespace
    text = re.sub(r"\s{3,}", "\n\n", text)

    # 5. Normalize repeated punctuation
    text = re.sub(r"[!]{3,}", "!", text)
    text = re.sub(r"[?]{3,}", "?", text)
    text = re.sub(r"[.]{4,}", "...", text)

    return text.strip()


def batch_reviews(text: str, max_chars: int = 12000) -> list[str]:
    """
    Split large review payloads into manageable chunks.
    Each chunk is under max_chars to stay well within token limits.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    lines = text.split("\n")
    current_chunk: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current_len + line_len > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = line_len
        else:
            current_chunk.append(line)
            current_len += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PROMPT BUILDERS  →  Persona-aware, structured schema prompts
# ═══════════════════════════════════════════════════════════════════════════════

PERSONA_CONTEXT: dict[str, str] = {
    "General / Cross-Industry": "You are an expert business consultant analyzing customer feedback across industries.",
    "E-Commerce & Retail": "You are a senior e-commerce analyst specializing in customer experience, conversion optimization, and retail operations.",
    "SaaS & Technology": "You are a product-led growth strategist analyzing user feedback for software products, focusing on UX, onboarding, and feature adoption.",
    "Hospitality & Travel": "You are a hospitality consultant specializing in guest experience, service quality, and travel industry benchmarks.",
    "Healthcare & Wellness": "You are a healthcare quality improvement analyst focused on patient experience, care delivery, and wellness service feedback.",
    "Financial Services": "You are a fintech CX strategist analyzing customer feedback on banking, payments, and financial product experiences.",
    "Food & Beverage": "You are a foodservice industry analyst specializing in taste, quality, service speed, and dining experience metrics.",
    "Education & E-Learning": "You are an edtech learning experience designer analyzing student and educator feedback on course quality and platform usability.",
}


def build_sentiment_prompt(cleaned_text: str, persona_key: str, source_lang: str) -> str:
    ctx = PERSONA_CONTEXT.get(persona_key, PERSONA_CONTEXT["General / Cross-Industry"])
    lang_hint = f"\nThe reviews are written in: {source_lang}." if source_lang != "Detect Automatically" else ""
    return f"""{ctx}
{lang_hint}

Analyze the sentiment of the following customer reviews.
Classify the overall corpus into three categories with exact percentages.

CRITICAL RULES:
- positive + negative + neutral MUST sum to exactly 100.0
- Use exactly one decimal place for all values
- Determine the dominant sentiment category
- Provide a confidence score (0.0 to 1.0) reflecting how clear the sentiment signals are

Reviews:
{cleaned_text}
"""


def build_analysis_prompt(
    cleaned_text: str,
    persona_key: str,
    source_lang: str,
    target_lang: str,
) -> str:
    ctx = PERSONA_CONTEXT.get(persona_key, PERSONA_CONTEXT["General / Cross-Industry"])
    lang_hint = f"\nThe input reviews are written in: {source_lang}." if source_lang != "Detect Automatically" else ""
    return f"""{ctx}{lang_hint}

CRITICAL: Write the ENTIRE response in {target_lang}. Do not switch languages mid-response.

Analyze the following customer reviews and produce a structured strategic report.

Your output MUST include:

1. COMPLIMENTS: A list of specific things customers praised (bullet points)
2. COMPLAINTS: A list of specific pain points and friction areas (bullet points)
3. ADVICE: 3-5 actionable strategic recommendations prioritized by business impact
4. KEY_PHRASES: Extract the top 3-5 trending keywords/topics mentioned across reviews. For each, indicate:
   - The keyword/topic (1-3 words)
   - Frequency tier: high/medium/low
   - Sentiment polarity: positive/negative/neutral
   - Approximate review count mentioning it
5. EXECUTIVE_SUMMARY: A concise 2-3 sentence summary of the overall feedback landscape

Reviews:
{cleaned_text}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CHART BUILDER
# ═══════════════════════════════════════════════════════════════════════════════


def build_sentiment_donut(scores: SentimentResult) -> go.Figure:
    labels = ["Positive", "Negative", "Neutral"]
    values = [scores.positive, scores.negative, scores.neutral]
    colors = ["#10B981", "#EF4444", "#F59E0B"]
    emojis = ["😊", "😞", "😐"]

    dominant_emoji = emojis[values.index(max(values))]

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=[f"{lbl} {emo}" for lbl, emo in zip(labels, emojis)],
            values=values,
            hole=0.58,
            marker=dict(colors=colors, line=dict(color="#0B0F19", width=3)),
            textinfo="label+percent",
            textfont=dict(size=14, color="#E2E8F0", family="Inter, sans-serif"),
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
            pull=[0.05 if v == max(values) else 0 for v in values],
            rotation=45,
        )
    )
    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>{dominant_emoji}</b><br><span style='font-size:11px;color:#64748B'>{scores.dominant.upper()}</span>",
                x=0.5,
                y=0.5,
                font=dict(size=22, color="#E2E8F0"),
                showarrow=False,
            )
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=20, b=50, l=20, r=20),
        height=360,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  REPORT EXPORTER  →  Corporate-grade document generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_corporate_report(
    scores: SentimentResult,
    analysis: StrategicAnalysis,
    persona: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """Generate a beautifully structured corporate markdown report."""

    now = datetime.now().strftime("%B %d, %Y at %H:%M")

    # Build key phrases section
    kp_lines = []
    for kp in analysis.key_phrases:
        freq_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}[kp.frequency]
        sent_icon = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}[kp.sentiment]
        kp_lines.append(f"- {freq_icon} **{kp.keyword}** — {sent_icon} {kp.sentiment.title()} · ~{kp.review_count} mentions")

    # Build compliments
    comp_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(analysis.compliments)) if analysis.compliments else "_No compliments detected._"

    # Build complaints
    compl_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(analysis.complaints)) if analysis.complaints else "_No complaints detected._"

    # Build advice
    advice_lines = "\n".join(f"{i+1}. **{a}**" for i, a in enumerate(analysis.advice)) if analysis.advice else "_No recommendations generated._"

    report = f"""# 📊 Sentify AI — Strategic Customer Intelligence Report

---

**Generated:** {now}  
**Industry Persona:** {persona}  
**Source Language:** {source_lang}  
**Report Language:** {target_lang}  
**AI Model:** Google Gemini 2.5 Flash with Structured Output

---

## 📌 Executive Summary

{analysis.executive_summary}

---

## 📈 Sentiment Analysis Overview

| Metric | Value | Indicator |
|--------|-------|-----------|
| **Positive** | {scores.positive:.1f}% | {'🟢' if scores.positive > 50 else '⚪'} |
| **Negative** | {scores.negative:.1f}% | {'🔴' if scores.negative > 30 else '⚪'} |
| **Neutral**  | {scores.neutral:.1f}%  | {'🟡' if scores.neutral > 30 else '⚪'} |
| **Dominant** | {scores.dominant.title()} | {scores.confidence * 100:.0f}% confidence |

**Interpretation:** The feedback landscape is predominantly **{scores.dominant}** with a model confidence of {scores.confidence * 100:.0f}%.
{"This indicates strong favorable sentiment across the customer base." if scores.dominant == "positive" else "This signals significant customer friction requiring immediate strategic attention." if scores.dominant == "negative" else "This suggests a mixed or ambivalent customer experience with room for targeted improvement."}

---

## 🔑 Trending Topics & Key Phrases

The following high-frequency themes were automatically extracted from the review corpus:

{chr(10).join(kp_lines) if kp_lines else "_No key phrases extracted._"}

---

## 👍 Customer Compliments

{comp_lines}

---

## 👎 Customer Complaints

{compl_lines}

---

## 💡 Strategic Recommendations

{advice_lines}

---

## 📋 Methodology

This report was generated using **Sentify AI v2.0**, an enterprise customer feedback intelligence platform. The analysis pipeline includes:

1. **Review Pre-processing:** Automated cleanup (emoji stripping, blank-line removal, whitespace normalization)
2. **Structured AI Output:** Pydantic schema-validated responses for 100% data integrity
3. **Key-Phrase Extraction:** LLM-powered topic modeling with sentiment polarity detection
4. **Sentiment Scoring:** Three-class classification (Positive / Negative / Neutral) with confidence scoring
5. **Industry Context:** Analysis lens tailored to the {persona} vertical

---

*Report generated by Sentify AI v2.0 — Confidential*
"""
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN EXECUTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

if run_analysis:
    if not raw_reviews.strip():
        st.warning("⚠️ No review data found. Paste text or upload a file first.")
        st.stop()

    # ── 11a. Pre-process reviews ──
    with st.spinner("🔧 Cleaning & batching reviews..."):
        cleaned = clean_reviews(raw_reviews)
        batches = batch_reviews(cleaned, max_chars=batch_size * 60)
        st.toast(f"✅ Pre-processed: {len(raw_reviews.splitlines())} lines → {len(batches)} batch(es)", icon="🔧")

    # Use first batch for analysis (or we could aggregate multiple batches)
    review_payload = batches[0]

    # ── 11b. Build prompts ──
    sentiment_prompt = build_sentiment_prompt(review_payload, persona, input_lang)
    analysis_prompt = build_analysis_prompt(review_payload, persona, input_lang, output_lang)

    # ── 11c. Call Gemini with STRUCTURED OUTPUT ──
    with st.spinner("🤖 Gemini Analytics Engine running structured analysis..."):
        try:
            # --- Sentiment Scoring (Structured) ---
            sentiment_resp = client.models.generate_content(
                model=model_choice,
                contents=sentiment_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SentimentResult,
                    temperature=0.1,  # Low temp for exact scoring
                ),
            )
            scores = SentimentResult.model_validate_json(sentiment_resp.text)

            # --- Strategic Analysis (Structured) ---
            analysis_resp = client.models.generate_content(
                model=model_choice,
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=StrategicAnalysis,
                    temperature=0.3,
                ),
            )
            analysis = StrategicAnalysis.model_validate_json(analysis_resp.text)

        except Exception as api_err:
            st.error(
                f"❌ **Gemini API Error:** {api_err}\n\n"
                "Please verify your API key and quota. If using structured output, "
                "ensure you're using a compatible model (gemini-2.5-flash or newer).",
                icon="🔴",
            )
            st.stop()

    # ── 11d. Normalize scores to 100% ──
    total = scores.positive + scores.negative + scores.neutral
    if total > 0:
        scores.positive = (scores.positive / total) * 100
        scores.negative = (scores.negative / total) * 100
        scores.neutral = (scores.neutral / total) * 100

    # ── 11e. Generate corporate report ──
    corporate_md = generate_corporate_report(
        scores=scores,
        analysis=analysis,
        persona=persona,
        source_lang=input_lang,
        target_lang=output_lang,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 12.  DASHBOARD RENDERING  →  Premium visual layout
    # ═══════════════════════════════════════════════════════════════════════════

    st.success("🎯 Strategic analysis complete!", icon="✅")
    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # ── 12a. Executive Summary (Indigo) ──
    st.markdown("<div class='section-header'><span>📌</span> Executive Summary</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='exec-summary animate-in'>{analysis.executive_summary}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── 12b. KPI Stat Cards ──
    st.markdown("<div class='section-header'><span>📈</span> Sentiment Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"<div class='glow-card stat-card-emerald animate-in'>"
            f"<div class='stat-value' style='color:#34D399'>{scores.positive:.1f}%</div>"
            f"<div class='stat-label'>😊 Positive</div>"
            f"<div class='stat-delta' style='color:#10B981'>{'▲ Dominant' if scores.dominant == 'positive' else ''}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='glow-card stat-card-coral animate-in'>"
            f"<div class='stat-value' style='color:#F87171'>{scores.negative:.1f}%</div>"
            f"<div class='stat-label'>😞 Negative</div>"
            f"<div class='stat-delta' style='color:#EF4444'>{'▲ Dominant' if scores.dominant == 'negative' else ''}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='glow-card stat-card-amber animate-in'>"
            f"<div class='stat-value' style='color:#FBBF24'>{scores.neutral:.1f}%</div>"
            f"<div class='stat-label'>😐 Neutral</div>"
            f"<div class='stat-delta' style='color:#F59E0B'>{'▲ Dominant' if scores.dominant == 'neutral' else ''}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c4:
        conf_color = "#34D399" if scores.confidence >= 0.8 else "#FBBF24" if scores.confidence >= 0.5 else "#F87171"
        st.markdown(
            f"<div class='glow-card stat-card-indigo animate-in'>"
            f"<div class='stat-value' style='color:{conf_color}'>{scores.confidence * 100:.0f}%</div>"
            f"<div class='stat-label'>🎯 Confidence</div>"
            f"<div class='stat-delta' style='color:#818CF8'>{scores.dominant.title()} signal</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 12c. Donut Chart ──
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        fig = build_sentiment_donut(scores)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown("<div class='section-header'><span>🔑</span> Trending Topics</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#64748B; font-size:0.82rem; margin-bottom:0.6rem;'>Clickable sentiment-tagged keyword cloud extracted by AI</div>", unsafe_allow_html=True)

        if analysis.key_phrases:
            tags_html = "<div class='topic-cloud'>"
            for kp in analysis.key_phrases:
                tag_class = f"topic-tag-{kp.sentiment}"
                freq_class = f"topic-freq-{kp.frequency}"
                tags_html += (
                    f"<div class='topic-tag {tag_class}'>"
                    f"{kp.keyword}"
                    f"<span class='topic-freq {freq_class}'>{kp.frequency.upper()}</span>"
                    f"<span style='font-size:0.7rem; color:#64748B;'>~{kp.review_count}</span>"
                    f"</div>"
                )
            tags_html += "</div>"
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.caption("_No trending topics extracted._")

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # ── 12d. Tabbed Strategic Report ──
    st.markdown(
        f"<div class='section-header'><span>📋</span> Strategic Intelligence Report — {output_lang}</div>",
        unsafe_allow_html=True,
    )

    is_rtl = "Arabic" in output_lang

    tab_comp, tab_compl, tab_adv = st.tabs(["👍 Compliments", "👎 Complaints", "💡 Consulting Advice"])

    with tab_comp:
        if analysis.compliments:
            for item in analysis.compliments:
                st.markdown(
                    f"<div class='insight-card insight-card-compliment'>"
                    f"<span class='insight-bullet'>✦</span>"
                    f"<span class='insight-text'>{item}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("_No compliments detected._")

    with tab_compl:
        if analysis.complaints:
            for item in analysis.complaints:
                st.markdown(
                    f"<div class='insight-card insight-card-complaint'>"
                    f"<span class='insight-bullet'>▸</span>"
                    f"<span class='insight-text'>{item}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("_No complaints detected._")

    with tab_adv:
        if analysis.advice:
            for i, item in enumerate(analysis.advice, 1):
                st.markdown(
                    f"<div class='insight-card insight-card-advice'>"
                    f"<span class='insight-bullet' style='color:#818CF8'>{i}.</span>"
                    f"<span class='insight-text'>{item}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("_No recommendations generated._")

    # ── 12e. Expandable Raw JSON ──
    with st.expander("🔍 View Raw Structured Data (JSON)"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Sentiment Scores**")
            st.json(scores.model_dump())
        with col_b:
            st.markdown("**Key Phrases**")
            st.json([kp.model_dump() for kp in analysis.key_phrases])

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # ── 12f. Export Section ──
    st.markdown("<div class='section-header'><span>⬇️</span> Export Report</div>", unsafe_allow_html=True)

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        st.download_button(
            label="📄 Corporate Markdown Report (.md)",
            data=corporate_md.encode("utf-8"),
            file_name=f"sentify_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with dl_col2:
        # Plain text version (strip markdown formatting)
        txt_version = re.sub(r"#+\s*", "", corporate_md)
        txt_version = re.sub(r"\*\*(.*?)\*\*", r"\1", txt_version)
        txt_version = re.sub(r"\|", "  ", txt_version)
        txt_version = re.sub(r"-{3,}", "-" * 50, txt_version)
        st.download_button(
            label="📃 Plain Text Report (.txt)",
            data=txt_version.encode("utf-8"),
            file_name=f"sentify_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.caption(
        "📎 Reports include sentiment summaries, topic analysis, strategic recommendations, and methodology notes. "
        "All data stays local — nothing is stored on external servers."
    )