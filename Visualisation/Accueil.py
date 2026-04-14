from pathlib import Path

import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Project Presentation and Requirements Specification",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# GLOBAL STYLE
# ============================================================
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(99,102,241,0.08), transparent 24%),
        linear-gradient(180deg, #f8fbff 0%, #f6f8fc 45%, #f8fafc 100%);
}

.main .block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #0f172a;
    letter-spacing: -0.02em;
}

.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
    color: white;
    border-radius: 24px;
    padding: 1.6rem 1.6rem 1.4rem 1.6rem;
    box-shadow: 0 18px 45px rgba(30, 58, 138, 0.22);
    margin-bottom: 1.2rem;
}

.section-box {
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 24px rgba(15,23,42,0.05);
    margin-bottom: 1rem;
}

.slide-frame {
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 8px 24px rgba(15,23,42,0.05);
    margin-top: 0.8rem;
}

.badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    margin: 0.15rem 0.2rem 0.15rem 0;
    font-size: 0.88rem;
    font-weight: 600;
    background: #eef2ff;
    color: #3730a3;
    border: 1px solid #c7d2fe;
}

[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid rgba(226, 232, 240, 0.9);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
}

div[data-testid="stExpander"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(191, 219, 254, 0.9);
}

[data-testid="stButton"] button {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def hero(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div class="hero-box">
        <div style="font-size:0.8rem;font-weight:800;letter-spacing:0.10em;text-transform:uppercase;opacity:0.82;margin-bottom:0.4rem;">
            Retail Demand Forecasting
        </div>
        <div style="font-size:2.15rem;font-weight:800;line-height:1.08;margin-bottom:0.55rem;">
            {title}
        </div>
        <div style="font-size:1rem;line-height:1.72;opacity:0.92;max-width:980px;">
            {subtitle}
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_banner(index: str, title: str, description: str) -> None:
    st.markdown(f"""
    <div class="section-box">
        <div style="font-size:0.8rem;font-weight:800;text-transform:uppercase;letter-spacing:0.08em;color:#2563eb;margin-bottom:0.25rem;">
            Section {index}
        </div>
        <div style="font-size:1.35rem;font-weight:800;color:#0f172a;margin-bottom:0.35rem;">
            {title}
        </div>
        <div style="color:#475569;line-height:1.7;">
            {description}
        </div>
    </div>
    """, unsafe_allow_html=True)


def text_box(title: str, body_md: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(body_md)


def badge(label: str, value: str) -> None:
    st.markdown(
        f'<span class="badge">{label}: {value}</span>',
        unsafe_allow_html=True
    )


# ============================================================
# LOAD SLIDES
# ============================================================
slides_dir = Path("Visualisation/slide")
slides = []

if slides_dir.exists():
    slides = sorted([
        p for p in slides_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    ])

# ============================================================
# STATE
# ============================================================
if "slide_idx" not in st.session_state:
    st.session_state.slide_idx = 0

n = len(slides)

# ============================================================
# HEADER
# ============================================================
hero(
    "Project Presentation and Requirements Specification",
    "This page presents the project in slide format and summarizes the functional requirements of the pricing optimization dashboard."
)

# ============================================================
# 1. PROJECT SLIDES
# ============================================================
section_banner(
    "01",
    "Project overview in slides",
    "Visual navigation through the project presentation slides and its functional scope."
)

if n == 0:
    st.warning(f"No slides found in the folder: {slides_dir}")
    text_box(
        "Verification",
        """
Check the following points:

- the `Visualisation/slide` folder exists
- it contains `.png`, `.jpg`, `.jpeg` or `.webp` images
- the application is launched from the correct project root
"""
    )
else:
    st.markdown("### Navigation")
    badge("Number of slides", str(n))
    badge("Folder", str(slides_dir))

    c1, c2, c3 = st.columns([1.2, 2.2, 1.2])

    with c1:
        if st.button("⬅️ Previous", width="stretch"):
            st.session_state.slide_idx = (st.session_state.slide_idx - 1) % n

    with c2:
        current_label = f"Slide {st.session_state.slide_idx + 1} / {n}"
        with st.container(border=True):
            st.markdown(f"**{current_label}**")

    with c3:
        if st.button("Next ➡️", width="stretch"):
            st.session_state.slide_idx = (st.session_state.slide_idx + 1) % n

    st.markdown('<div class="slide-frame">', unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1, 3, 1])

    with col_center:
        current_slide = slides[st.session_state.slide_idx]
        st.image(str(current_slide), width="stretch")
        st.caption(f"File: {current_slide.name}")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 2. REQUIREMENTS
# ============================================================
section_banner(
    "02",
    "Requirements — ML pricing optimization",
    "Summary of objectives, expected deliverables, business constraints, and acceptance criteria."
)

text_box(
    "1) Objectives",
    """
**Main objective:**

Provide a decision-support tool enabling:

- short-term demand forecasting
- monitoring consumer price sensitivity (elasticity) by product category

**Operational objectives:**

**Demand forecasting**

- Produce demand forecasts with a **21-day horizon**

**Pricing optimization KPIs monitoring**

- Visualize price elasticity evolution over time with aggregations
- Visualize basket intensity proxy evolution over time with aggregations
"""
)

text_box(
    "2) Expected deliverables",
    """
A dashboard containing at least 3 charts:

- **Chart 1 — Demand Forecasting**
    - Time series: historical demand + 3-month forecast
    - Display confidence intervals if supported by the model

- **Chart 2 — Price elasticity evolution**
    - Elasticity curve over time
    - Filter or segmentation by product category
"""
)

text_box(
    "3) Business constraints",
    """
- Mandatory forecast horizon: **21 days**
- Visualizations and calculations must be updated monthly using new available data
"""
)

text_box(
    "4) Acceptance criteria",
    """
The deliverable is validated if:

- the forecast correctly displays a **21-day horizon**
- both charts are present and readable
- the dashboard updates monthly with new data
- elasticity is available by product category
- results are consistent, without unexplained outliers, and produce actionable curves
"""
)