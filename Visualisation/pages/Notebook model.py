from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Model Training - Sales Analysis",
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def hero(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div class="hero-box">
        <div style="font-size:0.8rem;font-weight:800;letter-spacing:0.10em;text-transform:uppercase;opacity:0.80;margin-bottom:0.4rem;">
            Retail Demand Forecasting
        </div>
        <div style="font-size:2.1rem;font-weight:800;line-height:1.1;margin-bottom:0.55rem;">
            {title}
        </div>
        <div style="font-size:1rem;line-height:1.7;opacity:0.92;max-width:950px;">
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


def summary_box(title: str, body_md: str) -> None:
    with st.container(border=True):
        st.caption(title.upper())
        st.markdown(body_md)


def badge(label: str, value: str) -> None:
    st.markdown(
        f'<span class="badge">{label}: {value}</span>',
        unsafe_allow_html=True
    )


# ============================================================
# NOTEBOOK PATH
# ============================================================
NOTEBOOK_HTML_PATH = Path("Notebooks/test_model.html")


# ============================================================
# MAIN
# ============================================================
def main() -> None:

    hero(
        "Model Training Notebook",
        "This page summarizes the structure of the model training workflow, including preprocessing, evaluation tools, model selection, and final interpretation."
    )

    # ========================================================
    # 1. STRUCTURE
    # ========================================================
    section_banner(
        "01",
        "Notebook Overview",
        "High-level structure of the training notebook, covering preprocessing, evaluation tools, model selection, and interpretation."
    )

    text_box(
        "Notebook structure",
        """
- Imports
    - Data Loading
- Preprocessing
    - Dataset Splitting
- Model Evaluation Tools
    - Metrics Analysis
    - Resource Usage Analysis
    - Learning Curves
    - SHAP Interpretation
- Training
    - Baseline
    - Machine Learning
        - Dataset Benchmarking
        - Best Model Selection
        - Final Model Training
        - Model Interpretation
"""
    )

    st.markdown("### Quick tags")
    badge("Notebook", "Model Training")
    badge("Scope", "Sales Forecasting")
    badge("Models", "ML + Tree-based")
    badge("Explainability", "SHAP")

    # ========================================================
    # 2. EMBEDDED NOTEBOOK
    # ========================================================
    section_banner(
        "02",
        "Embedded Notebook",
        "Interactive rendering of the model training notebook exported as HTML."
    )

    if NOTEBOOK_HTML_PATH.exists():

        summary_box(
            "Notebook file",
            f"""
The HTML notebook file was successfully loaded.

**Path:** `{NOTEBOOK_HTML_PATH}`
"""
        )

        with st.container(border=True):
            st.markdown("**Notebook preview**")
            html = NOTEBOOK_HTML_PATH.read_text(encoding="utf-8")
            components.html(html, height=800, scrolling=True)

    else:
        st.error(f"Notebook file not found: {NOTEBOOK_HTML_PATH}")

        text_box(
            "Troubleshooting",
            """
Check the following:

- The `Notebooks` directory exists
- The file `test_model.html` is present
- The app is launched from the correct project root
- The notebook was exported correctly in HTML format
"""
        )


if __name__ == "__main__":
    main()