from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="EDA - Transaction Analysis",
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

div[data-testid="stExpander"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(191, 219, 254, 0.9);
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

.notebook-frame {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 18px;
    padding: 0.75rem;
    box-shadow: 0 8px 24px rgba(15,23,42,0.05);
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


def badge(label: str, value: str) -> None:
    st.markdown(
        f'<span class="badge">{label}: {value}</span>',
        unsafe_allow_html=True
    )


# ============================================================
# NOTEBOOK PATH
# ============================================================
NOTEBOOK_HTML_PATH = Path("Notebooks/exploratory_data_analysis.html")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    hero(
        "Exploratory Data Analysis Notebook",
        "This page provides a structured view of the EDA notebook, including its analytical sections, cleaning workflow, statistical exploration, and embedded HTML notebook rendering."
    )

    section_banner(
        "01",
        "Notebook Overview",
        "High-level structure of the exploratory notebook, covering data loading, cleaning, analysis, feature engineering, and final verification."
    )

    text_box(
        "Notebook structure",
        """
- Imports
- Load and Overview of Datasets
    - Memory Usage Analysis
- Data Cleaning
    - Missing Values
    - Duplicate Values
    - Outliers, Inconsistent, and Invalid Data
    - Column Type Conversion
    - Outlier Visualization and IQR Analysis
    - Cleaned and Filtered Dataset
- Analysis
    - Univariate Analysis
        - Analysis of Numerical Variables
        - Analysis of Categorical Variables
    - Bivariate Analysis
        - Correlation Analysis
        - Trend Analysis
    - Statistical Tests
    - Price Elasticity
    - Basket Intensity Proxy
- Feature Engineering
    - ML Datasets
- Final Verification
    - Filter to Remove Non-Relevant Columns
"""
    )

    st.markdown("### Quick tags")
    badge("Notebook", "EDA")
    badge("Scope", "Transaction Analysis")
    badge("Format", "Embedded HTML")
    badge("Layout", "Wide")

    section_banner(
        "02",
        "Embedded Notebook",
        "Interactive rendering of the exported exploratory notebook inside the Streamlit page."
    )

    if NOTEBOOK_HTML_PATH.exists():
        text_box(
            "Notebook rendering",
            f"""
The HTML notebook file was found successfully.

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
Check the following points:

- The `Notebooks` folder exists at the expected location.
- The file name is exactly `exploratory_data_analysis.html`.
- The app is launched from the correct project root.
- The exported notebook is encoded in UTF-8.
"""
        )


if __name__ == "__main__":
    main()