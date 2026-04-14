from pathlib import Path

import streamlit as st

# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="Model Selection Strategy",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# Minimal global CSS
# ==============================
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

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 16px;
    padding: 0.8rem 0.9rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
}

div[data-testid="stExpander"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(191, 219, 254, 0.9);
}

[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid rgba(226, 232, 240, 0.9);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
}

.section-box {
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 24px rgba(15,23,42,0.05);
    margin-bottom: 1rem;
}

.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
    color: white;
    border-radius: 24px;
    padding: 1.6rem 1.6rem 1.4rem 1.6rem;
    box-shadow: 0 18px 45px rgba(30, 58, 138, 0.22);
    margin-bottom: 1.2rem;
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

.badge-good {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}

.badge-bad {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Assets
# ==============================
GRAPH_1_PATH = Path("graphique/entrainement/Dataset_comparaison.png")
GRAPH_2_PATH = Path("graphique/entrainement/cv_performance.png")
GRAPH_3_PATH = Path("graphique/entrainement/cv_performance_2.png")
GRAPH_4_PATH = Path("graphique/entrainement/benchmark_model.png")
GRAPH_5_PATH = Path("graphique/entrainement/learning_curve.png")
GRAPH_6_PATH = Path("graphique/entrainement/SHAP_global_feature_importance.png")
GRAPH_7_PATH = Path("graphique/entrainement/SHAPE_beeswarm_summary.png")
GRAPH_8_PATH = Path("graphique/entrainement/waterfall_local_explanation.png")
GRAPH_9_PATH = Path("graphique/entrainement/SHAP_dependence.png")
GRAPH_10_PATH = Path("graphique/entrainement/PDP_ICE.png")

# ==============================
# Data
# ==============================
BASELINE = {
    "model": "Baseline",
    "CV_MAE": 0.40,
    "TEST_MAE": 0.32,
    "CV_RMSE": 0.86,
}

FINAL_RESULTS = {
    "dataset": "df_feature_1",
    "best_model": "RandomForestRegressor",
    "CV_MAE": 0.39583872514860585,
    "CV_MAE_STD": 0.09727518402735794,
    "CV_RMSE": 0.8615471732631826,
    "CV_RMSE_STD": 0.1654603461436886,
    "TEST_MAE": 0.4095647344569175,
    "TEST_RMSE": 0.6145572826379082,
    "best_params": {
        "model__bootstrap": True,
        "model__max_depth": None,
        "model__max_features": 0.5,
        "model__min_samples_leaf": 1,
        "model__min_samples_split": 2,
        "model__n_estimators": 300,
    },
}

INTERPRETATION_INSIGHTS = {
    "top_feature": "Elasticity",
    "top_feature_importance": 14,
    "top3_cumulative_importance": 41,
    "price_effect_range_min": -8.0,
    "price_effect_range_max": 2.0,
    "dominant_effect": "Context-dependent",
    "main_signal": "Price sensitivity",
}

# ==============================
# Helpers
# ==============================
def format_metric(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"

def format_pct(value: float, decimals: int = 1) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"

def format_pct_plain(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"

def compute_improvement(baseline: float, final: float) -> float:
    if baseline == 0:
        return 0.0
    return ((baseline - final) / baseline) * 100

def get_generalization_label(cv_mae: float, test_mae: float, tolerance: float = 0.05) -> str:
    if cv_mae == 0:
        return "Stable"

    relative_gap = abs(test_mae - cv_mae) / cv_mae

    if test_mae < cv_mae:
        return "Very Good"
    elif relative_gap <= tolerance:
        return "Good"
    elif relative_gap <= 0.15:
        return "Acceptable"
    return "Monitor"

def show_visual(asset: Path, height: int = 420) -> None:
    if asset.exists():
        st.image(str(asset), width="stretch")
    else:
        st.info(f"Graphic placeholder — missing file: {asset}")

def hero(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div class="hero-box">
        <div style="font-size:0.8rem;font-weight:800;letter-spacing:0.10em;text-transform:uppercase;opacity:0.80;margin-bottom:0.4rem;">
            Retail Demand Forecasting
        </div>
        <div style="font-size:2.1rem;font-weight:800;line-height:1.1;margin-bottom:0.55rem;">
            {title}
        </div>
        <div style="font-size:1rem;line-height:1.7;opacity:0.92;max-width:900px;">
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

def summary_box(title: str, body_md: str) -> None:
    with st.container(border=True):
        st.caption(title.upper())
        st.markdown(body_md)

def text_box(title: str, body_md: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(body_md)

def metric_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(item["label"], item["value"], help=item.get("help"))

def badge(label: str, value: str, positive: bool | None = None):
    if positive is True:
        cls = "badge badge-good"
    elif positive is False:
        cls = "badge badge-bad"
    else:
        cls = "badge"
    st.markdown(
        f'<span class="{cls}">{label}: {value}</span>',
        unsafe_allow_html=True
    )

# ==============================
# Derived metrics
# ==============================
baseline_generalization = get_generalization_label(
    BASELINE["CV_MAE"],
    BASELINE["TEST_MAE"]
)

final_generalization = get_generalization_label(
    FINAL_RESULTS["CV_MAE"],
    FINAL_RESULTS["TEST_MAE"]
)

cv_mae_improvement = compute_improvement(BASELINE["CV_MAE"], FINAL_RESULTS["CV_MAE"])
test_mae_improvement = compute_improvement(BASELINE["TEST_MAE"], FINAL_RESULTS["TEST_MAE"])
cv_rmse_improvement = compute_improvement(BASELINE["CV_RMSE"], FINAL_RESULTS["CV_RMSE"])

# ==============================
# Header
# ==============================
hero(
    "Model Selection Strategy",
    "This dashboard summarizes the baseline benchmark, the model selection workflow, the final training configuration, and the interpretation insights for the retail demand forecasting project."
)

# ==============================
# 1. Baseline
# ==============================
section_banner(
    "01",
    "Baseline",
    "Reference benchmark used to assess whether the final pipeline delivers a meaningful gain in predictive performance and generalization."
)

summary_box(
    "Reference configuration",
    """
**Baseline**

This benchmark provides the initial anchor point for the full modeling workflow.  
All subsequent training results are evaluated relative to this baseline.
"""
)

metric_row([
    {"label": "CV_MAE", "value": format_metric(BASELINE["CV_MAE"], 2), "help": "Cross-validation mean absolute error"},
    {"label": "TEST_MAE", "value": format_metric(BASELINE["TEST_MAE"], 2), "help": "Hold-out test mean absolute error"},
    {"label": "CV_RMSE", "value": format_metric(BASELINE["CV_RMSE"], 2), "help": "Cross-validation root mean squared error"},
    {"label": "Generalization", "value": baseline_generalization, "help": "Consistency between CV and test"},
])

text_box(
    "Baseline insights",
    """
- **Solid overall performance.** The baseline already delivers a reasonable validation error.
- **Good generalization.** Test MAE is lower than CV MAE, which is a positive signal.
- **Moderate variability.** Fold-to-fold fluctuations remain acceptable for a benchmark model.
- **RMSE remains above MAE.** This suggests the presence of larger isolated errors driven by skewness and outliers.
"""
)

# ==============================
# 2. Machine Learning
# ==============================
section_banner(
    "02",
    "Machine Learning",
    "Model comparison process combining dataset choice, cross-validation performance, and inference trade-offs."
)

st.subheader("Dataset Selection Based on Metric Scores")
text_box(
    "Dataset selection insight",
    """
- The most performant dataset is **df_feature_1**, even though it contains fewer explanatory variables.
- Its main strength is the much larger sample size, with roughly **7 times more observations**.
- This improves generalization and reduces overfitting risk.
- The result confirms that **data quantity has a stronger impact than feature count** in this project.
"""
)
show_visual(GRAPH_1_PATH, height=440)

st.subheader("Best Model Selection")
text_box(
    "Model comparison insight",
    """
The overall consistency between cross-validation and test performance indicates sound generalization across the candidate models.
"""
)
show_visual(GRAPH_2_PATH, height=430)
show_visual(GRAPH_3_PATH, height=430)

text_box(
    "Selection summary",
    """
- **RandomForestRegressor** achieves the best overall performance, with the lowest average MAE and RMSE.
- **SGDRegressor** shows intermediate performance and higher variability across folds.
- **HistGradientBoostingRegressor** remains competitive but slightly weaker in predictive accuracy on this dataset.
- The selected model offers the strongest **bias-variance trade-off**.
"""
)

show_visual(GRAPH_4_PATH, height=450)

text_box(
    "Inference benchmark insight",
    """
- **SGDRegressor** is the lightest and fastest model, suitable for low-latency environments.
- **RandomForestRegressor** is more expensive at inference time, with higher latency and larger model size.
- **HistGradientBoostingRegressor** offers a balanced compromise between compactness and speed.
- **RandomForestRegressor is retained** because predictive quality remains the main objective and it performs best on that criterion.
"""
)

# ==============================
# 3. Final model training
# ==============================
section_banner(
    "03",
    "Final Model Training",
    "Final retained configuration, validation stability, and selected hyperparameters."
)

summary_box(
    "Final selected configuration",
    f"""
**{FINAL_RESULTS["best_model"]}**

Trained on **{FINAL_RESULTS["dataset"]}** with the best trade-off between predictive performance, stability, and generalization.
"""
)

metric_row([
    {"label": "CV_MAE", "value": format_metric(FINAL_RESULTS["CV_MAE"]), "help": "Cross-validation mean absolute error"},
    {"label": "TEST_MAE", "value": format_metric(FINAL_RESULTS["TEST_MAE"]), "help": "Hold-out test mean absolute error"},
    {"label": "CV_RMSE", "value": format_metric(FINAL_RESULTS["CV_RMSE"]), "help": "Cross-validation root mean squared error"},
    {"label": "TEST_RMSE", "value": format_metric(FINAL_RESULTS["TEST_RMSE"]), "help": "Hold-out test root mean squared error"},
])

metric_row([
    {"label": "CV_MAE_STD", "value": format_metric(FINAL_RESULTS["CV_MAE_STD"]), "help": "Fold-to-fold MAE variability"},
    {"label": "CV_RMSE_STD", "value": format_metric(FINAL_RESULTS["CV_RMSE_STD"]), "help": "Fold-to-fold RMSE variability"},
    {"label": "Dataset", "value": FINAL_RESULTS["dataset"], "help": "Selected training dataset"},
    {"label": "Generalization", "value": final_generalization, "help": "Consistency between CV and test"},
])

st.markdown("#### Selected configuration details")
badge("Model", FINAL_RESULTS["best_model"])
badge("Dataset", FINAL_RESULTS["dataset"])
badge("Bootstrap", str(FINAL_RESULTS["best_params"]["model__bootstrap"]))
badge("Max features", str(FINAL_RESULTS["best_params"]["model__max_features"]))
badge("n_estimators", str(FINAL_RESULTS["best_params"]["model__n_estimators"]))

with st.expander("Show full best hyperparameters"):
    st.json(FINAL_RESULTS["best_params"])

text_box(
    "Final training insight",
    """
The final training results confirm that **RandomForestRegressor on df_feature_1** remains the best configuration.

Cross-validation performance is strong, and the test RMSE stays lower than the CV RMSE, which supports the conclusion that the selected model generalizes well.

The test MAE is slightly above the CV MAE, but the gap remains limited and still compatible with a stable final model.
"""
)

# ==============================
# 4. Improvement vs Baseline
# ==============================
section_banner(
    "04",
    "Improvement vs Baseline",
    "Relative performance evolution between the benchmark model and the final retained training pipeline."
)

summary_box(
    "Performance evolution",
    """
**Baseline vs Final Model**

This section quantifies the percentage change between the baseline and the final configuration.  
A positive value indicates an improvement, while a negative value indicates a degradation.
"""
)

metric_row([
    {"label": "CV_MAE Evolution", "value": format_pct(cv_mae_improvement), "help": "Percentage change vs baseline"},
    {"label": "TEST_MAE Evolution", "value": format_pct(test_mae_improvement), "help": "Percentage change vs baseline"},
    {"label": "CV_RMSE Evolution", "value": format_pct(cv_rmse_improvement), "help": "Percentage change vs baseline"},
])


text_box(
    "Evolution insight",
    f"""
- **CV MAE evolution:** {format_pct(cv_mae_improvement)}. The final model remains nearly stable relative to the baseline on cross-validation.
- **TEST MAE evolution:** {format_pct(test_mae_improvement)}. This is the most important business-facing comparison because it reflects out-of-sample predictive quality.
- **CV RMSE evolution:** {format_pct(cv_rmse_improvement)}. This measures whether the final model better controls larger errors than the baseline.
"""
)

show_visual(GRAPH_5_PATH, height=450)

text_box(
    "Learning curve insight",
    """
- The learning curve shows **strong overfitting with small training samples**, with a large train / validation gap.
- As training size increases, generalization improves and the gap narrows significantly.
- Performance appears to stabilize around **5,000 observations**, after which additional data yields smaller marginal gains.
"""
)

# ==============================
# 5. Model interpretation
# ==============================
section_banner(
    "05",
    "Model Interpretation",
    "Global and local explainability results extracted from feature importance, SHAP analysis, and PDP / ICE behavior."
)

summary_box(
    "Model interpretation insights",
    """
**Key drivers and local behavior**

These indicators summarize the most important interpretation results extracted from global importance, SHAP analysis, and PDP/ICE behavior.
"""
)

metric_row([
    {"label": "Top Feature", "value": INTERPRETATION_INSIGHTS["top_feature"], "help": "Most influential variable in the model"},
    {"label": "Top Feature Importance", "value": format_pct_plain(INTERPRETATION_INSIGHTS["top_feature_importance"]), "help": "Global contribution share"},
    {"label": "Top 3 Features", "value": format_pct_plain(INTERPRETATION_INSIGHTS["top3_cumulative_importance"]), "help": "Cumulative importance"},
    {"label": "Main Signal", "value": INTERPRETATION_INSIGHTS["main_signal"], "help": "Primary business driver"},
])

metric_row([
    {"label": "Price Effect Min", "value": format_pct_plain(INTERPRETATION_INSIGHTS["price_effect_range_min"]), "help": "Lower bound across observations"},
    {"label": "Price Effect Max", "value": format_pct_plain(INTERPRETATION_INSIGHTS["price_effect_range_max"]), "help": "Upper bound across observations"},
    {"label": "Effect Type", "value": INTERPRETATION_INSIGHTS["dominant_effect"], "help": "Global interpretation"},
])

text_box(
    "Interpretation summary",
    f"""
- **{INTERPRETATION_INSIGHTS["top_feature"]}** is the strongest driver of the model, with an estimated contribution of **{format_pct_plain(INTERPRETATION_INSIGHTS["top_feature_importance"])}**.
- The **top 3 variables** explain approximately **{format_pct_plain(INTERPRETATION_INSIGHTS["top3_cumulative_importance"])}** of the model behavior.
- The estimated **price effect** varies from **{format_pct_plain(INTERPRETATION_INSIGHTS["price_effect_range_min"])}** to **{format_pct_plain(INTERPRETATION_INSIGHTS["price_effect_range_max"])}**, confirming strong heterogeneity across products.
- The dominant economic signal captured by the model is **{INTERPRETATION_INSIGHTS["main_signal"]}**, and the overall effect remains **{INTERPRETATION_INSIGHTS["dominant_effect"].lower()}** rather than purely linear.
"""
)

col_left, col_right = st.columns(2)
with col_left:
    show_visual(GRAPH_6_PATH, height=430)
with col_right:
    show_visual(GRAPH_7_PATH, height=430)

text_box(
    "Global explanation insight",
    """
- **Global feature importance.** Elasticity-related features dominate the model, meaning the prediction logic is driven primarily by price sensitivity.
- **SHAP beeswarm summary.** High elasticity values tend to increase the prediction, while low elasticity values reduce it.
- Demand variation and cyclic temporal effects still matter, but their impact is more moderate and more dispersed across observations.
"""
)

show_visual(GRAPH_8_PATH, height=430)

text_box(
    "Local explanation insight",
    """
- **SHAP waterfall.** For the selected instance, elasticity and price-related features push the prediction downward.
- A smaller set of variables, such as style and cyclic effects, partially offsets this downward pressure.
- The final prediction is therefore mostly driven by strong negative contributions.
"""
)

show_visual(GRAPH_9_PATH, height=430)

text_box(
    "Dependence insight",
    """
- **SHAP dependence.** Price does not show a strong or strictly structured effect by itself.
- Its impact depends on the elasticity context and can shift from slightly positive to slightly negative across observations.
- This supports the interpretation that the price effect is **context-dependent rather than directly linear**.
"""
)

show_visual(GRAPH_10_PATH, height=430)

text_box(
    "PDP / ICE insight",
    """
- On average, price has a weak and globally stable effect on the prediction.
- However, the individual ICE curves reveal **strong heterogeneity**, meaning that the local effect of price varies substantially from one observation to another.
- This confirms that the model captures a **conditional price effect** rather than a single average relationship.
"""
)