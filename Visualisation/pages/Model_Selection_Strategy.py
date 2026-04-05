import streamlit as st

# ==============================
# External assets and references
# ==============================

GRAPH_1_PATH = "graphique/entrainement/Dataset_comparaison.png"   # Dataset selection based on metric scores
GRAPH_2_PATH = "graphique/entrainement/cv_performance.png"   # Best model selection - metric comparison 1
GRAPH_3_PATH = "graphique/entrainement/cv_performance_2.png"   # Best model selection - metric comparison 2
GRAPH_4_PATH = "graphique/entrainement/benchmark_model.png"   # Inference cost / resource comparison
GRAPH_5_PATH = "graphique/entrainement/learning_curve.png"   # Learning curve
GRAPH_6_PATH = "graphique/entrainement/SHAP_global_feature_importance.png"   # Global feature importance
GRAPH_7_PATH = "graphique/entrainement/SHAPE_beeswarm_summary.png"   # SHAP beeswarm summary
GRAPH_8_PATH = "graphique/entrainement/waterfall_local_explanation.png"   # SHAP waterfall
GRAPH_9_PATH = "graphique/entrainement/SHAP_dependence.png"   # SHAP dependence
GRAPH_10_PATH = "graphique/entrainement/PDP_ICE.png"  # PDP / ICE



# ==============================
# Helper
# ==============================
def show_visual(asset: str, height: int = 420) -> None:
    """Display a chart/image if a path or URL is provided, otherwise show a visual placeholder."""
    if asset and asset.strip():
        st.image(asset, width="stretch")
    else:
        st.markdown(
            f"""
            <div style="
                border: 1px dashed #9CA3AF;
                border-radius: 12px;
                padding: 1.5rem;
                height: {height}px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                color: #6B7280;
                background-color: rgba(240, 242, 246, 0.35);
            ">
                <div><strong>Graphic placeholder</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================
# Page header
# ==============================
st.title("Model Selection Strategy")
st.markdown(
    """
    This page summarizes the baseline results, model selection process, final model training,
    and model interpretation for the retail demand forecasting workflow.
    """
)



# ==============================
# 1. Baseline
# ==============================
st.header("1. Baseline")

col1, col2, col3, col4 = st.columns(4)
col1.metric("CV_MAE", "0.40")
col2.metric("TEST_MAE", "0.32")
col3.metric("CV_RMSE", "0.86")
col4.metric("Generalization", "Good")

st.markdown(
    """
    - **Solid overall performance.** The baseline already delivers a reasonable average validation error.
    - **Good generalization.** Test MAE is lower than cross-validation MAE, which is a positive signal and does not suggest overfitting.
    - **Moderate variability.** Fold-to-fold variation remains visible but still acceptable for a baseline model.
    - **RMSE remains higher than MAE.** This indicates the presence of larger individual errors, likely driven by skewness and outliers.
    """
)


# ==============================
# 2. Machine Learning
# ==============================
st.header("2. Machine Learning")

# Dataset selection
st.subheader("Dataset Selection Based on Metric Scores")
st.markdown(
    """
    - The most performant dataset is **df_feature_1**, even though it contains fewer explanatory variables.
      Its main advantage is the much larger sample size, with roughly **7 times more observations**, which improves
      generalization and reduces overfitting risk.
    - In both cross-validation and test evaluation, the best model trained on **df_feature_1** achieves lower MAE and RMSE values.
      This confirms that, in this project, **data quantity has a stronger impact than feature count**.
    """
)
show_visual(GRAPH_1_PATH, height=440)

# Best model selection
st.subheader("Best Model Selection")
st.markdown(
    """
    The overall consistency between cross-validation and test performance indicates sound generalization.
    """
)
show_visual(GRAPH_2_PATH, height=430)
show_visual(GRAPH_3_PATH, height=430)

st.markdown(
    """
    - **RandomForestRegressor** achieves the best overall performance, with the lowest average MAE and RMSE.
      The gap between cross-validation and test scores remains small, suggesting a stable model with a strong bias-variance trade-off.
    - **SGDRegressor** shows intermediate performance. Its cross-validation variability is noticeably higher,
      which indicates stronger sensitivity to data fluctuations. However, test performance stays close to the CV mean,
      so there is no clear sign of severe overfitting.
    - **HistGradientBoostingRegressor** produces slightly higher average errors than Random Forest.
      Still, the test scores remain inside the cross-validation error range, which supports correct generalization,
      even if predictive performance is slightly weaker on this dataset.
    """
)

show_visual(GRAPH_4_PATH, height=450)

st.markdown(
    """
    - **SGDRegressor** is the lightest and fastest model. It combines low latency, very high throughput,
      and negligible model size, making it well suited to low-latency environments.
    - **RandomForestRegressor** is significantly more expensive at inference time. Latency is higher,
      throughput is lower, and the serialized model is much larger. Despite this, it delivers the best predictive performance,
      which illustrates the classic trade-off between accuracy and computational cost.
    - **HistGradientBoostingRegressor** offers a strong compromise. It combines very low latency, high throughput,
      and compact model size, but with higher CPU usage during inference.
    - **RandomForestRegressor is retained** because it delivers the best predictive performance while keeping test results
      aligned with cross-validation behavior.
    """
)


# Final model training
st.subheader("Final Model Training")

with st.container(border=True):
    st.code(
        """{
    'dataset': 'df_feature_1',
    'best_model': 'RandomForestRegressor',
    'CV_MAE': 0.39583872514860585,
    'CV_MAE_STD': 0.09727518402735794,
    'CV_RMSE': 0.8615471732631826,
    'CV_RMSE_STD': 0.1654603461436886,
    'TEST_MAE': 0.4095647344569175,
    'TEST_RMSE': 0.6145572826379082,
    'best_params': {
        'model__bootstrap': True,
        'model__max_depth': None,
        'model__max_features': 0.5,
        'model__min_samples_leaf': 1,
        'model__min_samples_split': 2,
        'model__n_estimators': 300
    }
}""",
        language="python",
    )

st.markdown(
    """
    The final training results confirm that **RandomForestRegressor on df_feature_1** remains the best configuration.
    Cross-validation performance is strong, and the test RMSE stays lower than the CV RMSE, which supports the conclusion
    that the selected model generalizes well. The test MAE is slightly above the CV MAE, but the overall gap remains limited
    and still compatible with a stable final model.
    """
)

show_visual(GRAPH_5_PATH, height=450)

st.markdown(
    """
    - The learning curve shows **strong overfitting with small training samples**, as reflected by a large gap
      between training and cross-validation scores.
    - As training size increases, generalization improves significantly and the gap narrows.
    - Performance appears to stabilize around **5,000 observations**, after which additional data brings
      more limited marginal gains.
    """
)


# Model interpretation
st.subheader("Model Interpretation")

col_left, col_right = st.columns(2)
with col_left:
    show_visual(GRAPH_6_PATH, height=430)
with col_right:
    show_visual(GRAPH_7_PATH, height=430)

st.markdown(
    """
    - **Global feature importance.** Elasticity-related features dominate the model, which means the prediction logic is driven primarily by price sensitivity.
      Demand variation features and temporal effects also contribute meaningfully, while raw price has a lower direct effect because its information is already embedded in elasticity features.
    - **SHAP beeswarm summary.** High elasticity values tend to increase the prediction, while low elasticity values reduce it.
      Demand variation and cyclic time effects still matter, but their impact is more moderate and more dispersed across observations.
    """
)

show_visual(GRAPH_8_PATH, height=430)

st.markdown(
    """
    - **SHAP waterfall.** For the selected instance, elasticity and price-related features push the prediction downward.
      A smaller set of variables, such as style and cyclic effects, partially offsets this downward pressure.
      The final prediction is therefore mostly driven by strong negative contributions.
    """
)

show_visual(GRAPH_9_PATH, height=430)

st.markdown(
    """
    - **SHAP dependence.** Price does not show a strong or strictly structured effect by itself.
      Its impact depends on the elasticity context and can shift from slightly positive to slightly negative across observations.
      This supports the interpretation that the price effect is **context-dependent rather than directly linear**.
    """
)

show_visual(GRAPH_10_PATH, height=430)

st.markdown(
    """
    - **PDP and ICE.** On average, price has a weak and globally stable effect on the prediction.
      However, the individual ICE curves reveal strong heterogeneity, meaning that the local effect of price varies substantially from one observation to another.
      This again confirms that the model captures a **conditional price effect** rather than a single average relationship.
    """
)

