import streamlit as st

# ============================================================
# GRAPHIC ASSET VARIABLES
# ============================================================
GRAPH_DATASET_SOURCE = "https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data/data?select=Expense+IIGF.csv"
GRAPH_OUTLIERS_QTY = "graphique/EDA/qty_outliers.png"
GRAPH_OUTLIERS_PRICE = "graphique/EDA/price_outliers.png"
GRAPH_DIST_QTY = "graphique/EDA/Distribution_qty.png"
GRAPH_DIST_PRICE = "graphique/EDA/Distribution_price.png"
GRAPH_DIST_CATEGORY = "graphique/EDA/Distribution_category.png"
GRAPH_CORR_PRICE_QTY = "graphique/EDA/scatter.png"
GRAPH_MONTHLY_SALES_TREND = "graphique/EDA/trend_qty.png"
GRAPH_MONTHLY_MEDIAN_PRICE = "graphique/EDA/trend_price.png"
GRAPH_QQ_PLOT = "graphique/EDA/QQ_plot.png"
GRAPH_ELASTICITY = "graphique/EDA/elasticity.png"
GRAPH_BASKET_INTENSITY = "graphique/EDA/Basket_intensity.png"

# ------------------------------------------------------------
# FEATURE ENGINEERING VARIABLES
# ------------------------------------------------------------
cat_features_f1 = ['Style', 'Category', 'saison']
cat_features_f2 = ['Category', 'saison']

num_features_f1 = [
    'price',
    'log_price',
    'dlog_qty_SKU',
    'dlog_price_SKU',
    'dlog_qty_Style',
    'dlog_price_Style',
    'dlog_qty_Category',
    'dlog_price_Category',
    'elasticity_rolling_SKU',
    'elasticity_rolling_Style',
    'elasticity_rolling_Category',
    'var_elasticity_rolling_SKU',
    'var_elasticity_rolling_Style',
    'var_elasticity_rolling_Category',
]

num_features_f2 = [
    'price',
    'roll_mean_7',
    'roll_std_7',
    'roll_mean_14',
    'roll_std_14',
    'roll_mean_28',
    'roll_std_28',
    'lag_1',
    'lag_7',
    'lag_14',
    'lag_28',
    'log_price',
    'dlog_qty_SKU',
    'dlog_price_SKU',
    'dlog_qty_Style',
    'dlog_price_Style',
    'dlog_qty_Category',
    'dlog_price_Category',
    'elasticity_rolling_SKU',
    'elasticity_rolling_Style',
    'elasticity_rolling_Category',
    'var_elasticity_rolling_SKU',
    'var_elasticity_rolling_Style',
    'var_elasticity_rolling_Category',
]

pas_transf = [
    'year',
    'flag_imputed_elasticity_rolling_SKU',
    'flag_imputed_var_elasticity_rolling_SKU',
    'flag_imputed_elasticity_rolling_Category',
    'flag_imputed_var_elasticity_rolling_Category',
    'flag_imputed_elasticity_rolling_Style',
    'flag_imputed_var_elasticity_rolling_Style',
]

cyclic_features_f1 = ['month_sin', 'month_cos', 'week_sin', 'week_cos', 'day_sin', 'day_cos']
cyclic_features_f2 = ['month_sin', 'month_cos', 'week_sin', 'week_cos', 'day_sin', 'day_cos']

def show_visual(asset: str, height: int = 420) -> None:
    if asset and asset.strip():
        st.image(asset, width='stretch')
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
                <div>
                    Graphic placeholder
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def feature_block(title: str, values: list[str]) -> None:
    st.markdown(f"**{title}**")
    st.code("[\n    " + ",\n    ".join([repr(v) for v in values]) + "\n]", language="python")


def main() -> None:

    st.title("Data Quality, Analysis, and Feature Engineering")

    # ========================================================
    # 1. DATA QUALITY
    # ========================================================
    st.header("1. Data Quality")

    st.subheader("a. General information")
    st.markdown(f"**Source dataset:** [Kaggle dataset]({GRAPH_DATASET_SOURCE})")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", "35,305")
        st.metric("Columns", "11")
    with col2:
        st.metric("Dataset RAM usage", "0.02 GB")
        st.metric("Available RAM used", "4%")

    with st.expander("Memory usage analysis", expanded=True):
        st.markdown(
            """
            - The dataset contains approximately **35,305 rows and 11 columns**.
            - The dataset occupies around **0.02 GB of RAM**, representing only **4% of the available memory**, which indicates very low memory pressure.
            - **Pandas is fully sufficient** for all EDA and preprocessing operations in this project.
            - The total process memory usage is approximately **1.35 GB**, which remains well below the available RAM (**46.6 GB**).
            - The dataset was enriched through merge operations to add categorical information to the original *International Sales Report* dataset.
            - Some columns contain mixed data types and therefore require preprocessing to ensure consistency.
            """
        )

    st.subheader("b. Data cleaning")
    st.markdown(
        """
        - The dataset was filtered to keep only the variables relevant to the analytical objective.
        - Only a small number of missing values were detected.
        - Data types were converted to ensure consistency and correct downstream processing.
        - Duplicate records were identified and removed.
        - Outliers and inconsistent values were identified and handled.
        """
    )

    outlier_col1, outlier_col2 = st.columns(2)
    with outlier_col1:
        show_visual(GRAPH_OUTLIERS_QTY)
    with outlier_col2:
        show_visual(GRAPH_OUTLIERS_PRICE)

    st.markdown(
        """
        **Interpretation of outliers**

        - **Quantity hypothesis:** the distribution suggests the coexistence of multiple customer segments, including individual buyers and professional buyers.
        - **Price hypothesis:** the distribution suggests the presence of multiple product tiers, such as standard, premium, and luxury products.

        **Conclusion:** outliers were retained because they carry meaningful business information rather than representing pure noise.
        """
    )

    st.markdown(
        """
        **Data inconsistencies identified**

        - The **customer** column contains date information (month and year).
        - The **size** column contains quantity values instead of size labels.
        - The **date** column contains customer names.

        These issues suggest data entry errors or data integration problems.
        After a detailed review, they were corrected using **mask-based rules** and **regular expressions** to restructure the dataset.
        """
    )

    # ========================================================
    # 2. ANALYSIS
    # ========================================================
    st.header("2. Analysis")

    st.subheader("a. Univariate analysis")
    uni1, uni2, uni3 = st.columns(3)
    with uni1:
        show_visual(GRAPH_DIST_QTY, height=360)
    with uni2:
        show_visual(GRAPH_DIST_PRICE, height=360)
    with uni3:
        show_visual(GRAPH_DIST_CATEGORY, height=360)

    st.markdown(
        """
        - **Quantity distribution:** the target is highly concentrated around low sales volumes, which indicates a right-skewed distribution. This suggests that most SKUs sell in small quantities while a limited number of observations account for much larger volumes.
        - **Price distribution:** prices are dispersed across several ranges, which supports the existence of heterogeneous product positioning and multiple commercial tiers.
        - **Category distribution:** the category frequencies are unbalanced, meaning that some product families are much more represented than others. This imbalance must be considered when interpreting descriptive statistics and model behavior.
        """
    )

    st.subheader("b. Bivariate analysis")
    show_visual(GRAPH_CORR_PRICE_QTY)
    st.markdown(
        """
        - Certain categories, including **Kurta**, **Bottom**, and **Top**, show a positive correlation between variables.
        - The **Set** category exhibits a weak correlation between price and quantity.
        - No significant correlation is observed for **Ethnic Dress**.
        - Overall, there is no strong evidence of a global linear relationship between price and quantity.
        """
    )

    show_visual(GRAPH_MONTHLY_SALES_TREND)
    st.markdown(
        """
        - The three main product groups follow a broadly similar pattern.
        - Two pronounced peaks appear near the end of each year.
        - These peaks may be explained by year-end holidays and promotional events such as **Black Friday**.
        """
    )

    show_visual(GRAPH_MONTHLY_MEDIAN_PRICE)
    st.markdown(
        """
        - For **Set**, prices remain relatively stable between **$800 and $1,100**, with lower variation in 2021 than in 2022.
        - For **Kurta**, prices range approximately between **$400 and $600**, with a notable spike in **May 2022** that may reflect either a pricing anomaly or a data consistency issue.
        - For **Top**, prices remain relatively stable between **$300 and $550**.
        """
    )

    show_visual(GRAPH_QQ_PLOT)
    st.markdown(
        """
        - The OLS model reveals a statistically significant linear signal for some explanatory variables.
        - However, the explanatory power remains limited (**R² ≈ 0.22**), meaning that a large share of the variance is not captured by linear effects.
        - The high dimensionality induced by categorical encoding (**around 1,000 features**) creates a substantial risk of multicollinearity.
        - As a result, coefficients become unstable, difficult to interpret, and economically fragile.
        - The QQ plot shows a clear deviation from normality, with skewness, heavy tails, and significant outliers.
        - The very high kurtosis confirms a strongly leptokurtic residual distribution, which points to a noisy and non-Gaussian target.
        - These results indicate that key OLS assumptions, especially **normality of residuals** and **homoscedasticity**, are violated.
        - The OLS step therefore remains exploratory: it helps detect first-order linear signals and diagnose modeling limits.
        - This justifies the use of more robust non-linear machine learning models such as **Random Forest** and **Gradient Boosting**.
        """
    )

    show_visual(GRAPH_ELASTICITY)
    st.markdown(
        """
        **Category-level analysis**

        - **Kurta** shows elasticity values consistently close to zero, indicating low price sensitivity and relatively inelastic demand.
        - **Set** and **Top** show larger elasticity fluctuations during 2021, including both positive and negative spikes.
        - This suggests stronger responsiveness to price changes, possibly driven by promotional activity or pricing strategy adjustments.
        - Elasticity variance also differs across categories:
            - **Saree** displays high variance, which indicates a more volatile and less predictable response to price changes.
            - **Kurta** maintains low variance, confirming a more stable and resilient demand pattern.

        **SKU-level analysis**

        - Some SKUs exhibit almost constant elasticity, often close to **-1** or **0**, which suggests a stable price-demand relationship over time.
        - Other SKUs show a marked increase in elasticity variance, especially from 2022 onward, reflecting a more unstable and evolving price sensitivity.
        - Sharp drops followed by recoveries may be driven by:
            - pricing strategy changes,
            - promotional effects,
            - structural shifts in demand dynamics.
        """
    )

    show_visual(GRAPH_BASKET_INTENSITY)
    st.markdown(
        """
        - **Basket intensity – top categories (normalized by category):** **Top** and **Set** exceed the average during some periods, suggesting stronger inclusion in customer baskets, while **Bottom** tends to remain below average.
        - **Basket intensity – top categories (normalized by SKU):** differences become more moderate, which suggests that basket presence is relatively stable once SKU-level normalization is applied.
        - **Basket intensity – top SKU (normalized by SKU):** some products show temporary spikes in intensity, meaning that they become especially prominent during specific periods, but most return to the average afterward.
        - **Monthly mean quantity – top SKU:** some SKUs experience temporary demand surges, while others remain more stable across time.
        """
    )

    # ========================================================
    # 3. FEATURE ENGINEERING
    # ========================================================
    st.header("3. Feature Engineering")
    st.subheader("Leakage-aware feature engineering design")

    st.markdown(
        """
        This feature engineering framework was designed with a strong **anti-leakage discipline**.
        In a time series setting, leakage can severely bias performance estimates by introducing future information into the training process.
        To prevent this, every transformation must be computed using only information that is available **at or before the prediction date**.

        **Leakage prevention principles applied in the project**

        - All splits must remain **chronological**: training data always precedes validation data, and validation always precedes test data.
        - Rolling statistics such as moving averages, rolling standard deviations, and rolling elasticity indicators must be computed using **past windows only**.
        - Lag features such as **lag_1**, **lag_7**, **lag_14**, and **lag_28** are valid only if they are shifted correctly so they never reuse the current target value.
        - Price elasticity features must be constructed from historical observations only. At prediction time, they must not depend on future quantity or future price realizations.
        - Any imputation strategy for rolling features should also be designed without forward-looking information.
        - Encoding, scaling, and preprocessing steps must be fitted **inside the training folds only**, typically through a pipeline.
        - The target variable must never be used directly or indirectly to create features at the same timestamp.

        **Why this matters**

        In this project, the goal is not only to maximize predictive performance but also to preserve methodological validity.
        A leakage-aware design ensures that model evaluation remains realistic and that the measured performance is reproducible in production conditions.
        """
    )

    st.markdown("### Feature groups used in the project")
    feature_block("Categorical variables – Feature set 1", cat_features_f1)
    feature_block("Categorical variables – Feature set 2", cat_features_f2)
    feature_block("Numeric variables – Feature set 1", num_features_f1)
    feature_block("Numeric variables – Feature set 2", num_features_f2)
    feature_block("Variables passed without transformation", pas_transf)
    feature_block("Cyclic variables – Feature set 1", cyclic_features_f1)
    feature_block("Cyclic variables – Feature set 2", cyclic_features_f2)

    st.markdown("### Column coverage verification")
    st.code(
        '''# Verification that all variables are assigned to a preprocessing group\n\ncol_f1 = set(cat_features_f1) | set(num_features_f1) | set(cyclic_features_f1) | set(pas_transf)\ncol_f2 = set(cat_features_f2) | set(num_features_f2) | set(cyclic_features_f2) | set(pas_transf)\n\ndf_col_f1 = set(df_feature_1.columns)\ndf_col_f2 = set(df_feature_2.columns)\n\nnon_preproc_f1 = df_col_f1 - set(col_f1)\nnon_preproc_f2 = df_col_f2 - set(col_f2)\n\ndisplay(non_preproc_f1)\ndisplay(non_preproc_f2)''',
        language="python",
    )
if __name__ == "__main__":
    main()
