import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ====================
# Page configuration
# ====================
st.set_page_config(page_title="Retail ML Dashboard", layout="wide")

API_URL = "http://127.0.0.1:8000/api"
API_URL_HEALTH = "http://127.0.0.1:8000/api/health"

# Demo CSV files used when the API is offline or no file is uploaded.
BASE_DIR = Path(__file__).resolve().parents[2]

LOCAL_PREDICTION_CSV_PATH = BASE_DIR / "st_demo" / "predictions_globales.csv"
LOCAL_KPI_CSV_PATH = BASE_DIR / "st_demo" / "kpi_3_last_month_quantity_Daily.csv"


# ====================
# Session state
# ====================
if "df" not in st.session_state:
    st.session_state.df = None

if "df_pred" not in st.session_state:
    st.session_state.df_pred = None

if "df_kpi" not in st.session_state:
    st.session_state.df_kpi = None

if "api_available" not in st.session_state:
    st.session_state.api_available = None

if "prediction_source" not in st.session_state:
    st.session_state.prediction_source = None

if "kpi_source" not in st.session_state:
    st.session_state.kpi_source = None


# ====================
# Functions
# ====================
def check_api_health(api_health_url: str, timeout: int = 5) -> tuple[bool, str]:
    """Check whether the API health endpoint is reachable.

    Args:
        api_health_url: API health endpoint URL.
        timeout: Request timeout in seconds.

    Returns:
        A tuple with the API status and a message.
    """
    try:
        response = requests.get(api_health_url, timeout=timeout)
        if response.status_code == 200:
            try:
                return True, f"API available: {response.json()}"
            except Exception:
                return True, "API available."
        return False, f"API unavailable: status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Unable to reach the API: {e}"


def call_api(
    df: pd.DataFrame,
    api_url: str,
    endpoint: str,
    timeout: int = 120
) -> pd.DataFrame:
    """Send a CSV file to the API and return the response as a DataFrame.

    Args:
        df: Input DataFrame to send.
        api_url: Base API URL.
        endpoint: API endpoint path.
        timeout: Request timeout in seconds.

    Returns:
        API response converted to a DataFrame.

    Raises:
        RuntimeError: If the API returns a non-200 status code.
        ValueError: If the response format is invalid.
    """
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"

    # Convert the input DataFrame to an in-memory CSV payload.
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    files = {
        "file": ("data.csv", csv_bytes, "text/csv")
    }

    try:
        response = requests.post(url, files=files, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"API call failed: {e}")

    if response.status_code != 200:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text
        raise RuntimeError(f"API error {response.status_code}: {error_detail}")

    try:
        result = response.json()
    except Exception:
        raise ValueError("Invalid JSON response from API.")

    if not isinstance(result, list):
        raise ValueError("API response is not a list of records.")

    df_result = pd.DataFrame(result)

    if df_result.empty:
        raise ValueError("API returned an empty result.")

    # Normalize and sort temporal data if available.
    if "DATE" in df_result.columns:
        df_result["DATE"] = pd.to_datetime(df_result["DATE"], errors="coerce")
        df_result = df_result.dropna(subset=["DATE"])
        sort_cols = [col for col in ["SKU", "DATE"] if col in df_result.columns]
        if sort_cols:
            df_result = df_result.sort_values(sort_cols).reset_index(drop=True)

    return df_result


def load_csv_file(csv_path: str, file_label: str) -> pd.DataFrame:
    """Load a CSV file using flexible delimiter detection.

    Args:
        csv_path: CSV file path.
        file_label: Human-readable file label for error messages.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    file_path = Path(csv_path)

    if not file_path.exists():
        raise FileNotFoundError(f"The {file_label} file does not exist: {csv_path}")

    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(file_path, delimiter=";")
        except Exception:
            df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"The {file_label} file is empty.")

    return df


def load_local_prediction_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate the local prediction CSV file.

    Args:
        csv_path: Prediction CSV path.

    Returns:
        Cleaned prediction DataFrame.
    """
    df_pred = load_csv_file(csv_path, "prediction")

    required_cols = ["DATE", "prediction_quantity"]
    missing_cols = [col for col in required_cols if col not in df_pred.columns]
    if missing_cols:
        raise ValueError(
            f"The prediction CSV must contain at least {required_cols}. "
            f"Missing columns: {missing_cols}"
        )

    df_pred["DATE"] = pd.to_datetime(df_pred["DATE"], errors="coerce")
    df_pred["prediction_quantity"] = pd.to_numeric(
        df_pred["prediction_quantity"], errors="coerce"
    )
    df_pred = df_pred.dropna(subset=["DATE", "prediction_quantity"])

    return df_pred.sort_values("DATE").reset_index(drop=True)


def load_local_kpi_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate the local KPI CSV file.

    Args:
        csv_path: KPI CSV path.

    Returns:
        Cleaned KPI DataFrame.
    """
    df_kpi = load_csv_file(csv_path, "KPI")

    if "DATE" not in df_kpi.columns:
        raise ValueError("The KPI CSV must contain a 'DATE' column.")

    df_kpi["DATE"] = pd.to_datetime(df_kpi["DATE"], errors="coerce")
    df_kpi = df_kpi.dropna(subset=["DATE"])

    return df_kpi.sort_values("DATE").reset_index(drop=True)


def normalize_uploaded_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the uploaded historical dataset.

    Args:
        df: Uploaded raw DataFrame.

    Returns:
        Cleaned historical DataFrame.
    """
    df = df.copy()

    required_cols = ["DATE", "SKU", "Style", "Category", "price", "quantity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required upload columns: {missing_cols}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    sort_cols = [col for col in ["SKU", "DATE"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def get_prediction_data(df_hist: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """Get prediction data from the API or fallback local CSV.

    Args:
        df_hist: Historical input DataFrame.

    Returns:
        Prediction DataFrame and source label.
    """
    api_ok, _ = check_api_health(API_URL_HEALTH)

    if api_ok and df_hist is not None and not df_hist.empty:
        try:
            df_pred = call_api(df_hist, API_URL, "/predict-file")
            return df_pred, "API"
        except Exception:
            pass

    df_pred = load_local_prediction_csv(LOCAL_PREDICTION_CSV_PATH)
    return df_pred, "LOCAL_CSV"


def get_kpi_data(df_hist: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """Get KPI data from the API or fallback local CSV.

    Args:
        df_hist: Historical input DataFrame.

    Returns:
        KPI DataFrame and source label.
    """
    api_ok, _ = check_api_health(API_URL_HEALTH)

    if api_ok and df_hist is not None and not df_hist.empty:
        try:
            df_kpi = call_api(df_hist, API_URL, "/features-file")
            return df_kpi, "API"
        except Exception:
            pass

    df_kpi = load_local_kpi_csv(LOCAL_KPI_CSV_PATH)
    return df_kpi, "LOCAL_CSV"


def aggregate_kpi(df: pd.DataFrame, type_prdt: str, freq: str) -> pd.DataFrame:
    """Aggregate KPI data by product level and time frequency.

    Args:
        df: KPI DataFrame.
        type_prdt: Aggregation level.
        freq: Resampling frequency.

    Returns:
        Aggregated KPI DataFrame.
    """
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    if type_prdt not in df.columns:
        raise ValueError(f"Missing grouping column: {type_prdt}")

    agg_dict = {
        "quantity": "sum",
        "elasticite_rolling_SKU": "mean",
        "elasticite_rolling_Style": "mean",
        "elasticite_rolling_Category": "mean",
        "var_elasticite_rolling_SKU": "mean",
        "var_elasticite_rolling_Style": "mean",
        "var_elasticite_rolling_Category": "mean",
        "basket_intensity_sku_norm": "mean",
        "basket_intensity_style_norm": "mean",
        "basket_intensity_cat_norm": "mean",
    }

    existing_agg = {col: rule for col, rule in agg_dict.items() if col in df.columns}

    if not existing_agg:
        raise ValueError(
            "No KPI columns are available for aggregation. "
            f"Available columns: {df.columns.tolist()}"
        )

    out = []

    # Aggregate each product group independently.
    for product_value, g in df.groupby(type_prdt, observed=True):
        if g.empty:
            continue

        g = g.sort_values("DATE").set_index("DATE")
        g_resampled = g.resample(freq).agg(existing_agg)

        if g_resampled.empty:
            continue

        g_resampled[type_prdt] = product_value

        if "quantity" in g_resampled.columns:
            g_resampled["quantity"] = g_resampled["quantity"].fillna(0)

        out.append(g_resampled.reset_index())

    if not out:
        raise ValueError("No data available after aggregation.")

    return pd.concat(out, ignore_index=True)


def rank_quantity(df: pd.DataFrame, type_prdt: str, freq: str, top_n: int = 5):
    """Rank entities by total quantity.

    Args:
        df: KPI DataFrame.
        type_prdt: Aggregation level.
        freq: Resampling frequency.
        top_n: Number of top and bottom entities.

    Returns:
        Top and bottom ranked DataFrames.
    """
    df_agg = aggregate_kpi(df, type_prdt, freq)

    if "quantity" not in df_agg.columns:
        raise ValueError("The 'quantity' column is missing after aggregation.")

    df_rank = (
        df_agg.groupby(type_prdt, observed=True, as_index=False)["quantity"]
        .sum()
        .sort_values("quantity", ascending=False)
    )

    top_5 = df_rank.head(top_n).copy()
    bottom_5 = df_rank.tail(top_n).sort_values("quantity", ascending=True).copy()

    return top_5, bottom_5


def prepare_rank_timeseries(
    df: pd.DataFrame,
    type_prdt: str,
    freq: str,
    rank_mode: str = "Highest",
    top_n: int = 5
):
    """Prepare time series data for top or bottom ranked entities.

    Args:
        df: KPI DataFrame.
        type_prdt: Aggregation level.
        freq: Resampling frequency.
        rank_mode: Ranking mode.
        top_n: Number of selected entities.

    Returns:
        Plot DataFrame, top table, and bottom table.
    """
    df_agg = aggregate_kpi(df, type_prdt, freq)
    top_5, bottom_5 = rank_quantity(df, type_prdt, freq, top_n=top_n)

    selected_items = (
        top_5[type_prdt].tolist()
        if rank_mode == "Highest"
        else bottom_5[type_prdt].tolist()
    )

    df_plot = df_agg[df_agg[type_prdt].isin(selected_items)].copy()

    if df_plot.empty:
        raise ValueError("No data to plot after filtering.")

    return df_plot, top_5, bottom_5


def plot_line(df, x_col, y_col, multi_lane=None, title="Dynamic Time Series"):
    """Create a Plotly line chart.

    Args:
        df: Input DataFrame.
        x_col: X-axis column.
        y_col: Y-axis column.
        multi_lane: Optional grouping column.
        title: Chart title.

    Returns:
        Plotly figure.
    """
    if multi_lane and multi_lane in df.columns and df[multi_lane].nunique() > 1:
        fig = px.line(df, x=x_col, y=y_col, color=multi_lane, markers=True)
    else:
        fig = px.line(df, x=x_col, y=y_col, markers=True)

    fig.update_layout(
        template="plotly_dark",
        title=title,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        title_font=dict(size=20, color="#38bdf8"),
        legend_title_font=dict(color="#38bdf8"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
    return fig


def build_daily_aggregated_frame(
    df: pd.DataFrame,
    value_col: str,
    include_levels: list[str]
) -> pd.DataFrame:
    """Build a daily aggregated DataFrame for multiple hierarchy levels.

    Args:
        df: Input DataFrame.
        value_col: Value column to aggregate.
        include_levels: Additional hierarchy levels.

    Returns:
        Aggregated DataFrame with level and entity columns.
    """
    df = df.copy()

    if "DATE" not in df.columns:
        raise ValueError("The 'DATE' column is missing.")

    if value_col not in df.columns:
        raise ValueError(f"The '{value_col}' column is missing.")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["DATE", value_col])

    frames = []

    # Create a global daily aggregation.
    general_daily = (
        df.groupby("DATE", as_index=False)[value_col]
        .sum()
        .assign(level="GENERAL", entity="GENERAL")
        .rename(columns={value_col: "value"})
    )
    frames.append(general_daily)

    # Create a daily aggregation for each available hierarchy level.
    for level in include_levels:
        if level in df.columns:
            tmp = (
                df.groupby(["DATE", level], observed=True, as_index=False)[value_col]
                .sum()
                .rename(columns={level: "entity", value_col: "value"})
            )
            tmp["level"] = level
            frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)
    out = out[["DATE", "level", "entity", "value"]].sort_values(
        ["level", "entity", "DATE"]
    )
    return out.reset_index(drop=True)


def compute_forecast_confidence_band(
    hist_daily: pd.DataFrame | None,
    pred_daily: pd.DataFrame,
    z_score: float = 1.96,
    fallback_std_ratio: float = 0.15
) -> pd.DataFrame:
    """Compute confidence intervals around forecast values.

    Args:
        hist_daily: Historical daily data.
        pred_daily: Forecast daily data.
        z_score: Z-score used for the interval.
        fallback_std_ratio: Fallback ratio when no historical std is available.

    Returns:
        Forecast DataFrame with lower and upper bounds.
    """
    pred_daily = pred_daily.copy()
    pred_daily["DATE"] = pd.to_datetime(pred_daily["DATE"], errors="coerce")

    if hist_daily is not None and not hist_daily.empty:
        hist_daily = hist_daily.copy()
        hist_daily["DATE"] = pd.to_datetime(hist_daily["DATE"], errors="coerce")
    else:
        hist_daily = None

    out = []

    # Compute a confidence band for each level/entity pair.
    for (level_value, entity_value), pred_grp in pred_daily.groupby(
        ["level", "entity"], observed=True
    ):
        pred_grp = pred_grp.sort_values("DATE").copy()

        hist_std = np.nan

        if hist_daily is not None:
            hist_grp = hist_daily[
                (hist_daily["level"] == level_value) &
                (hist_daily["entity"] == entity_value)
            ].sort_values("DATE").copy()

            if not hist_grp.empty:
                max_hist_date = hist_grp["DATE"].max()
                hist_last_30 = hist_grp[
                    hist_grp["DATE"] >= (max_hist_date - pd.Timedelta(days=30))
                ]
                hist_std = hist_last_30["value"].std()

        pred_mean = pred_grp["value"].mean()

        if pd.isna(hist_std) or hist_std == 0:
            if pd.isna(pred_mean):
                hist_std = 1.0
            else:
                hist_std = max(abs(pred_mean) * fallback_std_ratio, 1.0)

        margin = z_score * hist_std
        pred_grp["pred_lower"] = np.maximum(pred_grp["value"] - margin, 0)
        pred_grp["pred_upper"] = pred_grp["value"] + margin

        out.append(pred_grp)

    if not out:
        raise ValueError("Unable to compute confidence intervals.")

    return pd.concat(out, ignore_index=True)


def plot_forecast_with_confidence(
    hist_daily: pd.DataFrame | None,
    pred_daily: pd.DataFrame,
    selected_level: str,
    selected_entity: str
) -> go.Figure:
    """Plot recent history and forecast with confidence intervals.

    Args:
        hist_daily: Historical daily data.
        pred_daily: Forecast daily data.
        selected_level: Selected hierarchy level.
        selected_entity: Selected entity.

    Returns:
        Plotly figure.
    """
    if hist_daily is not None and not hist_daily.empty:
        hist_sel = hist_daily[
            (hist_daily["level"] == selected_level) &
            (hist_daily["entity"].astype(str) == str(selected_entity))
        ].copy()
        hist_sel = hist_sel.sort_values("DATE")
    else:
        hist_sel = pd.DataFrame(columns=["DATE", "value"])

    pred_sel = pred_daily[
        (pred_daily["level"] == selected_level) &
        (pred_daily["entity"].astype(str) == str(selected_entity))
    ].copy()
    pred_sel = pred_sel.sort_values("DATE")

    if hist_sel.empty and pred_sel.empty:
        raise ValueError(f"No data available for {selected_level} - {selected_entity}")

    # Keep only the last 30 days of historical data.
    if not hist_sel.empty:
        max_hist_date = hist_sel["DATE"].max()
        hist_sel = hist_sel[hist_sel["DATE"] >= (max_hist_date - pd.Timedelta(days=30))]

    fig = go.Figure()

    if not hist_sel.empty:
        fig.add_trace(
            go.Scatter(
                x=hist_sel["DATE"],
                y=hist_sel["value"],
                mode="lines+markers",
                name="History (1 month)"
            )
        )

    if not pred_sel.empty:
        fig.add_trace(
            go.Scatter(
                x=pred_sel["DATE"],
                y=pred_sel["pred_upper"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pred_sel["DATE"],
                y=pred_sel["pred_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="95% CI"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pred_sel["DATE"],
                y=pred_sel["value"],
                mode="lines+markers",
                name="Forecast"
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=f"Daily history (1 month) + forecast - {selected_level}: {selected_entity}",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        title_font=dict(size=20, color="#38bdf8"),
        legend_title_font=dict(color="#38bdf8"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b", title="Quantity")

    return fig


# ====================
# Navigation
# ====================
page = st.sidebar.radio("Navigation", ["Upload", "Charts"])
st.title("Retail ML Dashboard")


# ====================
# Upload page
# ====================
if page == "Upload":
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    st.info(
        "Demo mode: without an uploaded file, the dashboard uses local "
        "prediction and KPI CSV files."
    )

    if uploaded_file is not None:
        try:
            try:
                df = pd.read_csv(uploaded_file, delimiter=";")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)

            st.session_state.df = df
            st.session_state.df_pred = None
            st.session_state.df_kpi = None
            st.session_state.prediction_source = None
            st.session_state.kpi_source = None

            st.success("File loaded successfully.")
            st.dataframe(df.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Error while reading the file: {e}")


# ====================
# Charts page
# ====================
elif page == "Charts":
    st.header("Analysis and Visualization")

    df_uploaded = st.session_state.df
    df_hist = None

    if df_uploaded is not None:
        try:
            df_hist = normalize_uploaded_history(df_uploaded)
        except Exception as e:
            st.warning(f"Upload detected but unusable as history: {e}")
            df_hist = None
    else:
        st.info("No file uploaded. Demo mode with local CSV files.")

    st.subheader("API Services")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Check API", use_container_width=True):
            api_ok, msg = check_api_health(API_URL_HEALTH)
            st.session_state.api_available = api_ok
            if api_ok:
                st.success(msg)
            else:
                st.warning(
                    f"{msg}\n\nDemo mode enabled: the dashboard will use local CSV files."
                )

    with col2:
        if st.button("Generate Forecasts", use_container_width=True):
            try:
                df_pred, source = get_prediction_data(df_hist)
                st.session_state.df_pred = df_pred
                st.session_state.prediction_source = source

                if source == "API":
                    st.success("Forecasts generated successfully from the API.")
                else:
                    st.warning("Forecasts loaded from the local demo CSV file.")

            except Exception as e:
                st.error(f"Forecast error: {e}")

    with col3:
        if st.button("Compute Historical KPI", use_container_width=True):
            try:
                df_kpi, source = get_kpi_data(df_hist)
                st.session_state.df_kpi = df_kpi
                st.session_state.kpi_source = source

                if source == "API":
                    st.success("Historical KPI computed successfully from the API.")
                else:
                    st.warning("KPI loaded from the local demo CSV file.")

            except Exception as e:
                st.error(f"KPI error: {e}")

    # Auto-load predictions if not already loaded.
    if st.session_state.df_pred is None:
        try:
            df_pred, source = get_prediction_data(df_hist)
            st.session_state.df_pred = df_pred
            st.session_state.prediction_source = source
        except Exception as e:
            st.warning(f"Forecasts unavailable: {e}")

    # Auto-load KPI if not already loaded.
    if st.session_state.df_kpi is None:
        try:
            df_kpi, source = get_kpi_data(df_hist)
            st.session_state.df_kpi = df_kpi
            st.session_state.kpi_source = source
        except Exception as e:
            st.warning(f"KPI unavailable: {e}")

    # ====================
    # Forecast section
    # ====================
    if st.session_state.df_pred is not None:
        df_pred = st.session_state.df_pred.copy()

        st.subheader("Historical Data + Forecast Visualization")

        if st.session_state.prediction_source is not None:
            st.caption(f"Forecast source: {st.session_state.prediction_source}")

        try:
            hist_daily = None
            if df_hist is not None and not df_hist.empty:
                hist_daily = build_daily_aggregated_frame(
                    df=df_hist,
                    value_col="quantity",
                    include_levels=["Category", "Style", "SKU"]
                )

            pred_daily = build_daily_aggregated_frame(
                df=df_pred,
                value_col="prediction_quantity",
                include_levels=["Category", "Style", "SKU"]
            )

            pred_daily = compute_forecast_confidence_band(hist_daily, pred_daily)

            level_options = [
                lvl for lvl in ["GENERAL", "Category", "Style", "SKU"]
                if lvl in pred_daily["level"].unique()
            ]

            if not level_options:
                raise ValueError("No level available in forecast data.")

            c_level, c_entity = st.columns(2)

            with c_level:
                selected_level = st.selectbox(
                    "Visualization level",
                    level_options,
                    index=0,
                    key="forecast_level"
                )

            entity_list = sorted(
                pred_daily[pred_daily["level"] == selected_level]["entity"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not entity_list:
                raise ValueError(f"No entity available for level {selected_level}.")

            default_entity_index = 0
            if selected_level == "GENERAL" and "GENERAL" in entity_list:
                default_entity_index = entity_list.index("GENERAL")

            with c_entity:
                selected_entity = st.selectbox(
                    "Selected value",
                    entity_list,
                    index=default_entity_index,
                    key="forecast_entity"
                )

            fig_forecast = plot_forecast_with_confidence(
                hist_daily=hist_daily,
                pred_daily=pred_daily,
                selected_level=selected_level,
                selected_entity=selected_entity
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            with st.expander("View aggregated forecast data"):
                pred_view = pred_daily[
                    (pred_daily["level"] == selected_level) &
                    (pred_daily["entity"].astype(str) == str(selected_entity))
                ].copy()
                st.dataframe(pred_view, use_container_width=True)

        except Exception as e:
            st.error(f"Forecast visualization error: {e}")

    # ====================
    # Historical KPI section
    # ====================
    if st.session_state.df_kpi is not None:
        df_kpi = st.session_state.df_kpi.copy()

        st.subheader("Historical KPI")

        if st.session_state.kpi_source is not None:
            st.caption(f"KPI source: {st.session_state.kpi_source}")

        level_agg = [lvl for lvl in ["Category", "Style", "SKU"] if lvl in df_kpi.columns]
        if not level_agg:
            st.warning("No KPI grouping column available among Category, Style, SKU.")
        else:
            kpi_list = [
                "quantity",
                "rolling elasticity variance",
                "rolling elasticity",
                "basket intensity"
            ]
            freq_list = ["Daily", "Weekly", "Monthly"]

            opt_agg = {
                "Category": {
                    "quantity": "quantity",
                    "rolling elasticity variance": "var_elasticite_rolling_Category",
                    "rolling elasticity": "elasticite_rolling_Category",
                    "basket intensity": "basket_intensity_cat_norm"
                },
                "Style": {
                    "quantity": "quantity",
                    "rolling elasticity variance": "var_elasticite_rolling_Style",
                    "rolling elasticity": "elasticite_rolling_Style",
                    "basket intensity": "basket_intensity_style_norm"
                },
                "SKU": {
                    "quantity": "quantity",
                    "rolling elasticity variance": "var_elasticite_rolling_SKU",
                    "rolling elasticity": "elasticite_rolling_SKU",
                    "basket intensity": "basket_intensity_sku_norm"
                }
            }

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                select_level = st.selectbox("Level", level_agg, key="kpi_level")
            with c2:
                select_freq = st.selectbox("Frequency", freq_list, key="kpi_freq")

            # Top/Bottom filter is available only for Style and SKU.
            if select_level in ["Style", "SKU"]:
                with c3:
                    select_rank = st.selectbox(
                        "Quantity filter",
                        ["Highest", "Smallest"],
                        key="kpi_rank"
                    )
            else:
                select_rank = "Highest"
                with c3:
                    st.selectbox(
                        "Quantity filter",
                        ["Highest"],
                        index=0,
                        key="kpi_rank_disabled",
                        disabled=True
                    )

            available_kpi = []
            for kpi_label, colname in opt_agg[select_level].items():
                if colname in df_kpi.columns:
                    available_kpi.append(kpi_label)

            with c4:
                if available_kpi:
                    select_kpi = st.selectbox("KPI", available_kpi, key="kpi_metric")
                else:
                    select_kpi = None

            if select_kpi is None:
                st.warning(f"No KPI available for level {select_level}.")
            else:
                opt_freq = {
                    "Daily": "D",
                    "Weekly": "W",
                    "Monthly": "M"
                }

                freq_code = opt_freq[select_freq]
                metric_col = opt_agg[select_level][select_kpi]

                try:
                    df_plot, top_5, bottom_5 = prepare_rank_timeseries(
                        df=df_kpi,
                        type_prdt=select_level,
                        freq=freq_code,
                        rank_mode=select_rank,
                        top_n=5
                    )

                    if select_level in ["Style", "SKU"]:
                        col_top, col_bottom = st.columns(2)

                        with col_top:
                            st.markdown("#### Top 5 quantities")
                            st.dataframe(top_5, use_container_width=True)

                        with col_bottom:
                            st.markdown("#### Bottom 5 quantities")
                            st.dataframe(bottom_5, use_container_width=True)

                    if metric_col not in df_plot.columns:
                        raise ValueError(
                            f"The KPI column '{metric_col}' is missing from the loaded KPI data."
                        )

                    fig = plot_line(
                        df=df_plot,
                        x_col="DATE",
                        y_col=metric_col,
                        multi_lane=select_level,
                        title=f"{select_rank} 5 {select_level} - {select_kpi} ({select_freq})"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"KPI error: {e}")