import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io


# ====================
# Configuration page
# ====================
st.set_page_config(page_title="Dashboard Retail ML", layout="wide")

API_URL = "http://127.0.0.1:8000/api"
API_URL_HEALTH = "http://127.0.0.1:8000/api/health"


# ====================
# Session state
# ====================
if "df" not in st.session_state:
    st.session_state.df = None

if "df_pred" not in st.session_state:
    st.session_state.df_pred = None

if "df_kpi" not in st.session_state:
    st.session_state.df_kpi = None


# ====================
# Functions
# ====================
def call_api(
    df: pd.DataFrame,
    api_url: str,
    endpoint: str,
    timeout: int = 120
) -> pd.DataFrame:
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"

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

    if "DATE" in df_result.columns:
        df_result["DATE"] = pd.to_datetime(df_result["DATE"], errors="coerce")
        df_result = df_result.dropna(subset=["DATE"])
        sort_cols = [col for col in ["SKU", "DATE"] if col in df_result.columns]
        if sort_cols:
            df_result = df_result.sort_values(sort_cols).reset_index(drop=True)

    return df_result


def aggregate_kpi(df, type_prdt, freq):
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    if type_prdt not in df.columns:
        raise ValueError(f"Colonne de regroupement absente : {type_prdt}")

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
            "Aucune colonne KPI disponible pour l’agrégation. "
            f"Colonnes présentes : {df.columns.tolist()}"
        )

    out = []

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
        raise ValueError("Aucune donnée disponible après agrégation.")

    return pd.concat(out, ignore_index=True)


def rank_quantity(df, type_prdt, freq, top_n=5):
    df_agg = aggregate_kpi(df, type_prdt, freq)

    if "quantity" not in df_agg.columns:
        raise ValueError("La colonne 'quantity' est absente après agrégation.")

    df_rank = (
        df_agg.groupby(type_prdt, observed=True, as_index=False)["quantity"]
        .sum()
        .sort_values("quantity", ascending=False)
    )

    top_5 = df_rank.head(top_n).copy()
    bottom_5 = df_rank.tail(top_n).sort_values("quantity", ascending=True).copy()

    return top_5, bottom_5


def prepare_rank_timeseries(df, type_prdt, freq, rank_mode="Highest", top_n=5):
    df_agg = aggregate_kpi(df, type_prdt, freq)
    top_5, bottom_5 = rank_quantity(df, type_prdt, freq, top_n=top_n)

    selected_items = top_5[type_prdt].tolist() if rank_mode == "Highest" else bottom_5[type_prdt].tolist()
    df_plot = df_agg[df_agg[type_prdt].isin(selected_items)].copy()

    if df_plot.empty:
        raise ValueError("Aucune donnée à tracer après filtrage.")

    return df_plot, top_5, bottom_5


def plot_line(df, x_col, y_col, multi_lane=None, title="Dynamic Time Series"):
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


# ====================
# Navigation
# ====================
page = st.sidebar.radio("Navigation", ["Upload", "Graphiques"])

st.title("Dashboard Retail ML")


# ====================
# Upload page
# ====================
if page == "Upload":
    st.header("Import des données")
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, delimiter=";")
            st.session_state.df = df
            st.session_state.df_pred = None
            st.session_state.df_kpi = None

            st.success("Fichier chargé avec succès.")
            st.dataframe(df.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")


# ====================
# Charts page
# ====================
elif page == "Graphiques":
    st.header("Analyse et visualisation")

    df = st.session_state.df

    if df is None:
        st.warning("Veuillez d'abord charger un fichier dans la page Upload.")
    else:
        required_cols = ["DATE", "SKU", "Style", "Category", "price"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Colonnes obligatoires manquantes : {missing_cols}")
        else:
            df_hist = df.copy()
            df_hist["DATE"] = pd.to_datetime(df_hist["DATE"], errors="coerce")
            df_hist = df_hist.dropna(subset=["DATE"])
            df_hist = df_hist.sort_values(["SKU", "DATE"]).reset_index(drop=True)

            st.subheader("1. Historique préparé")
            st.dataframe(df_hist.head(), use_container_width=True)

            st.subheader("2. Services API")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Vérifier API", use_container_width=True):
                    try:
                        response = requests.get(API_URL_HEALTH, timeout=10)
                        if response.status_code == 200:
                            st.success(f"API disponible : {response.json()}")
                        else:
                            st.error(f"API indisponible : {response.status_code}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Impossible de joindre l'API : {e}")

            with col2:
                if st.button("Générer prévisions", use_container_width=True):
                    try:
                        df_pred = call_api(df_hist, API_URL, "/predict-file")
                        st.session_state.df_pred = df_pred
                        st.success("Prévisions générées avec succès.")
                    except Exception as e:
                        st.error(str(e))

            with col3:
                if st.button("Calculer KPI historiques", use_container_width=True):
                    try:
                        df_kpi = call_api(df_hist, API_URL, "/features-file")
                        st.session_state.df_kpi = df_kpi
                        st.success("KPI historiques calculés avec succès.")
                    except Exception as e:
                        st.error(str(e))

            if st.session_state.df_pred is not None:
                df_pred = st.session_state.df_pred.copy()

                st.subheader("3. Visualisation historique + prévisions")

                cat_list = sorted(df_hist["Category"].dropna().unique().tolist())
                selected_cat = st.selectbox("Choisir une Category", cat_list, key="forecast_category")

                hist_cat = df_hist[df_hist["Category"] == selected_cat].copy()
                pred_cat = df_pred[df_pred["Category"] == selected_cat].copy()

                hist_plot = hist_cat[["DATE", "quantity"]].copy()
                hist_plot["type"] = "Historique"
                hist_plot = hist_plot.rename(columns={"quantity": "value"})

                pred_plot = pred_cat[["DATE", "prediction_quantity"]].copy()
                pred_plot["type"] = "Prévision"
                pred_plot = pred_plot.rename(columns={"prediction_quantity": "value"})

                df_all = pd.concat([hist_plot, pred_plot], ignore_index=True)
                df_all = df_all.sort_values("DATE").reset_index(drop=True)

                fig = px.line(
                    df_all,
                    x="DATE",
                    y="value",
                    color="type",
                    markers=True,
                    title=f"Historique et prévisions - Category {selected_cat}"
                )
                st.plotly_chart(fig, use_container_width=True)

            if st.session_state.df_kpi is not None:
                df_kpi = st.session_state.df_kpi.copy()

                st.subheader("4. KPI historiques")

                level_agg = ["Style", "SKU"]
                kpi_list = ["quantity", "variance elasticite rolling", "elasticite rolling", "basket intensity"]
                freq_list = ["Daily", "Weekly", "Monthly"]
                rank_options = ["Highest", "Smallest"]

                opt_agg = {
                    "Style": {
                        "quantity": "quantity",
                        "variance elasticite rolling": "var_elasticite_rolling_Style",
                        "elasticite rolling": "elasticite_rolling_Style",
                        "basket intensity": "basket_intensity_style_norm"
                    },
                    "SKU": {
                        "quantity": "quantity",
                        "variance elasticite rolling": "var_elasticite_rolling_SKU",
                        "elasticite rolling": "elasticite_rolling_SKU",
                        "basket intensity": "basket_intensity_sku_norm"
                    }
                }

                opt_freq = {
                    "Daily": "D",
                    "Weekly": "W",
                    "Monthly": "M"
                }

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    select_level = st.selectbox("Niveau", level_agg)
                with c2:
                    select_freq = st.selectbox("Fréquence", freq_list)
                with c3:
                    select_rank = st.selectbox("Classement", rank_options)
                with c4:
                    select_kpi = st.selectbox("KPI", kpi_list)

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

                    col_top, col_bottom = st.columns(2)

                    with col_top:
                        st.markdown("#### Top 5 quantités")
                        st.dataframe(top_5, use_container_width=True)

                    with col_bottom:
                        st.markdown("#### Bottom 5 quantités")
                        st.dataframe(bottom_5, use_container_width=True)

                    fig = plot_line(
                        df=df_plot,
                        x_col="DATE",
                        y_col=metric_col,
                        multi_lane=select_level,
                        title=f"{select_rank} 5 {select_level} - {select_kpi} ({select_freq})"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur KPI : {e}")