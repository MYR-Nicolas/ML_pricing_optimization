import numpy as np
import pandas as pd


def elast_rolling(
    df,
    group_cols=("SKU", "Style", "Category"),
    date_col="DATE",
    qty_col="quantity",
    price_col="price",
    window=14,
    min_periods=5,
    clip_abs=10,
):
    """
    Compute rolling price elasticity features by group.

    Args:
        df: Input DataFrame.
        group_cols (tuple): Grouping columns.
        date_col (str): Date column.
        qty_col (str): Quantity column.
        price_col (str): Price column.
        window (int): Rolling window size.
        min_periods (int): Minimum observations required.
        clip_abs (float): Max absolute elasticity allowed.

    Returns:
        pd.DataFrame: DataFrame enriched with elasticity features.
    """
    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")
    df_out = df_out.dropna(subset=[date_col])

    required_cols = [date_col, qty_col, price_col, *group_cols]
    missing_cols = [col for col in required_cols if col not in df_out.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in elast_rolling: {missing_cols}")

    df_out = df_out.sort_values(list(group_cols) + [date_col])

    df_out["log_qty"] = np.log1p(df_out[qty_col])
    df_out["log_price"] = np.where(df_out[price_col] > 0, np.log(df_out[price_col]), np.nan)

    for col in group_cols:
        dlog_qty_col = f"dlog_qty_{col}"
        dlog_price_col = f"dlog_price_{col}"
        elas_loc_col = f"elasticite_locale_{col}"
        elas_roll_col = f"elasticite_rolling_{col}"

        df_out = df_out.sort_values([col, date_col])

        dlog_qty_raw = df_out.groupby(col, observed=True)["log_qty"].diff()
        dlog_price_raw = df_out.groupby(col, observed=True)["log_price"].diff()

        # Historical usable deltas
        df_out[dlog_qty_col] = dlog_qty_raw.groupby(df_out[col], observed=True).shift(1)
        df_out[dlog_price_col] = dlog_price_raw.groupby(df_out[col], observed=True).shift(1)

        # Local elasticity
        df_out[elas_loc_col] = dlog_qty_raw / dlog_price_raw
        df_out[elas_loc_col] = df_out[elas_loc_col].replace([np.inf, -np.inf], np.nan)
        df_out[elas_loc_col] = df_out[elas_loc_col].mask(df_out[elas_loc_col].abs() > clip_abs)

        # Historical rolling mean
        df_out[elas_roll_col] = (
            df_out.groupby(col, observed=True)[elas_loc_col]
                  .transform(lambda x: x.rolling(window=window, min_periods=min_periods).mean().shift(1))
        )

    df_out = df_out.drop(columns=["log_qty"], errors="ignore")
    return df_out


def elast_variance(
    df,
    group_cols=("SKU", "Style", "Category"),
    date_col="DATE",
    window=14,
    min_periods=5,
    ddof=1,
):
    """
    Compute rolling variance of local price elasticity.

    Args:
        df: Input DataFrame.
        group_cols (tuple): Grouping columns.
        date_col (str): Date column name.
        window (int): Rolling window size.
        min_periods (int): Minimum periods for rolling variance.
        ddof (int): Delta degrees of freedom.

    Returns:
        pd.DataFrame: DataFrame with rolling elasticity variance features.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    for col in group_cols:
        elas_loc_col = f"elasticite_locale_{col}"
        var_roll_col = f"var_elasticite_rolling_{col}"

        if elas_loc_col not in df.columns:
            raise ValueError(f"Missing column: {elas_loc_col}")

        df = df.sort_values([col, date_col])

        df[var_roll_col] = (
            df.groupby(col, observed=True)[elas_loc_col]
              .transform(
                  lambda x: x.rolling(window=window, min_periods=min_periods)
                            .var(ddof=ddof)
                            .shift(1)
              )
        )

        df.drop(columns=[elas_loc_col], inplace=True, errors="ignore")

    return df


def fill_elast_features(df, date_col="DATE"):
    """
    Fill missing elasticity and variance features and create imputation flags.

    Strategy:
    1. Forward-fill past values within each group
    2. Fill remaining gaps with historical median within the group
    3. Fallback to historical global median by date
    4. Final fallback to 0.0

    Args:
        df: Input DataFrame.
        date_col (str): Date column name.

    Returns:
        pd.DataFrame: DataFrame with filled elasticity features and imputation flags.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    sku_cols = ["elasticite_rolling_SKU", "var_elasticite_rolling_SKU"]
    cat_cols = ["elasticite_rolling_Category", "var_elasticite_rolling_Category"]
    style_cols = ["elasticite_rolling_Style", "var_elasticite_rolling_Style"]

    all_cols = sku_cols + cat_cols + style_cols

    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing elasticity columns for filling: {missing_cols}")

    df = df.sort_values(date_col)

    # Create imputation flags
    for col in all_cols:
        df[f"flag_imputed_{col}"] = df[col].isna().astype("int8")

    # Compute global historical medians by date
    global_history_map = {}
    for col in all_cols:
        median_by_date = (
            df.groupby(date_col, observed=True)[col]
              .median()
              .sort_index()
        )

        historical_global_median = (
            median_by_date.expanding(min_periods=1)
                          .median()
                          .shift(1)
        )

        global_history_map[col] = df[date_col].map(historical_global_median).astype(float)

    def _fill_groupwise(series, fallback_series):
        s = series.ffill()
        hist_med = s.expanding(min_periods=1).median().shift(1)
        s = s.fillna(hist_med)
        s = s.fillna(fallback_series)
        return s.fillna(0.0)

    def apply_fill(group_col, cols):
        nonlocal df
        df = df.sort_values([group_col, date_col])

        for col in cols:
            fallback_series = global_history_map[col].reindex(df.index)

            df[col] = (
                df.groupby(group_col, observed=True)[col]
                  .transform(lambda x: _fill_groupwise(x, fallback_series.loc[x.index]))
            )

    apply_fill("SKU", sku_cols)
    apply_fill("Category", cat_cols)
    apply_fill("Style", style_cols)

    return df


def basket_int(df):
    """
    Compute normalized basket intensity by SKU, Style, and Category.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with basket intensity features.
    """
    df = df.copy()

    required_cols = ["SKU", "Style", "Category", "quantity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in basket_int: {missing_cols}")

    df["sku_mean_qty"] = df.groupby("SKU", observed=True)["quantity"].transform("mean")
    df["style_mean_qty"] = df.groupby("Style", observed=True)["quantity"].transform("mean")
    df["cat_mean_qty"] = df.groupby("Category", observed=True)["quantity"].transform("mean")

    df["basket_intensity_sku_norm"] = df["quantity"] / df["sku_mean_qty"]
    df["basket_intensity_style_norm"] = df["quantity"] / df["style_mean_qty"]
    df["basket_intensity_cat_norm"] = df["quantity"] / df["cat_mean_qty"]

    df = df.drop(columns=["sku_mean_qty", "style_mean_qty", "cat_mean_qty"], errors="ignore")
    return df


def build_feature(df: pd.DataFrame, kpi="no") -> pd.DataFrame:
    """
    Build calendar, elasticity, variance, imputation flag, and optional KPI features.

    Args:
        df (pd.DataFrame): Input raw data.
        kpi (str): "yes" to compute basket intensity features.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values(["SKU", "DATE"]).reset_index(drop=True)

    required_cols = ["DATE", "SKU", "Style", "Category", "quantity", "price"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in build_feature: {missing_cols}")

    # Calendar features
    df["year"] = df["DATE"].dt.year
    df["Months"] = df["DATE"].dt.month
    df["day"] = df["DATE"].dt.day
    df["week"] = df["DATE"].dt.isocalendar().week.astype(int)

    # Cyclical month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["Months"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Months"] / 12)

    # Cyclical day encoding
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)

    # Cyclical week encoding
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    # Season
    df["mmdd"] = df["DATE"].dt.month * 100 + df["DATE"].dt.day

    df["saison"] = 3  # Winter
    df.loc[(df["mmdd"] >= 320) & (df["mmdd"] < 621), "saison"] = 0   # Spring
    df.loc[(df["mmdd"] >= 621) & (df["mmdd"] < 923), "saison"] = 1   # Summer
    df.loc[(df["mmdd"] >= 923) & (df["mmdd"] < 1221), "saison"] = 2  # Autumn

    # Rolling elasticity
    df = elast_rolling(
        df,
        group_cols=("SKU", "Style", "Category"),
        date_col="DATE",
        qty_col="quantity",
        price_col="price",
        window=14,
        min_periods=5,
    )

    # Rolling variance
    df = elast_variance(
        df,
        group_cols=("SKU", "Style", "Category"),
        date_col="DATE",
        window=14,
        min_periods=5,
    )

    # Fill elasticity features + create imputation flags
    df = fill_elast_features(df, date_col="DATE")

    # Optional KPI features
    if kpi == "yes":
        df = basket_int(df)

    # Drop intermediate columns
    col_del_f1 = ["mmdd", "Months", "day", "week"]
    df = df.drop(columns=col_del_f1, errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)

    return df