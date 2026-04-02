import pandas as pd
import numpy as np

from app.models.model_loader import model
from app.services.feature_engineering_service import build_feature


def predict_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int = 21
) -> pd.DataFrame:

    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    if df["DATE"].isna().any():
        raise ValueError("Certaines dates sont invalides.")

    work_df = df.copy()
    all_predictions = []

    last_date = work_df["DATE"].max()

    # récupérer la dernière info connue par SKU
    sku_info = (
        work_df.sort_values(["SKU", "DATE"])
               .groupby("SKU", as_index=False)
               .last()[["SKU", "Style", "Category", "price"]]
    )

    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)

        future_rows = sku_info.copy()
        future_rows["DATE"] = next_date
        future_rows["quantity"] = np.nan

        temp_df = pd.concat([work_df, future_rows], ignore_index=True)
        temp_df = temp_df.sort_values(["SKU", "DATE"])

        temp_df = build_feature(temp_df)

        missing_features = [col for col in feature_cols if col not in temp_df.columns]
        if missing_features:
            raise ValueError(f"Features manquantes : {missing_features}")

        rows_future = temp_df["DATE"] == next_date
        X_future = temp_df.loc[rows_future, feature_cols]

        y_pred = model.predict(X_future)

        temp_df.loc[rows_future, "quantity"] = y_pred

        pred_rows = temp_df.loc[rows_future, ["DATE", "SKU", "Style", "Category", "price"]].copy()
        pred_rows["prediction_quantity"] = y_pred

        all_predictions.append(pred_rows)

        history_rows = temp_df.loc[rows_future, ["DATE", "SKU", "Style", "Category", "price", "quantity"]].copy()
        work_df = pd.concat([work_df, history_rows], ignore_index=True)

    df_predict = pd.concat(all_predictions, ignore_index=True)

    return df_predict