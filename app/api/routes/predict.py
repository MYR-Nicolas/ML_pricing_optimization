from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.prediction_service import predict_df
from app.services.feature_engineering_service import build_feature
import pandas as pd
import numpy as np
import io
import traceback

def dataframe_to_json_records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()

    # Remplace inf / -inf par NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Force les colonnes en object pour permettre de vrais None Python
    df = df.astype(object)

    # Remplace tous les NaN/NaT par None
    df = df.where(pd.notnull(df), None)

    return df.to_dict(orient="records")


router = APIRouter()

FEATURE_COLS = [
    "DATE",
    "Style",
    "quantity",
    "price",
    "year",
    "Category",
    "log_price",
    "dlog_qty_SKU",
    "dlog_price_SKU",
    "elasticite_rolling_SKU",
    "dlog_qty_Style",
    "dlog_price_Style",
    "elasticite_rolling_Style",
    "dlog_qty_Category",
    "dlog_price_Category",
    "elasticite_rolling_Category",
    "var_elasticite_rolling_SKU",
    "var_elasticite_rolling_Style",
    "var_elasticite_rolling_Category",
    "flag_imputed_elasticite_rolling_SKU",
    "flag_imputed_var_elasticite_rolling_SKU",
    "flag_imputed_elasticite_rolling_Category",
    "flag_imputed_var_elasticite_rolling_Category",
    "flag_imputed_elasticite_rolling_Style",
    "flag_imputed_var_elasticite_rolling_Style",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "week_sin",
    "week_cos",
    "saison"
]

def dataframe_to_json_records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un CSV.")

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture CSV : {str(e)}")

    try:
        result_df = predict_df(df, feature_cols=FEATURE_COLS, horizon=21)
        return dataframe_to_json_records(result_df)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur prédiction : {type(e).__name__}: {str(e)}"
        )


@router.post("/features-file")
async def features_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un CSV.")

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture CSV : {str(e)}")

    try:
        features_df = build_feature(df, kpi="yes")
        return dataframe_to_json_records(features_df)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur calcul features : {type(e).__name__}: {str(e)}"
        )