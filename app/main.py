from fastapi import FastAPI
from app.api.routes.predict import router as predict_router

app = FastAPI(
    title="ML Prediction API",
    version="1.0"
)

app.include_router(predict_router, prefix="/api")