from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "artifacts" / "best_model_20260401_160834.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

loaded = joblib.load(MODEL_PATH)
model = loaded["pipeline"] if isinstance(loaded, dict) else loaded