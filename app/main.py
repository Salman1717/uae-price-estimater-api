# app/main.py
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# ---------- Pydantic input schema ----------
class PredictRequest(BaseModel):
    bedrooms: int = Field(..., example=2)
    bathrooms: int = Field(..., example=2)
    size_sqft: float = Field(..., example=1200.0)
    verified: int = Field(1, example=1, ge=0, le=1)
    area: Optional[str] = Field(None, example="Marina")
    property_type: Optional[str] = Field(None, example="Apartment")
    furnishing: Optional[str] = Field(None, example="Furnished")
    priceDuration: Optional[str] = Field(None, example="per month")

class PredictResponse(BaseModel):
    price_aed: float
    price_per_sqft: float
    model_used: str
    explain: Dict[str, Any]

# ---------- App bootstrap ----------
app = FastAPI(title="UAE Real Estate Price Estimator API", version="0.1.0")

def load_joblib_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

# Load ensemble metadata
try:
    best_data = load_joblib_safe(os.path.join(MODEL_DIR, "best_ensemble_model.joblib"))
    feature_columns = best_data.get("feature_columns")
    best_name = best_data.get("best_name", "CatBoostStacking")
    best_weights = best_data.get("best_weights", None)
except Exception as e:
    best_data = None
    feature_columns = None
    best_name = "CatBoostOnly"
    best_weights = None
    print("⚠️ Warning: best_ensemble_model.joblib not loaded:", e)

# Load only existing models
cat_model = cat_meta = None

def try_load_models():
    global cat_model, cat_meta
    try:
        cat_model = load_joblib_safe(os.path.join(MODEL_DIR, "cat_model.joblib"))
    except Exception as e:
        print("⚠️ cat_model load failed:", e)

    try:
        cat_meta = load_joblib_safe(os.path.join(MODEL_DIR, "cat_meta.joblib"))
    except Exception as e:
        print("⚠️ cat_meta load failed:", e)

try_load_models()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {
            "best_ensemble_model": best_data is not None,
            "cat_model": cat_model is not None,
            "cat_meta": cat_meta is not None
        }
    }

@app.get("/version")
def version():
    return {"api_version": app.version, "model_name": best_name}

# ---------- Feature engineering ----------
def build_feature_row(req: PredictRequest) -> pd.DataFrame:
    now = datetime.utcnow()
    year, month, day_of_week = now.year, now.month, now.weekday()

    base = {
        "bedrooms": max(1, req.bedrooms),
        "bathrooms": max(1, req.bathrooms),
        "sizeMin": float(max(1, req.size_sqft)),
        "verified": int(req.verified),
        "year": year,
        "month": month,
        "day_of_week": day_of_week,
        "density_index": req.size_sqft / max(1, req.bedrooms),
        "bathroom_bedroom_ratio": req.bathrooms / max(1, req.bedrooms),
        "bed_bath_interaction": req.bedrooms * req.bathrooms,
        "is_large_property": int(req.size_sqft > 2000),
        "is_luxury": 0,
    }

    df = pd.DataFrame([base])
    if feature_columns is None:
        raise HTTPException(status_code=500, detail="Feature columns metadata missing.")

    # Add missing columns as 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

# ---------- Simple explanation ----------
def simple_explain(req: PredictRequest, predicted_price: float) -> Dict[str, Any]:
    contribs = {
        "size_contribution": predicted_price * min(0.7, req.size_sqft / (req.size_sqft + 1000)),
        "bedroom_contribution": predicted_price * 0.08 * (req.bedrooms - 1),
        "bathroom_contribution": predicted_price * 0.05 * (req.bathrooms - 1),
        "verified_bonus": predicted_price * 0.02 if req.verified == 1 else 0,
    }
    top = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))[:3]
    return {"top_contributors": top, "note": "Heuristic explanation only (not SHAP)."}

# ---------- Prediction ----------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if best_data is None:
        raise HTTPException(status_code=500, detail="Server missing model metadata (best_ensemble_model.joblib).")

    # Build feature row
    try:
        X = build_feature_row(req)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    preds = {}

    # --- CatBoost base prediction ---
    if cat_model is not None:
        try:
            cat_pred = cat_model.predict(X)
            cat_pred = np.expm1(cat_pred) if np.mean(cat_pred) < 20 else cat_pred  # unlog if needed
            preds["CatBoost"] = float(np.asarray(cat_pred).ravel()[0])
        except Exception as e:
            preds["CatBoost"] = None
    else:
        preds["CatBoost"] = None

    # --- Ensemble / Meta prediction ---
    if cat_meta is not None:
        try:
            ensemble_pred = cat_meta.predict(X)
            ensemble_pred = np.expm1(ensemble_pred) if np.mean(ensemble_pred) < 20 else ensemble_pred
            preds["Ensemble"] = float(np.asarray(ensemble_pred).ravel()[0])
        except Exception as e:
            preds["Ensemble"] = None
    else:
        preds["Ensemble"] = None

    valid_preds = [v for v in preds.values() if v is not None]
    if not valid_preds:
        raise HTTPException(status_code=500, detail="No valid model predictions available (CatBoost/Ensemble missing).")

    # --- Compute consensus and confidence ---
    avg_pred = float(np.mean(valid_preds))
    std_pred = float(np.std(valid_preds))
    confidence = round(max(0.0, 1.0 - std_pred / (avg_pred + 1e-6)), 3)  # simple [0–1] confidence

    price_per_sqft = avg_pred / max(1.0, req.size_sqft)
    explain = simple_explain(req, avg_pred)

    return {
        "price_aed": avg_pred,
        "price_per_sqft": price_per_sqft,
        "model_used": "CatBoost + Ensemble Consensus",
        "explain": {
            **explain,
            "individual_predictions": preds,
            "ensemble_std": std_pred,
            "confidence_score": confidence,
            "note": "Confidence is higher when CatBoost and Ensemble predictions agree closely."
        }
    }
