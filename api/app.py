import os
import joblib
import xgboost as xgb
import pandas as pd
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# FASTAPI APP INIT
app = FastAPI(
    title="ACH Fraud Detection API",
    description="ML-based fraud scoring service with API key authentication",
    version="1.0"
)

# API KEY AUTHENTICATION
API_KEY = os.getenv("API_KEY")  # MUST be set in Azure App Service
API_KEY_NAME = "x-api-key"

def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY is None:
        raise HTTPException(
            status_code=500,
            detail="API key not configured on server."
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key."
        )

# LOADING MODEL + FEATURES
MODEL_PATH   = os.getenv("MODEL_PATH", "models/xgb_fraud_model.bin")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_columns.pkl")

try:
    feature_cols: List[str] = joblib.load(FEATURES_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load feature columns: {e}")

# Loading XGBoost booster
try:
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load XGBoost model: {e}")

# REQUEST SCHEMAS
class SingleScoreRequest(BaseModel):
    features: Dict[str, float]

class BatchScoreRequest(BaseModel):
    records: List[Dict[str, float]]

# ENDPOINTS
@app.get("/health")
def health_check():
    """Simple health check"""
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "n_features": len(feature_cols)
    }


@app.get("/features")
def list_features():
    return {"required_features": feature_cols}


# SINGLE TRANSACTION SCORING
@app.post("/score", dependencies=[Depends(verify_api_key)])
def score_single(request: SingleScoreRequest):

    # Validating incoming feature keys
    missing = [f for f in feature_cols if f not in request.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}"
        )

    df = pd.DataFrame([request.features])[feature_cols]
    dmatrix = xgb.DMatrix(df)
    pred = booster.predict(dmatrix)[0]

    return {
        "fraud_probability": float(pred),
        "fraud_score": float(pred * 100)
    }


#BATCH SCORING 
@app.post("/score_batch", dependencies=[Depends(verify_api_key)])
def score_batch(request: BatchScoreRequest):

    if len(request.records) == 0:
        raise HTTPException(status_code=400, detail="No records provided.")

    df = pd.DataFrame(request.records)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing_cols}"
        )

    df = df[feature_cols]
    dmatrix = xgb.DMatrix(df)
    preds = booster.predict(dmatrix)

    return {
        "n_records": len(preds),
        "scores": [
            {"fraud_probability": float(p), "fraud_score": float(p * 100)}
            for p in preds
        ]
    }

# ROOT ENDPOINT
@app.get("/")
def root():
    return {
        "message": "ACH Fraud Detection API is running",
        "docs_url": "/docs",
        "secured_endpoints": ["/score", "/score_batch"],
        "auth": "Use header x-api-key"
    }
