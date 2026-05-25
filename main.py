"""
Phase 2: FastAPI Backend
Wraps the XGBoost model + inventory logic into a REST API.
The pharmacist dashboard calls this; the model never surfaces directly.
 
Run with:
    uvicorn api:app --reload --port 8000
 
Endpoints:
    POST /forecast   → get decision for one drug
    GET  /drugs      → list all configured drugs
    GET  /health     → health check
"""
 
import os
import pickle
from fastapi.concurrency import asynccontextmanager
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional
import xgboost as xgb
import joblib
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
 
from inventory import InventoryParams, InventoryDecision, make_decision
 
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup using the JSON format
    MODEL_PATH = os.getenv("MODEL_PATH", "xgb_sales_model.json")
    if os.path.exists(MODEL_PATH):
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        ml_models["xgb"] = model
        print(f"[startup] Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"[startup] WARNING: model file not found at {MODEL_PATH}. Using mock predictions.")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(
    title="Pharmacy Demand API",
    description="XGBoost Poisson demand forecasting + inventory decisions for pharmacists",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
DRUG_CATALOGUE = {
    "sertraline_50mg": {
        "display_name": "Sertraline 50mg",
        "lead_time_days": 10,
        "ordering_cost": 45.0,
        "holding_cost_per_unit": 3.0, 
        "max_shelf_capacity": 100, 
    }
}


class ForecastRequest(BaseModel):
    drug_id: str = Field(..., example="sertraline_50mg")
    current_stock: int = Field(..., ge=0, example=38)
    service_level: float = Field(0.95, ge=0.80, le=0.99, example=0.95)
    lead_time_override: Optional[int] = Field(None, ge=1, le=90)

class ForecastResponse(BaseModel):
    drug_name: str
    current_stock: int
    forecast_date: str
    forecast_7d: list[float]
    forecast_dates: list[str]
    mean_daily_demand: float
    forecast_std: float
    safety_stock: int
    reorder_point: int
    eoq: int
    days_of_cover: float
    stockout_risk_pct: float
    action: str
    action_message: str
    order_quantity: Optional[int]
    feature_importance: dict

class DrugListItem(BaseModel):
    drug_id: str
    display_name: str
    lead_time_days: int

def build_features(start_date: date, n_days: int = 10) -> pd.DataFrame:
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    df = pd.DataFrame({"date": pd.to_datetime(dates)})

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month

    # Placeholders: Replace with real DB lookups in production
    df["sales_lag_7"] = 0.87  
    df["sales_lag_14"] = 0.92 
    df["rolling_28_mean"] = 0.87 

    last_train_idx = int(os.getenv("LAST_TIME_INDEX", "730"))
    df["time_index"] = last_train_idx + np.arange(n_days)

    return df

FEATURE_COLS = [
    "day_of_week",
    "is_weekend",
    "month",
    "sales_lag_7",
    "sales_lag_14",
    "rolling_28_mean",
    "time_index",
]

MOCK_IMPORTANCE = {
    "time_index": 497, "day_of_week": 246, "rolling_28_mean": 215,
    "month": 194, "sales_lag_14": 84, "sales_lag_7": 74,
}

def get_predictions(n_days: int = 10) -> np.ndarray:
    today = date.today()
    features_df = build_features(today, n_days)
    X = features_df[FEATURE_COLS]

    model = ml_models.get("xgb")
    
    if model is not None:
        preds = model.predict(X)
    else:
        rng = np.random.default_rng(seed=int(today.strftime("%Y%m%d")))
        preds = rng.poisson(0.87, size=n_days).astype(float)

    return np.clip(preds, 0, None)

def get_feature_importance() -> dict:
    model = ml_models.get("xgb")
    if model is not None:
        scores = model.get_booster().get_score(importance_type="weight")
        return {k: int(v) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    return MOCK_IMPORTANCE

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "xgb" in ml_models}

@app.get("/drugs", response_model=list[DrugListItem])
def list_drugs():
    return [
        DrugListItem(drug_id=k, display_name=v["display_name"], lead_time_days=v["lead_time_days"])
        for k, v in DRUG_CATALOGUE.items()
    ]

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if req.drug_id not in DRUG_CATALOGUE:
        raise HTTPException(status_code=404, detail=f"Drug '{req.drug_id}' not found in catalogue.")

    drug_cfg = DRUG_CATALOGUE[req.drug_id]
    lead_time = req.lead_time_override or drug_cfg["lead_time_days"]

    params = InventoryParams(
        drug_name=drug_cfg["display_name"],
        current_stock=req.current_stock,
        lead_time_days=lead_time,
        service_level=req.service_level,
        ordering_cost=drug_cfg["ordering_cost"],
        holding_cost_per_unit=drug_cfg["holding_cost_per_unit"],
        max_shelf_capacity=drug_cfg["max_shelf_capacity"],
    )

    predictions = get_predictions(n_days=10)
    decision = make_decision(params, predictions)
    importance = get_feature_importance()

    today = date.today()
    forecast_dates = [(today + timedelta(days=i)).isoformat() for i in range(7)]

    return ForecastResponse(
        drug_name=decision.drug_name,
        current_stock=decision.current_stock,
        forecast_date=today.isoformat(),
        forecast_7d=decision.forecast_7d,
        forecast_dates=forecast_dates,
        mean_daily_demand=decision.mean_daily_demand,
        forecast_std=decision.forecast_std,
        safety_stock=decision.safety_stock,
        reorder_point=decision.reorder_point,
        eoq=decision.eoq,
        days_of_cover=decision.days_of_cover,
        stockout_risk_pct=decision.stockout_risk_pct,
        action=decision.action,
        action_message=decision.action_message,
        order_quantity=decision.order_quantity,
        feature_importance=importance,
    )