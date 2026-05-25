"""
Phase 1: Inventory Logic
Sits on top of your XGBoost predictions and converts them into
reorder decisions using pharmaceutical inventory principles.
"""
 
import numpy as np
from dataclasses import dataclass
from typing import Optional
 
 
# ---------------------------------------------------------------------------
# Phase 1: Inventory Logic & Dataclasses
# ---------------------------------------------------------------------------
Z_SCORES = {
    0.80: 0.84,
    0.85: 1.04,
    0.90: 1.28,
    0.95: 1.65,
    0.99: 2.33,
}

@dataclass
class InventoryParams:
    drug_name: str
    current_stock: int
    lead_time_days: int
    service_level: float = 0.95
    ordering_cost: float = 50.0
    holding_cost_per_unit: float = 2.0
    max_shelf_capacity: int = 200

@dataclass
class InventoryDecision:
    drug_name: str
    current_stock: int
    mean_daily_demand: float
    forecast_std: float
    forecast_7d: list[float]
    safety_stock: int
    reorder_point: int
    eoq: int
    days_of_cover: float
    action: str
    action_message: str
    order_quantity: Optional[int]
    stockout_risk_pct: float

def compute_safety_stock(forecast_std: float, lead_time_days: int, service_level: float) -> int:
    z = Z_SCORES.get(service_level, 1.65)
    safety = z * forecast_std * np.sqrt(lead_time_days)
    return int(np.ceil(safety))

def compute_reorder_point(mean_daily_demand: float, lead_time_days: int, safety_stock: int) -> int:
    demand_during_lt = mean_daily_demand * lead_time_days
    return int(np.ceil(demand_during_lt + safety_stock))

def compute_eoq(mean_daily_demand: float, ordering_cost: float, holding_cost_per_unit: float) -> int:
    annual_demand = mean_daily_demand * 365
    if annual_demand == 0 or holding_cost_per_unit == 0:
        return 1
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
    return max(1, int(np.ceil(eoq)))

def estimate_stockout_risk(current_stock: int, reorder_point: int, mean_daily_demand: float, forecast_std: float, lead_time_days: int) -> float:
    if current_stock <= 0: return 100.0
    if current_stock > reorder_point * 2: return 0.0

    mu = mean_daily_demand * lead_time_days
    sigma = forecast_std * np.sqrt(lead_time_days)

    if sigma == 0:
        return 0.0 if current_stock >= mu else 100.0

    from scipy.stats import norm
    risk = norm.sf(current_stock, loc=mu, scale=sigma) * 100
    return round(float(np.clip(risk, 0, 100)), 1)

def make_decision(params: InventoryParams, predictions: np.ndarray) -> InventoryDecision:
    forecast_7d = list(np.clip(predictions[:7], 0, None).round(2))
    mean_daily = float(np.mean(forecast_7d))
    std_daily = float(np.std(forecast_7d)) if len(forecast_7d) > 1 else mean_daily * 0.3
    std_daily = max(std_daily, mean_daily * 0.1)

    safety_stock = compute_safety_stock(std_daily, params.lead_time_days, params.service_level)
    rop = compute_reorder_point(mean_daily, params.lead_time_days, safety_stock)
    eoq = compute_eoq(mean_daily, params.ordering_cost, params.holding_cost_per_unit)

    eoq = min(eoq, params.max_shelf_capacity - params.current_stock)
    eoq = max(eoq, 1)

    days_cover = (params.current_stock / mean_daily) if mean_daily > 0 else float("inf")
    stockout_risk = estimate_stockout_risk(params.current_stock, rop, mean_daily, std_daily, params.lead_time_days)

    if params.current_stock <= 0:
        action = "STOCKOUT"
        action_message = f"STOCKOUT: No units of {params.drug_name} remain. Place an emergency order for {eoq} units immediately. Consider contacting an alternate supplier to reduce the {params.lead_time_days}-day lead time."
        order_quantity = eoq
    elif params.current_stock <= rop:
        action = "REORDER"
        action_message = f"Stock ({params.current_stock} units) has reached the reorder point ({rop} units). Place an order for {eoq} units today. At the current rate you have ~{days_cover:.0f} days of cover, which is less than your {params.lead_time_days}-day lead time."
        order_quantity = eoq
    else:
        action = "OK"
        action_message = f"Stock is healthy at {params.current_stock} units (~{days_cover:.0f} days of cover). No action needed. Next review when stock approaches {rop} units."
        order_quantity = None

    return InventoryDecision(
        drug_name=params.drug_name,
        current_stock=params.current_stock,
        mean_daily_demand=round(mean_daily, 3),
        forecast_std=round(std_daily, 3),
        forecast_7d=forecast_7d,
        safety_stock=safety_stock,
        reorder_point=rop,
        eoq=eoq,
        days_of_cover=round(days_cover, 1),
        action=action,
        action_message=action_message,
        order_quantity=order_quantity,
        stockout_risk_pct=stockout_risk,
    )