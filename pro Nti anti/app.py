import os
import pickle
import numpy as np
import pandas as pd
from typing import Union, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==============================================================================
# NumPy 1.x / 2.x Cross-Version Pickle Compatibility Patch
# ==============================================================================
try:
    import numpy.random._pickle as _np_pickle
    _orig_ctor = _np_pickle.__bit_generator_ctor
    def _patched_bit_generator_ctor(bit_generator_name="MT19937"):
        if hasattr(bit_generator_name, "__name__"):
            bit_generator_name = bit_generator_name.__name__
        elif isinstance(bit_generator_name, str) and "." in bit_generator_name:
            bit_generator_name = bit_generator_name.split(".")[-1]
        return _orig_ctor(str(bit_generator_name))
    _np_pickle.__bit_generator_ctor = _patched_bit_generator_ctor
except Exception:
    pass

# ==============================================================================
# FastAPI Application Initialization
# ==============================================================================
app = FastAPI(
    title="SME Financial Intelligence Multi-Target API",
    description=(
        "Comprehensive FastAPI suite serving ML predictions for:\n"
        "1. Expenses (`df2`): Predicts `amount_egp` (Regression)\n"
        "2. Daily Cash Flow (`df1`): Predicts `net_cash_flow_egp` (Regression)\n"
        "3. Payments (`df`): Predicts `payment_delay_days` (Regression)\n"
        "4. Invoices (`df3`): Predicts `is_late` (Binary Classification)"
    ),
    version="2.1.0"
)

# Enable CORS for Streamlit and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_artifact(filename, root_fallback=None):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    if root_fallback and os.path.exists(root_fallback):
        with open(root_fallback, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Artifact {filename} not found in {MODELS_DIR} or fallback.")

# 1. Expenses (df2)
try:
    model_exp = load_artifact("best_exp_model.pkl", "best_xgb_model.pkl")
    scaler_exp = load_artifact("scaler_exp.pkl", "scaler.pkl")
    le_exp_bus = load_artifact("le_exp_bus.pkl")
    le_exp_sec = load_artifact("le_exp_sec.pkl")
    le_exp_type = load_artifact("le_exp_type.pkl")
    features_exp = load_artifact("features_exp.pkl")
except Exception as e:
    print(f"Warning: Expenses artifacts loading issue: {e}")

# 2. Daily Cash Flow (df1)
try:
    model_cash = load_artifact("best_cash_model.pkl")
    scaler_cash = load_artifact("scaler_cash.pkl")
    le_cash_bus = load_artifact("le_cash_bus.pkl")
    le_cash_sec = load_artifact("le_cash_sec.pkl")
    features_cash = load_artifact("features_cash.pkl")
except Exception as e:
    print(f"Warning: Cash Flow artifacts loading issue: {e}")

# 3. Payments (df)
try:
    model_pay = load_artifact("best_pay_model.pkl")
    scaler_pay = load_artifact("scaler_pay.pkl")
    le_pay = load_artifact("le_pay.pkl")
    features_pay = load_artifact("features_pay.pkl")
except Exception as e:
    print(f"Warning: Payments artifacts loading issue: {e}")

# 4. Invoices (df3)
try:
    model_inv = load_artifact("best_inv_model.pkl")
    scaler_inv = load_artifact("scaler_inv.pkl")
    le_inv_bus = load_artifact("le_inv_bus.pkl")
    le_inv_cust = load_artifact("le_inv_cust.pkl")
    features_inv = load_artifact("features_inv.pkl")
except Exception as e:
    print(f"Warning: Invoices artifacts loading issue: {e}")


# ==============================================================================
# Helper encoder functions
# ==============================================================================
def encode_val(val, le):
    if isinstance(val, (int, float, np.integer)):
        return int(val)
    str_val = str(val).strip()
    if str_val in le.classes_:
        return int(le.transform([str_val])[0])
    try:
        return int(str_val)
    except ValueError:
        return 0


# ==============================================================================
# Request Schemas
# ==============================================================================

# 1. df2 Expense Input
class ExpenseInput(BaseModel):
    business_id: Union[str, int] = Field(default="SME_001", description="Business ID (e.g. 'SME_001' or int)")
    sector: Union[str, int] = Field(default="Wholesale", description="Sector name (e.g. 'Wholesale') or int")
    expense_type: Union[str, int] = Field(default="supplier_payments_egp", description="Expense type or int")
    date_year: int = Field(default=2024, description="Year")
    date_month: int = Field(default=5, ge=1, le=12, description="Month (1-12)")
    date_day: int = Field(default=15, ge=1, le=31, description="Day (1-31)")

# 2. df1 Daily Cash Flow Input
class CashFlowInput(BaseModel):
    business_id: Union[str, int] = Field(default="SME_001")
    sector: Union[str, int] = Field(default="Wholesale")
    day_of_week: int = Field(default=2, ge=0, le=6)
    month: int = Field(default=6, ge=1, le=12)
    day: int = Field(default=15, ge=1, le=31)
    year: int = Field(default=2024)
    is_weekend: int = Field(default=0, ge=0, le=1)
    promotion_flag: int = Field(default=0, ge=0, le=1)
    sales_egp: float = Field(default=25000.0)
    cash_sales_egp: float = Field(default=12000.0)
    receivables_created_egp: float = Field(default=13000.0)
    receivables_collected_egp: float = Field(default=10000.0)
    supplier_payables_created_egp: float = Field(default=14000.0)
    supplier_payments_egp: float = Field(default=11000.0)
    payroll_egp: float = Field(default=3000.0)
    rent_egp: float = Field(default=0.0)
    taxes_egp: float = Field(default=0.0)
    loan_payments_egp: float = Field(default=0.0)
    other_income_egp: float = Field(default=500.0)

# 3. df Payments Input
class PaymentInput(BaseModel):
    business_id: Union[str, int] = Field(default="SME_001")
    amount_egp: float = Field(default=5000.0)
    year: int = Field(default=2024)
    month: int = Field(default=6, ge=1, le=12)
    day: int = Field(default=15, ge=1, le=31)
    day_of_week: int = Field(default=3, ge=0, le=6)

# 4. df3 Invoices Input (Classification)
class InvoiceInput(BaseModel):
    business_id: Union[str, int] = Field(default="SME_001", description="Business ID (e.g. 'SME_001' or int)")
    customer_id: Union[str, int] = Field(default="CUST_001", description="Customer ID (e.g. 'CUST_001' or int)")
    invoice_amount_egp: float = Field(default=12500.0, description="Invoice amount in EGP")
    terms_days: int = Field(default=30, ge=1, le=180, description="Payment terms in days (due_date - invoice_date)")
    year: int = Field(default=2024, description="Invoice issue year")
    month: int = Field(default=6, ge=1, le=12, description="Invoice issue month (1-12)")
    day: int = Field(default=15, ge=1, le=31, description="Invoice issue day (1-31)")
    day_of_week: int = Field(default=2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "business_id": "SME_001",
                "customer_id": "CUST_044",
                "invoice_amount_egp": 12500.0,
                "terms_days": 30,
                "year": 2024,
                "month": 6,
                "day": 15,
                "day_of_week": 2
            }
        }
    }


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "api_name": "SME Financial Intelligence Multi-Target API",
        "version": "2.1.0",
        "endpoints": {
            "/predict/expenses": "Predicts expense amount (df2 Regression)",
            "/predict/cashflow": "Predicts daily net cash flow (df1 Regression)",
            "/predict/payments": "Predicts payment delay in days (df Regression)",
            "/predict/invoices": "Predicts invoice late payment status is_late (df3 Classification)",
            "/models/info": "Summary of active ML models & metrics",
            "/docs": "Interactive Swagger UI documentation"
        }
    }

@app.get("/models/info")
def models_info():
    return {
        "models": {
            "df_payments": {
                "target": "payment_delay_days",
                "type": "Regression",
                "best_model": str(type(model_pay).__name__),
                "features": features_pay
            },
            "df1_cashflow": {
                "target": "net_cash_flow_egp",
                "type": "Regression",
                "best_model": str(type(model_cash).__name__),
                "features": features_cash
            },
            "df2_expenses": {
                "target": "amount_egp",
                "type": "Regression",
                "best_model": str(type(model_exp).__name__),
                "features": features_exp
            },
            "df3_invoices": {
                "target": "is_late",
                "type": "Binary Classification (0: On-Time, 1: Late)",
                "best_model": str(type(model_inv).__name__),
                "features": features_inv
            }
        }
    }

# 1. Predict Expenses (df2)
@app.post("/predict/expenses")
@app.post("/predict")
def predict_expenses(data: ExpenseInput):
    try:
        bus_encoded = encode_val(data.business_id, le_exp_bus)
        sec_encoded = encode_val(data.sector, le_exp_sec)
        type_encoded = encode_val(data.expense_type, le_exp_type)

        input_df = pd.DataFrame([{
            'business_id': bus_encoded,
            'sector': sec_encoded,
            'expense_type': type_encoded,
            'year': data.date_year,
            'month': data.date_month,
            'day': data.date_day
        }])[features_exp]

        scaled_data = scaler_exp.transform(input_df)
        pred = float(model_exp.predict(scaled_data)[0])

        return {
            "target": "amount_egp",
            "predicted_amount_egp": round(pred, 2),
            "formatted": f"{round(pred, 2):,} EGP",
            "model_used": type(model_exp).__name__
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 2. Predict Cash Flow (df1)
@app.post("/predict/cashflow")
def predict_cashflow(data: CashFlowInput):
    try:
        bus_encoded = encode_val(data.business_id, le_cash_bus)
        sec_encoded = encode_val(data.sector, le_cash_sec)

        input_dict = {
            'business_id': bus_encoded,
            'sector': sec_encoded,
            'day_of_week': data.day_of_week,
            'month': data.month,
            'day': data.day,
            'year': data.year,
            'is_weekend': data.is_weekend,
            'promotion_flag': data.promotion_flag,
            'sales_egp': data.sales_egp,
            'cash_sales_egp': data.cash_sales_egp,
            'receivables_created_egp': data.receivables_created_egp,
            'receivables_collected_egp': data.receivables_collected_egp,
            'supplier_payables_created_egp': data.supplier_payables_created_egp,
            'supplier_payments_egp': data.supplier_payments_egp,
            'payroll_egp': data.payroll_egp,
            'rent_egp': data.rent_egp,
            'taxes_egp': data.taxes_egp,
            'loan_payments_egp': data.loan_payments_egp,
            'other_income_egp': data.other_income_egp
        }

        input_df = pd.DataFrame([input_dict])[features_cash]
        scaled_data = scaler_cash.transform(input_df)
        pred = float(model_cash.predict(scaled_data)[0])
        status = "Surplus (Positive)" if pred >= 0 else "Deficit (Negative)"

        return {
            "target": "net_cash_flow_egp",
            "predicted_net_cash_flow_egp": round(pred, 2),
            "formatted": f"{round(pred, 2):,} EGP",
            "cash_flow_status": status,
            "model_used": type(model_cash).__name__
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 3. Predict Payments (df)
@app.post("/predict/payments")
def predict_payments(data: PaymentInput):
    try:
        bus_encoded = encode_val(data.business_id, le_pay)

        input_dict = {
            'business_id': bus_encoded,
            'amount_egp': data.amount_egp,
            'year': data.year,
            'month': data.month,
            'day': data.day,
            'day_of_week': data.day_of_week
        }

        input_df = pd.DataFrame([input_dict])[features_pay]
        scaled_data = scaler_pay.transform(input_df)
        pred = float(model_pay.predict(scaled_data)[0])
        pred_rounded = max(0.0, round(pred, 1))
        status = "On Time" if pred_rounded < 1 else f"Delayed by ~{int(pred_rounded)} days"

        return {
            "target": "payment_delay_days",
            "predicted_delay_days": pred_rounded,
            "status": status,
            "model_used": type(model_pay).__name__
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 4. Predict Invoices Late Classification (df3)
@app.post("/predict/invoices")
@app.post("/predict/late")
def predict_invoices(data: InvoiceInput):
    try:
        bus_encoded = encode_val(data.business_id, le_inv_bus)
        cust_encoded = encode_val(data.customer_id, le_inv_cust)

        input_dict = {
            'business_id': bus_encoded,
            'customer_id': cust_encoded,
            'invoice_amount_egp': data.invoice_amount_egp,
            'terms_days': data.terms_days,
            'year': data.year,
            'month': data.month,
            'day': data.day,
            'day_of_week': data.day_of_week
        }

        input_df = pd.DataFrame([input_dict])[features_inv]
        scaled_data = scaler_inv.transform(input_df)

        pred_class = int(model_inv.predict(scaled_data)[0])
        if hasattr(model_inv, "predict_proba"):
            late_prob = float(model_inv.predict_proba(scaled_data)[0][1])
        else:
            late_prob = float(pred_class)

        if late_prob >= 0.5:
            status = "Late Payment Expected"
            risk_level = "High Risk"
        elif late_prob >= 0.25:
            status = "Moderate Delay Risk"
            risk_level = "Medium Risk"
        else:
            status = "On-Time Payment Expected"
            risk_level = "Low Risk"

        return {
            "target": "is_late",
            "predicted_is_late": pred_class,
            "late_probability": round(late_prob, 4),
            "late_probability_pct": f"{late_prob * 100:.1f}%",
            "status": status,
            "risk_level": risk_level,
            "model_used": type(model_inv).__name__
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
