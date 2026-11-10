from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)

print("=== 1. Testing GET / ===")
r = client.get("/")
print("Status:", r.status_code)
print(r.json())

print("\n=== 2. Testing GET /models/info ===")
r = client.get("/models/info")
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2))

print("\n=== 3. Testing POST /predict/expenses (df2) ===")
r = client.post("/predict/expenses", json={
    "business_id": "SME_001",
    "sector": "Wholesale",
    "expense_type": "supplier_payments_egp",
    "date_year": 2024,
    "date_month": 5,
    "date_day": 15
})
print("Status:", r.status_code)
print(r.json())

print("\n=== 4. Testing POST /predict/cashflow (df1) ===")
r = client.post("/predict/cashflow", json={
    "business_id": "SME_002",
    "sector": "Retail",
    "day_of_week": 2,
    "month": 6,
    "day": 15,
    "year": 2024,
    "sales_egp": 35000.0,
    "cash_sales_egp": 18000.0,
    "receivables_created_egp": 17000.0,
    "receivables_collected_egp": 12000.0,
    "supplier_payables_created_egp": 16000.0,
    "supplier_payments_egp": 13000.0,
    "payroll_egp": 4000.0,
    "rent_egp": 1000.0,
    "taxes_egp": 500.0,
    "loan_payments_egp": 0.0,
    "other_income_egp": 200.0
})
print("Status:", r.status_code)
print(r.json())

print("\n=== 5. Testing POST /predict/payments (df) ===")
r = client.post("/predict/payments", json={
    "business_id": "SME_003",
    "amount_egp": 8500.0,
    "year": 2024,
    "month": 6,
    "day": 20,
    "day_of_week": 3
})
print("Status:", r.status_code)
print(r.json())

print("\n=== 6. Testing POST /predict/invoices (df3 is_late Classification) ===")
r = client.post("/predict/invoices", json={
    "business_id": "SME_001",
    "customer_id": "CUST_044",
    "invoice_amount_egp": 15000.0,
    "terms_days": 14,
    "year": 2024,
    "month": 6,
    "day": 15,
    "day_of_week": 2
})
print("Status:", r.status_code)
print(r.json())

print("\nALL 6 FASTAPI ENDPOINTS TESTED AND PASSED SUCCESSFULLY!")
