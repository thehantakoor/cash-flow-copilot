import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('models', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# ==============================================================================
# 1. Pipeline for df (Payments -> Target: payment_delay_days)
# ==============================================================================
print("="*60)
print("TRAINING PIPELINE FOR DF (PAYMENTS)")
print("="*60)

df_pay = pd.read_csv('synthetic_sme_payments.csv')
print("Payments shape:", df_pay.shape)

df_pay_encoded = df_pay.copy()
df_pay_encoded['payment_date'] = pd.to_datetime(df_pay_encoded['payment_date'])
df_pay_encoded['year'] = df_pay_encoded['payment_date'].dt.year
df_pay_encoded['month'] = df_pay_encoded['payment_date'].dt.month
df_pay_encoded['day'] = df_pay_encoded['payment_date'].dt.day
df_pay_encoded['day_of_week'] = df_pay_encoded['payment_date'].dt.dayofweek
df_pay_encoded.drop(columns=['payment_date', 'payment_id', 'invoice_id'], inplace=True)

le_pay = LabelEncoder()
df_pay_encoded['business_id'] = le_pay.fit_transform(df_pay_encoded['business_id'])

X_pay = df_pay_encoded.drop(columns=['payment_delay_days'])
y_pay = df_pay_encoded['payment_delay_days']

X_pay_train, X_pay_test, y_pay_train, y_pay_test = train_test_split(
    X_pay, y_pay, test_size=0.2, random_state=42
)

scaler_pay = StandardScaler()
X_pay_train_scaled = scaler_pay.fit_transform(X_pay_train)
X_pay_test_scaled = scaler_pay.transform(X_pay_test)

models_pay = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}
if HAS_XGB:
    models_pay['XGBoost Regressor'] = XGBRegressor(random_state=42)

res_pay = {}
for name, m in models_pay.items():
    m.fit(X_pay_train_scaled, y_pay_train)
    preds = m.predict(X_pay_test_scaled)
    mae = mean_absolute_error(y_pay_test, preds)
    mse = mean_squared_error(y_pay_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_pay_test, preds)
    res_pay[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    print(f"[{name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

df_res_pay = pd.DataFrame(res_pay).T
df_res_pay.to_csv('models/df_payments_comparison.csv')

rf_pay_grid = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions={'n_estimators': [50, 100], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
    n_iter=4, cv=3, scoring='r2', random_state=42, n_jobs=1
)
rf_pay_grid.fit(X_pay_train_scaled, y_pay_train)
best_pay_model = rf_pay_grid.best_estimator_

with open('models/best_pay_model.pkl', 'wb') as f:
    pickle.dump(best_pay_model, f, protocol=4)
with open('models/scaler_pay.pkl', 'wb') as f:
    pickle.dump(scaler_pay, f, protocol=4)
with open('models/le_pay.pkl', 'wb') as f:
    pickle.dump(le_pay, f, protocol=4)
with open('models/features_pay.pkl', 'wb') as f:
    pickle.dump(list(X_pay.columns), f, protocol=4)

print("Saved Payments Model and Scaler successfully!")


# ==============================================================================
# 2. Pipeline for df1 (Cash Flow -> Target: net_cash_flow_egp)
# ==============================================================================
print("\n" + "="*60)
print("TRAINING PIPELINE FOR DF1 (DAILY CASH FLOW)")
print("="*60)

df_cash = pd.read_csv('synthetic_sme_daily_cash_flow.csv')
print("Daily Cash Flow shape:", df_cash.shape)

df_cash_encoded = df_cash.copy()
df_cash_encoded['date'] = pd.to_datetime(df_cash_encoded['date'])
df_cash_encoded['year'] = df_cash_encoded['date'].dt.year
df_cash_encoded['day'] = df_cash_encoded['date'].dt.day
leaky_cols = ['date', 'closing_cash_balance_egp', 'total_cash_inflow_egp', 'total_cash_outflow_egp']
df_cash_encoded.drop(columns=leaky_cols, inplace=True)

le_cash_bus = LabelEncoder()
df_cash_encoded['business_id'] = le_cash_bus.fit_transform(df_cash_encoded['business_id'])
le_cash_sec = LabelEncoder()
df_cash_encoded['sector'] = le_cash_sec.fit_transform(df_cash_encoded['sector'])

X_cash = df_cash_encoded.drop(columns=['net_cash_flow_egp'])
y_cash = df_cash_encoded['net_cash_flow_egp']

X_cash_train, X_cash_test, y_cash_train, y_cash_test = train_test_split(
    X_cash, y_cash, test_size=0.2, random_state=42
)

scaler_cash = StandardScaler()
X_cash_train_scaled = scaler_cash.fit_transform(X_cash_train)
X_cash_test_scaled = scaler_cash.transform(X_cash_test)

models_cash = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}
if HAS_XGB:
    models_cash['XGBoost Regressor'] = XGBRegressor(random_state=42)

res_cash = {}
for name, m in models_cash.items():
    m.fit(X_cash_train_scaled, y_cash_train)
    preds = m.predict(X_cash_test_scaled)
    mae = mean_absolute_error(y_cash_test, preds)
    mse = mean_squared_error(y_cash_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_cash_test, preds)
    res_cash[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    print(f"[{name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

df_res_cash = pd.DataFrame(res_cash).T
df_res_cash.to_csv('models/df1_cashflow_comparison.csv')

rf_cash_grid = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions={'n_estimators': [50, 100], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
    n_iter=4, cv=3, scoring='r2', random_state=42, n_jobs=1
)
rf_cash_grid.fit(X_cash_train_scaled, y_cash_train)
best_cash_model = rf_cash_grid.best_estimator_

with open('models/best_cash_model.pkl', 'wb') as f:
    pickle.dump(best_cash_model, f, protocol=4)
with open('models/scaler_cash.pkl', 'wb') as f:
    pickle.dump(scaler_cash, f, protocol=4)
with open('models/le_cash_bus.pkl', 'wb') as f:
    pickle.dump(le_cash_bus, f, protocol=4)
with open('models/le_cash_sec.pkl', 'wb') as f:
    pickle.dump(le_cash_sec, f, protocol=4)
with open('models/features_cash.pkl', 'wb') as f:
    pickle.dump(list(X_cash.columns), f, protocol=4)

print("Saved Cash Flow Model and Scaler successfully!")


# ==============================================================================
# 3. Pipeline for df2 (Expenses -> Target: amount_egp)
# ==============================================================================
print("\n" + "="*60)
print("TRAINING PIPELINE FOR DF2 (EXPENSES)")
print("="*60)

df_exp = pd.read_csv('synthetic_sme_expenses.csv')
print("Expenses shape:", df_exp.shape)

df_exp_encoded = df_exp.copy()
df_exp_encoded['date'] = pd.to_datetime(df_exp_encoded['date'])
df_exp_encoded['year'] = df_exp_encoded['date'].dt.year
df_exp_encoded['month'] = df_exp_encoded['date'].dt.month
df_exp_encoded['day'] = df_exp_encoded['date'].dt.day
df_exp_encoded.drop(columns=['date'], inplace=True)

le_exp_bus = LabelEncoder()
df_exp_encoded['business_id'] = le_exp_bus.fit_transform(df_exp_encoded['business_id'])
le_exp_sec = LabelEncoder()
df_exp_encoded['sector'] = le_exp_sec.fit_transform(df_exp_encoded['sector'])
le_exp_type = LabelEncoder()
df_exp_encoded['expense_type'] = le_exp_type.fit_transform(df_exp_encoded['expense_type'])

X_exp = df_exp_encoded.drop(columns=['amount_egp'])
y_exp = df_exp_encoded['amount_egp']

X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
    X_exp, y_exp, test_size=0.2, random_state=42
)

scaler_exp = StandardScaler()
X_exp_train_scaled = scaler_exp.fit_transform(X_exp_train)
X_exp_test_scaled = scaler_exp.transform(X_exp_test)

models_exp = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}
if HAS_XGB:
    models_exp['XGBoost Regressor'] = XGBRegressor(random_state=42)

res_exp = {}
for name, m in models_exp.items():
    m.fit(X_exp_train_scaled, y_exp_train)
    preds = m.predict(X_exp_test_scaled)
    mae = mean_absolute_error(y_exp_test, preds)
    mse = mean_squared_error(y_exp_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_exp_test, preds)
    res_exp[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    print(f"[{name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

df_res_exp = pd.DataFrame(res_exp).T
df_res_exp.to_csv('models/df2_expenses_comparison.csv')

rf_exp_grid = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions={'n_estimators': [50, 100], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
    n_iter=4, cv=3, scoring='r2', random_state=42, n_jobs=1
)
rf_exp_grid.fit(X_exp_train_scaled, y_exp_train)
best_exp_model = rf_exp_grid.best_estimator_

with open('models/best_exp_model.pkl', 'wb') as f:
    pickle.dump(best_exp_model, f, protocol=4)
with open('models/scaler_exp.pkl', 'wb') as f:
    pickle.dump(scaler_exp, f, protocol=4)
with open('models/le_exp_bus.pkl', 'wb') as f:
    pickle.dump(le_exp_bus, f, protocol=4)
with open('models/le_exp_sec.pkl', 'wb') as f:
    pickle.dump(le_exp_sec, f, protocol=4)
with open('models/le_exp_type.pkl', 'wb') as f:
    pickle.dump(le_exp_type, f, protocol=4)
with open('models/features_exp.pkl', 'wb') as f:
    pickle.dump(list(X_exp.columns), f, protocol=4)

with open('best_xgb_model.pkl', 'wb') as f:
    pickle.dump(best_exp_model, f, protocol=4)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler_exp, f, protocol=4)

print("Saved Expenses Model and Scaler successfully!")
print("\nALL 3 PIPELINES SUCCESSFULLY TRAINED AND SAVED WITH PROTOCOL 4!")
