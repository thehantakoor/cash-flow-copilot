import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.makedirs('models', exist_ok=True)

print("="*60)
print("TRAINING CLASSIFICATION PIPELINE FOR DF3 (INVOICES -> is_late)")
print("="*60)

df_inv = pd.read_csv('synthetic_sme_invoices.csv')

# Feature Engineering
df_inv_clean = df_inv.copy()
df_inv_clean['invoice_date'] = pd.to_datetime(df_inv_clean['invoice_date'])
df_inv_clean['due_date'] = pd.to_datetime(df_inv_clean['due_date'])
df_inv_clean['terms_days'] = (df_inv_clean['due_date'] - df_inv_clean['invoice_date']).dt.days

df_inv_clean['year'] = df_inv_clean['invoice_date'].dt.year
df_inv_clean['month'] = df_inv_clean['invoice_date'].dt.month
df_inv_clean['day'] = df_inv_clean['invoice_date'].dt.day
df_inv_clean['day_of_week'] = df_inv_clean['invoice_date'].dt.dayofweek

# Encode Categorical
le_bus = LabelEncoder()
df_inv_clean['business_id'] = le_bus.fit_transform(df_inv_clean['business_id'])
le_cust = LabelEncoder()
df_inv_clean['customer_id'] = le_cust.fit_transform(df_inv_clean['customer_id'])

# Target & Features
features = ['business_id', 'customer_id', 'invoice_amount_egp', 'terms_days', 'year', 'month', 'day', 'day_of_week']
X = df_inv_clean[features]
y = df_inv_clean['is_late']

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
if HAS_XGB:
    models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')

res = {}
print("\n=== Model Evaluation (Classification) ===")
for name, m in models.items():
    m.fit(X_train_scaled, y_train)
    preds = m.predict(X_test_scaled)
    probs = m.predict_proba(X_test_scaled)[:, 1] if hasattr(m, 'predict_proba') else preds

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    res[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC AUC': auc
    }
    print(f"[{name}] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

df_res = pd.DataFrame(res).T
df_res.to_csv('models/df3_invoices_comparison.csv')

# Hyperparameter Fine-Tuning Gradient Boosting
rf_grid = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions={'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
    n_iter=4, cv=3, scoring='roc_auc', random_state=42, n_jobs=1
)
rf_grid.fit(X_train_scaled, y_train)
best_model = rf_grid.best_estimator_

# Save artifacts with protocol 4 for full backward compatibility
with open('models/best_inv_model.pkl', 'wb') as f:
    pickle.dump(best_model, f, protocol=4)
with open('models/scaler_inv.pkl', 'wb') as f:
    pickle.dump(scaler, f, protocol=4)
with open('models/le_inv_bus.pkl', 'wb') as f:
    pickle.dump(le_bus, f, protocol=4)
with open('models/le_inv_cust.pkl', 'wb') as f:
    pickle.dump(le_cust, f, protocol=4)
with open('models/features_inv.pkl', 'wb') as f:
    pickle.dump(features, f, protocol=4)

print("\nSaved Invoices Model (best_inv_model.pkl) with protocol 4 successfully!")
