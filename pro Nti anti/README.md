# 💼 SME Financial Intelligence & Cash Flow Copilot
### مشروع الذكاء الاصطناعي للتنبؤ المالي للشركات الصغيرة والمتوسطة (NTI Project)

يوفر هذا المشروع خط إنتاج متكامل (End-to-End ML Pipeline) يغطي **كافة مجموعات البيانات الأربعة** (الانحدار والتصنيف) مع خادم **FastAPI** وتطبيق واجهة ويب تفاعلي باستخدام **Streamlit**.

---

## 📁 هيكل المشروع الكامل (Full Project Structure):

```text
c:\el D\ML\pro Nti anti\
├── df.ipynb                       # نوت بوك تأخير المدفوعات (df Payments - Regression)
├── df1.ipynb                      # نوت بوك التدفق النقدي اليومي (df1 Cash Flow - Regression)
├── df2.ipynb                      # نوت بوك نفقات ومصاريف الشركات (df2 Expenses - Regression)
├── df3.ipynb                      # نوت بوك تصنيف الفواتير (df3 Invoices - Binary Classification)
├── app.py                         # خادم FastAPI المحدث مع 4 Endpoints
├── streamlit_app.py               # تطبيق Streamlit المتكامل (تنبؤ تفاعلي + EDA + تقييم)
├── train_all.py                   # كود تدريب وحفظ نماذج الانحدار (df, df1, df2)
├── train_df3.py                   # كود تدريب وحفظ نموذج التصنيف (df3 is_late)
├── test_api.py                    # سكريبت اختبار شامل لنقاط اتصال FastAPI
├── requirements.txt               # متطلبات الحزم والمكتبات
├── .streamlit/
│   └── config.toml                # إعدادات مظهر وسيرفر Streamlit
├── models/                        # مجلد النماذج والمقاييس المحفوظة
│   ├── best_pay_model.pkl         # نموذج المدفوعات (payment_delay_days)
│   ├── best_cash_model.pkl        # نموذج التدفق النقدي (net_cash_flow_egp)
│   ├── best_exp_model.pkl         # نموذج النفقات (amount_egp)
│   ├── best_inv_model.pkl         # نموذج تصنيف الفواتير (is_late)
│   ├── scaler_*.pkl & le_*.pkl    # معايير التحجيم ومحولات النصوص لكافة النماذج
│   └── df*_comparison.csv         # جداول مقارنة مقاييس النماذج لكل مهمة
└── data/                          # ملفات البيانات الخام (CSV)
```

---

## 🎯 ملخص النماذج والمجموعات الأربعة (The 4 Targets):

| الرمز | مجموعة البيانات | الهدف المتوقع (Target) | نوع المهمة | أفضل خوارزمية | المقياس الأساسي |
|---|---|---|---|---|---|
| **df** | Payments | `payment_delay_days` (أيام التأخير) | Regression | Random Forest | MAE: ~6.09 |
| **df1** | Daily Cash Flow | `net_cash_flow_egp` (صافي التدفق) | Regression | Random Forest / XGBoost | $R^2 \approx 0.98$ |
| **df2** | Expenses | `amount_egp` (قيمة المصروف) | Regression | Random Forest / XGBoost | $R^2 \approx 0.68$ |
| **df3** | Invoices | `is_late` (0: موعد / 1: متأخر) | **Binary Classification** | Gradient Boosting / RF | ROC-AUC: ~0.60 |

---

## 🚀 طريقة التشغيل (How to Run):

### 1. تشغيل خادم FastAPI:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
- التوثيق التفاعلي للـ API متاح على: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **نقاط الاتصال المتاحة:**
  - `POST /predict/invoices` (أو `/predict/late`): لتصنيف احتمالية تأخر الفاتورة.
  - `POST /predict/expenses` (أو `/predict`): لتوقع قيمة المصروف.
  - `POST /predict/cashflow`: لتوقع صافي التدفق النقدي.
  - `POST /predict/payments`: لتوقع أيام تأخير المدفوعة.

### 2. تشغيل واجهة Streamlit:
```bash
streamlit run streamlit_app.py
```
- تفتح تلقائياً على: [http://localhost:8501](http://localhost:8501)
- تحتوي على تبويبات مخصصة لكل نموذج، مع حساب الفائض والعجز المالي، ومؤشر احتمالية التعثر للفواتير، ورسوم بيانية تفاعلية واستكشاف للبيانات، بالإضافة إلى أداة اختبار حي لخادم FastAPI.
