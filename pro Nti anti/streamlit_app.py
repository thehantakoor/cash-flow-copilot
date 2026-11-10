import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="SME Financial Intelligence AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value-green {
        font-size: 2rem;
        font-weight: bold;
        color: #059669;
    }
    .metric-value-red {
        font-size: 2rem;
        font-weight: bold;
        color: #DC2626;
    }
    .metric-value-yellow {
        font-size: 2rem;
        font-weight: bold;
        color: #D97706;
    }
    .metric-value-blue {
        font-size: 2rem;
        font-weight: bold;
        color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Cache Artifacts Loading
# ==============================================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_pkl(filename, fallback_path=None):
    p = os.path.join(MODELS_DIR, filename)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    if fallback_path and os.path.exists(fallback_path):
        with open(fallback_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_all_models():
    # 1. Expenses (df2)
    m_exp = load_pkl("best_exp_model.pkl", "best_xgb_model.pkl")
    s_exp = load_pkl("scaler_exp.pkl", "scaler.pkl")
    le_exp_bus = load_pkl("le_exp_bus.pkl")
    le_exp_sec = load_pkl("le_exp_sec.pkl")
    le_exp_type = load_pkl("le_exp_type.pkl")
    feat_exp = load_pkl("features_exp.pkl")

    # 2. Cash Flow (df1)
    m_cash = load_pkl("best_cash_model.pkl")
    s_cash = load_pkl("scaler_cash.pkl")
    le_cash_bus = load_pkl("le_cash_bus.pkl")
    le_cash_sec = load_pkl("le_cash_sec.pkl")
    feat_cash = load_pkl("features_cash.pkl")

    # 3. Payments (df)
    m_pay = load_pkl("best_pay_model.pkl")
    s_pay = load_pkl("scaler_pay.pkl")
    le_pay = load_pkl("le_pay.pkl")
    feat_pay = load_pkl("features_pay.pkl")

    # 4. Invoices (df3 - Classification)
    m_inv = load_pkl("best_inv_model.pkl")
    s_inv = load_pkl("scaler_inv.pkl")
    le_inv_bus = load_pkl("le_inv_bus.pkl")
    le_inv_cust = load_pkl("le_inv_cust.pkl")
    feat_inv = load_pkl("features_inv.pkl")

    return {
        "exp": (m_exp, s_exp, le_exp_bus, le_exp_sec, le_exp_type, feat_exp),
        "cash": (m_cash, s_cash, le_cash_bus, le_cash_sec, feat_cash),
        "pay": (m_pay, s_pay, le_pay, feat_pay),
        "inv": (m_inv, s_inv, le_inv_bus, le_inv_cust, feat_inv)
    }

models_dict = load_all_models()

@st.cache_data
def load_sample_datasets():
    data = {}
    if os.path.exists("synthetic_sme_daily_cash_flow.csv"):
        data["df1"] = pd.read_csv("synthetic_sme_daily_cash_flow.csv")
    if os.path.exists("synthetic_sme_payments.csv"):
        data["df"] = pd.read_csv("synthetic_sme_payments.csv")
    if os.path.exists("synthetic_sme_expenses.csv"):
        data["df2"] = pd.read_csv("synthetic_sme_expenses.csv")
    if os.path.exists("synthetic_sme_invoices.csv"):
        data["df3"] = pd.read_csv("synthetic_sme_invoices.csv")
    return data

datasets = load_sample_datasets()

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=70)
    st.title("SME AI Copilot")
    st.markdown("**نظام الذكاء الاصطناعي للتنبؤ المالي**")
    st.markdown("---")

    page = st.radio(
        "اختر القسم / Select Module:",
        [
            "💰 نفقات الشركات (df2 Expenses)",
            "📈 التدفق النقدي (df1 Cash Flow)",
            "💳 تأخير المدفوعات (df Payments)",
            "📑 تصنيف الفواتير (df3 Invoices is_late)",
            "📊 استكشاف البيانات (EDA Insights)",
            "🤖 مقارنة أداء النماذج (Models Benchmark)",
            "🚀 اختبار الـ FastAPI (API Tester)",
            "🌐 دليل النشر والتشغيل (Deployment Guide)"
        ]
    )

    st.markdown("---")
    st.markdown("### ⚙️ إعدادات التنبؤ:")
    backend_mode = st.radio(
        "محرك التنبؤ / Prediction Engine:",
        ["الموديلات المحلية مباشرة (Direct In-Memory)", "عبر خادم FastAPI (API Client)"]
    )
    api_url = st.text_input("FastAPI Base URL:", value="http://127.0.0.1:8000")

    st.markdown("---")
    st.caption("Developed with ❤️ for NTI Machine Learning Project")


# Helper API caller
def predict_via_api(endpoint, payload):
    try:
        url = f"{api_url.rstrip('/')}{endpoint}"
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"API Error ({resp.status_code}): {resp.text}"
    except Exception as ex:
        return None, f"Connection Failed to {api_url}: {str(ex)}"


# ==============================================================================
# PAGE 1: EXPENSES PREDICTOR (df2)
# ==============================================================================
if page == "💰 نفقات الشركات (df2 Expenses)":
    st.markdown('<p class="main-header">💰 التنبؤ بنفقات ومصاريف الشركات (df2: Expenses)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">توقع قيمة المصروف المالي بالجنيه المصري (amount_egp) بناءً على هوية الشركة والقطاع ونوع المصروف والتاريخ.</p>', unsafe_allow_html=True)

    m_exp, s_exp, le_exp_bus, le_exp_sec, le_exp_type, feat_exp = models_dict["exp"]

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📝 إدخال بيانات المصروف:")
        bus_opts = list(le_exp_bus.classes_) if le_exp_bus else [f"SME_00{i}" for i in range(1, 9)]
        sec_opts = list(le_exp_sec.classes_) if le_exp_sec else ["Wholesale", "Retail", "Services", "Distribution", "Ecommerce"]
        type_opts = list(le_exp_type.classes_) if le_exp_type else ["supplier_payments_egp", "payroll_egp", "rent_egp", "taxes_egp", "loan_payments_egp"]

        business_id = st.selectbox("هوية الشركة (Business ID):", bus_opts, index=0)
        sector = st.selectbox("القطاع التجاري (Sector):", sec_opts, index=0)
        expense_type = st.selectbox("نوع المصروف (Expense Type):", type_opts, index=0)
        selected_date = st.date_input("تاريخ المصروف (Expense Date):", datetime.date(2024, 6, 15))

        predict_btn = st.button("🔮 احسب قيمة المصروف المتوقعة", type="primary", use_container_width=True)

    with col2:
        st.subheader("🎯 النتيجة والتوقع:")
        if predict_btn:
            payload = {
                "business_id": business_id,
                "sector": sector,
                "expense_type": expense_type,
                "date_year": selected_date.year,
                "date_month": selected_date.month,
                "date_day": selected_date.day
            }

            if backend_mode == "عبر خادم FastAPI (API Client)":
                res, err = predict_via_api("/predict/expenses", payload)
                if err:
                    st.error(f"خطأ في الاتصال بالـ API:\n{err}\nجاري التبديل للموديل الداخلي مباشرة...")
                    bus_enc = int(le_exp_bus.transform([business_id])[0])
                    sec_enc = int(le_exp_sec.transform([sector])[0])
                    type_enc = int(le_exp_type.transform([expense_type])[0])
                    inp_df = pd.DataFrame([{
                        'business_id': bus_enc, 'sector': sec_enc, 'expense_type': type_enc,
                        'year': selected_date.year, 'month': selected_date.month, 'day': selected_date.day
                    }])[feat_exp]
                    scaled = s_exp.transform(inp_df)
                    pred = float(m_exp.predict(scaled)[0])
                else:
                    pred = res["predicted_amount_egp"]
            else:
                bus_enc = int(le_exp_bus.transform([business_id])[0])
                sec_enc = int(le_exp_sec.transform([sector])[0])
                type_enc = int(le_exp_type.transform([expense_type])[0])
                inp_df = pd.DataFrame([{
                    'business_id': bus_enc, 'sector': sec_enc, 'expense_type': type_enc,
                    'year': selected_date.year, 'month': selected_date.month, 'day': selected_date.day
                }])[feat_exp]
                scaled = s_exp.transform(inp_df)
                pred = float(m_exp.predict(scaled)[0])

            st.markdown(f"""
            <div class="metric-card">
                <h3>القيمة التقديرية للمصروف:</h3>
                <p class="metric-value-blue">{pred:,.2f} ج.م (EGP)</p>
                <p><strong>القطاع:</strong> {sector} | <strong>الشركة:</strong> {business_id}</p>
                <p><strong>نوع المصروف:</strong> {expense_type}</p>
                <p><strong>التاريخ المستهدف:</strong> {selected_date}</p>
            </div>
            """, unsafe_allow_html=True)
            st.success("تم التنبؤ بنجاح!")
        else:
            st.info("قم بضبط المدخلات في العمود الأيسر واضغط على 'احسب قيمة المصروف المتوقعة'.")


# ==============================================================================
# PAGE 2: CASH FLOW PREDICTOR (df1)
# ==============================================================================
elif page == "📈 التدفق النقدي (df1 Cash Flow)":
    st.markdown('<p class="main-header">📈 التنبؤ بصافي التدفق النقدي اليومي (df1: Daily Cash Flow)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">توقع صافي التدفق المالي (net_cash_flow_egp) لمعرفة ما إذا كانت الشركة ستحقق فائضاً نقدياً أو عجزاً تشغيلياً.</p>', unsafe_allow_html=True)

    m_cash, s_cash, le_cash_bus, le_cash_sec, feat_cash = models_dict["cash"]

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🏢 معلومات المنشأة والتاريخ:")
        c1, c2 = st.columns(2)
        bus_opts = list(le_cash_bus.classes_) if le_cash_bus else [f"SME_00{i}" for i in range(1, 9)]
        sec_opts = list(le_cash_sec.classes_) if le_cash_sec else ["Wholesale", "Retail", "Services", "Distribution", "Ecommerce"]

        with c1:
            business_id = st.selectbox("هوية الشركة:", bus_opts, index=0)
            selected_date = st.date_input("التاريخ المستهدف:", datetime.date(2024, 6, 15))
        with c2:
            sector = st.selectbox("القطاع:", sec_opts, index=0)
            day_of_week = st.selectbox("يوم الأسبوع:", ["الإثنين (0)", "الثلاثاء (1)", "الأربعاء (2)", "الخميس (3)", "الجمعة (4)", "السبت (5)", "الأحد (6)"], index=2)
            dow_idx = int(day_of_week.split("(")[1].replace(")", ""))

        is_weekend = 1 if dow_idx in [4, 5] else 0
        promotion_flag = st.checkbox("يوجد حملة ترويجية / تخفيضات (Promotion)", value=False)

        st.subheader("💵 المدخلات النقدية التشغيلية (EGP):")
        k1, k2 = st.columns(2)
        with k1:
            sales = st.number_input("إجمالي المبيعات (Sales):", min_value=0.0, value=25000.0, step=1000.0)
            cash_sales = st.number_input("المبيعات النقدية الفورية:", min_value=0.0, value=12000.0, step=500.0)
            rec_created = st.number_input("ذمم مدينة تم إنشاؤها (آجل):", min_value=0.0, value=13000.0, step=500.0)
            rec_collected = st.number_input("تحصيلات الذمم المدينة (كاش وارد):", min_value=0.0, value=10000.0, step=500.0)
            other_inc = st.number_input("إيرادات أخرى متنوعة:", min_value=0.0, value=500.0, step=100.0)
        with k2:
            pay_created = st.number_input("ذمم موردين مستحقة:", min_value=0.0, value=14000.0, step=500.0)
            supp_payments = st.number_input("مدفوعات سداد للموردين:", min_value=0.0, value=11000.0, step=500.0)
            payroll = st.number_input("رواتب وأجور (Payroll):", min_value=0.0, value=3000.0, step=500.0)
            rent = st.number_input("إيجار المقرات (Rent):", min_value=0.0, value=0.0, step=500.0)
            taxes = st.number_input("ضرائب ورسوم حكومية:", min_value=0.0, value=0.0, step=500.0)
            loans = st.number_input("أقساط قروض بنكية:", min_value=0.0, value=0.0, step=500.0)

        predict_cash_btn = st.button("🔮 احسب صافي التدفق النقدي", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 النتيجة والتحليل المالي:")
        if predict_cash_btn:
            bus_enc = int(le_cash_bus.transform([business_id])[0])
            sec_enc = int(le_cash_sec.transform([sector])[0])

            inp_dict = {
                'business_id': bus_enc,
                'sector': sec_enc,
                'day_of_week': dow_idx,
                'month': selected_date.month,
                'day': selected_date.day,
                'year': selected_date.year,
                'is_weekend': is_weekend,
                'promotion_flag': 1 if promotion_flag else 0,
                'sales_egp': sales,
                'cash_sales_egp': cash_sales,
                'receivables_created_egp': rec_created,
                'receivables_collected_egp': rec_collected,
                'supplier_payables_created_egp': pay_created,
                'supplier_payments_egp': supp_payments,
                'payroll_egp': payroll,
                'rent_egp': rent,
                'taxes_egp': taxes,
                'loan_payments_egp': loans,
                'other_income_egp': other_inc
            }

            inp_df = pd.DataFrame([inp_dict])[feat_cash]
            scaled = s_cash.transform(inp_df)
            pred_cf = float(m_cash.predict(scaled)[0])

            is_positive = pred_cf >= 0
            val_color = "metric-value-green" if is_positive else "metric-value-red"
            status_text = "🟢 فائض نقدي (Surplus - سيولة ممتازة)" if is_positive else "🔴 عجز نقدي متوقع (Deficit - تنبيه سيولة)"

            st.markdown(f"""
            <div class="metric-card">
                <h3>صافي التدفق النقدي المتوقع (Net Cash Flow):</h3>
                <p class="{val_color}">{pred_cf:,.2f} ج.م (EGP)</p>
                <h4><strong>الحالة المالية:</strong> {status_text}</h4>
                <p><strong>القطاع:</strong> {sector} | <strong>الشركة:</strong> {business_id}</p>
                <p><strong>تاريخ التوقع:</strong> {selected_date} | <strong>عطلة نهاية أسبوع:</strong> {'نعم' if is_weekend else 'لا'}</p>
            </div>
            """, unsafe_allow_html=True)

            if is_positive:
                st.balloons()
                st.success("السيولة إيجابية وآمنة لتمويل العمليات التشغيلية اليومية!")
            else:
                st.warning("⚠️ تنبيه: يرجى التخطيط لتغطية العجز النقدي عبر تحصيل الذمم أو تأجيل بعض المدفوعات.")
        else:
            st.info("اضغط على زر 'احسب صافي التدفق النقدي' لعرض التقرير.")


# ==============================================================================
# PAGE 3: PAYMENTS PREDICTOR (df)
# ==============================================================================
elif page == "💳 تأخير المدفوعات (df Payments)":
    st.markdown('<p class="main-header">💳 التنبؤ بتأخير المدفوعات (df: Payments Delay)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">توقع عدد أيام التأخير المتوقعة في السداد (payment_delay_days) لتقييم مخاطر الائتمان والتحصيل.</p>', unsafe_allow_html=True)

    m_pay, s_pay, le_pay, feat_pay = models_dict["pay"]

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📝 تفاصيل المدفوعة:")
        bus_opts = list(le_pay.classes_) if le_pay else [f"SME_00{i}" for i in range(1, 9)]
        business_id = st.selectbox("هوية الشركة (Business ID):", bus_opts, index=0)
        amount = st.number_input("قيمة المدفوعة (Payment Amount EGP):", min_value=100.0, value=7500.0, step=500.0)
        pay_date = st.date_input("تاريخ المدفوعة / الاستحقاق:", datetime.date(2024, 6, 20))
        day_of_week = pay_date.weekday()

        predict_pay_btn = st.button("🔮 احسب أيام التأخير المتوقعة", type="primary", use_container_width=True)

    with col2:
        st.subheader("🎯 مؤشر الالتزام ومخاطر السداد:")
        if predict_pay_btn:
            bus_enc = int(le_pay.transform([business_id])[0])
            inp_df = pd.DataFrame([{
                'business_id': bus_enc,
                'amount_egp': amount,
                'year': pay_date.year,
                'month': pay_date.month,
                'day': pay_date.day,
                'day_of_week': day_of_week
            }])[feat_pay]

            scaled = s_pay.transform(inp_df)
            pred_days = max(0.0, float(m_pay.predict(scaled)[0]))

            if pred_days < 1.0:
                status_color = "metric-value-green"
                status_title = "🟢 سداد فوري في الموعد (On Time)"
                badge = "مخاطر منخفضة جداً"
            elif pred_days <= 7.0:
                status_color = "metric-value-blue"
                status_title = f"🟡 تأخير طفيف (~{int(round(pred_days))} أيام)"
                badge = "مخاطر منخفضة إلى متوسطة"
            else:
                status_color = "metric-value-red"
                status_title = f"🔴 تأخير ملحوظ (~{int(round(pred_days))} يوماً)"
                badge = "مخاطر ائتمانية مرتفعة"

            st.markdown(f"""
            <div class="metric-card">
                <h3>أيام التأخير المتوقعة:</h3>
                <p class="{status_color}">{pred_days:.1f} يوم (Days)</p>
                <h4><strong>تقييم الالتزام:</strong> {status_title}</h4>
                <p><strong>مستوى المخاطر:</strong> <span style="padding: 4px 8px; border-radius: 6px; background: #E5E7EB; font-weight: bold;">{badge}</span></p>
                <p><strong>الشركة:</strong> {business_id} | <strong>المبلغ:</strong> {amount:,.2f} ج.م</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("أدخل بيانات المدفوعة ثم اضغط 'احسب أيام التأخير المتوقعة'.")


# ==============================================================================
# PAGE 4: INVOICES LATE PREDICTOR (df3 - CLASSIFICATION)
# ==============================================================================
elif page == "📑 تصنيف الفواتير (df3 Invoices is_late)":
    st.markdown('<p class="main-header">📑 تصنيف فواتير العملاء واحتمالية التأخير (df3: Invoices is_late)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">مهمة تصنيف ثنائي (Binary Classification) للتنبؤ بما إذا كانت الفاتورة ستتأخر في السداد (is_late = 1) أم ستسدد في الموعد (is_late = 0) مع حساب نسبة المخاطرة (Probability Score).</p>', unsafe_allow_html=True)

    m_inv, s_inv, le_inv_bus, le_inv_cust, feat_inv = models_dict["inv"]

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🧾 إدخال تفاصيل الفاتورة وشروط الدفع:")
        bus_opts = list(le_inv_bus.classes_) if le_inv_bus else [f"SME_00{i}" for i in range(1, 9)]
        cust_opts = list(le_inv_cust.classes_) if le_inv_cust else [f"CUST_{i:03d}" for i in range(1, 151)]

        b1, b2 = st.columns(2)
        with b1:
            business_id = st.selectbox("الشركة المصدرة للفاتورة:", bus_opts, index=0)
            inv_date = st.date_input("تاريخ إصدار الفاتورة:", datetime.date(2024, 6, 1))
        with b2:
            customer_id = st.selectbox("هوية العميل (Customer ID):", cust_opts, index=43)
            due_date = st.date_input("تاريخ الاستحقاق (Due Date):", datetime.date(2024, 7, 1))

        terms_days = (due_date - inv_date).days
        if terms_days <= 0:
            st.warning("⚠️ تنبيه: تاريخ الاستحقاق يجب أن يكون بعد تاريخ الإصدار! تم تعيين فترة السداد إلى 30 يوماً.")
            terms_days = 30

        st.info(f"📅 فترة السداد الممنوحة للعميل (Payment Terms): **{terms_days} يوم**")

        inv_amount = st.number_input("قيمة الفاتورة (Invoice Amount EGP):", min_value=100.0, value=12500.0, step=1000.0)

        predict_inv_btn = st.button("🔮 فحص مخاطر الفاتورة واحتمال التأخير", type="primary", use_container_width=True)

    with col2:
        st.subheader("🎯 تقييم مخاطر السداد للفاتورة:")
        if predict_inv_btn:
            bus_enc = int(le_inv_bus.transform([business_id])[0])
            cust_enc = int(le_inv_cust.transform([customer_id])[0]) if customer_id in le_inv_cust.classes_ else 0

            inp_dict = {
                'business_id': bus_enc,
                'customer_id': cust_enc,
                'invoice_amount_egp': inv_amount,
                'terms_days': terms_days,
                'year': inv_date.year,
                'month': inv_date.month,
                'day': inv_date.day,
                'day_of_week': inv_date.weekday()
            }

            inp_df = pd.DataFrame([inp_dict])[feat_inv]
            scaled = s_inv.transform(inp_df)

            pred_class = int(m_inv.predict(scaled)[0])
            late_prob = float(m_inv.predict_proba(scaled)[0][1]) if hasattr(m_inv, "predict_proba") else float(pred_class)

            if late_prob >= 0.5 or pred_class == 1:
                val_color = "metric-value-red"
                status_title = "🔴 خطر تأخر مرتفع (Late Payment Predicted)"
                risk_badge = "مخاطر سداد مرتفعة (High Risk)"
                recommendation = "يوصى بطلب دفعة مقدمة أو تقليص فترة السداد أو التواصل الاستباقي مع العميل قبل موعد الاستحقاق."
            elif late_prob >= 0.25:
                val_color = "metric-value-yellow"
                status_title = "🟡 خطر تأخر متوسط (Moderate Risk)"
                risk_badge = "مخاطر متوسطة (Medium Risk)"
                recommendation = "يوصى بإرسال تذكير سداد آلي للعميل قبل 3 أيام من تاريخ الاستحقاق."
            else:
                val_color = "metric-value-green"
                status_title = "🟢 التزام متوقع في الموعد (On-Time Likely)"
                risk_badge = "سجل ائتماني ممتاز (Low Risk)"
                recommendation = "العميل يتمتع بدرجة التزام مرتفعة وشروط الفاتورة آمنة."

            st.markdown(f"""
            <div class="metric-card">
                <h3>احتمالية تأخر السداد (Late Probability):</h3>
                <p class="{val_color}">{late_prob * 100:.1f}%</p>
                <h4><strong>النتيجة والتصنيف:</strong> {status_title}</h4>
                <p><strong>مستوى المخاطرة:</strong> <span style="padding: 4px 8px; border-radius: 6px; background: #E5E7EB; font-weight: bold;">{risk_badge}</span></p>
                <hr>
                <p><strong>العميل:</strong> {customer_id} | <strong>الشركة المصدرة:</strong> {business_id}</p>
                <p><strong>قيمة الفاتورة:</strong> {inv_amount:,.2f} ج.م | <strong>شروط السداد:</strong> {terms_days} يوم</p>
                <p><strong>💡 توصية النظام:</strong> {recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

            # Progress gauge for probability
            st.markdown("**مؤشر احتمالية التأخير:**")
            st.progress(min(1.0, late_prob))
        else:
            st.info("أدخل بيانات الفاتورة ثم اضغط على زر 'فحص مخاطر الفاتورة واحتمال التأخير'.")


# ==============================================================================
# PAGE 5: EDA & DATA INSIGHTS
# ==============================================================================
elif page == "📊 استكشاف البيانات (EDA Insights)":
    st.markdown('<p class="main-header">📊 استكشاف البيانات وتحليلها (EDA & Visual Insights)</p>', unsafe_allow_html=True)

    dataset_choice = st.selectbox(
        "اختر مجموعة البيانات للاستكشاف:",
        [
            "df3: فواتير العملاء (Invoices is_late)",
            "df1: التدفق النقدي اليومي (Daily Cash Flow)",
            "df: المدفوعات (Payments)",
            "df2: النفقات (Expenses)"
        ]
    )

    if dataset_choice.startswith("df3") and "df3" in datasets:
        curr_df = datasets["df3"]
        target_col = "is_late"
    elif dataset_choice.startswith("df1") and "df1" in datasets:
        curr_df = datasets["df1"]
        target_col = "net_cash_flow_egp"
    elif dataset_choice.startswith("df:") and "df" in datasets:
        curr_df = datasets["df"]
        target_col = "payment_delay_days"
    else:
        curr_df = datasets.get("df2", pd.DataFrame())
        target_col = "amount_egp"

    st.write(f"**أبعاد البيانات (Shape):** `{curr_df.shape[0]:,}` صف × `{curr_df.shape[1]}` عمود | **عدد القيم المفقودة:** `{curr_df.isnull().sum().sum()}` | **التكرارات:** `{curr_df.duplicated().sum()}`")

    tab_preview, tab_stats, tab_plots = st.tabs(["📋 معاينة البيانات (Preview)", "📈 الإحصاءات الوصفية (Stats)", "📉 رسوم بيانية (Charts)"])

    with tab_preview:
        st.dataframe(curr_df.head(20), use_container_width=True)

    with tab_stats:
        st.dataframe(curr_df.describe().T, use_container_width=True)

    with tab_plots:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if target_col == "is_late":
            sns.countplot(data=curr_df, x=target_col, ax=axes[0], palette=['#10B981', '#EF4444'])
            axes[0].set_title("توزيع الهدف: is_late (0: في الموعد, 1: متأخر)")

            sns.boxplot(data=curr_df, x=target_col, y="invoice_amount_egp", ax=axes[1], palette=['#A7F3D0', '#FECACA'])
            axes[1].set_title("مبالغ الفواتير حسب حالة التأخير")
        else:
            sns.histplot(curr_df[target_col].dropna(), kde=True, ax=axes[0], color="royalblue")
            axes[0].set_title(f"توزيع الهدف: {target_col}")

            group_col = "sector" if "sector" in curr_df.columns else "business_id"
            sns.boxplot(data=curr_df, x=group_col, y=target_col, ax=axes[1], palette="Set2")
            axes[1].set_title(f"{target_col} حسب {group_col}")
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)


# ==============================================================================
# PAGE 6: MODELS BENCHMARK
# ==============================================================================
elif page == "🤖 مقارنة أداء النماذج (Models Benchmark)":
    st.markdown('<p class="main-header">🤖 مقارنة أداء النماذج والتقييم الشامل لكافة البيانات</p>', unsafe_allow_html=True)
    st.markdown("مقارنة نتائج تدريب خوارزميات الذكاء الاصطناعي عبر كافة مجموعات البيانات (df, df1, df2, df3):")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. نفقات الشركات (df2: Expenses - Regression)")
        p_exp = os.path.join(MODELS_DIR, "df2_expenses_comparison.csv")
        if os.path.exists(p_exp):
            df_comp = pd.read_csv(p_exp, index_col=0)
            st.dataframe(df_comp.style.highlight_max(subset=['R2'], color='#D1FAE5').highlight_min(subset=['MAE'], color='#D1FAE5'), use_container_width=True)

    with c2:
        st.subheader("2. التدفق النقدي (df1: Cash Flow - Regression)")
        p_cash = os.path.join(MODELS_DIR, "df1_cashflow_comparison.csv")
        if os.path.exists(p_cash):
            df_comp = pd.read_csv(p_cash, index_col=0)
            st.dataframe(df_comp.style.highlight_max(subset=['R2'], color='#D1FAE5').highlight_min(subset=['MAE'], color='#D1FAE5'), use_container_width=True)

    st.markdown("---")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("3. تأخير المدفوعات (df: Payments - Regression)")
        p_pay = os.path.join(MODELS_DIR, "df_payments_comparison.csv")
        if os.path.exists(p_pay):
            df_comp = pd.read_csv(p_pay, index_col=0)
            st.dataframe(df_comp.style.highlight_min(subset=['MAE'], color='#D1FAE5'), use_container_width=True)

    with c4:
        st.subheader("4. تصنيف الفواتير (df3: Invoices is_late - Classification)")
        p_inv = os.path.join(MODELS_DIR, "df3_invoices_comparison.csv")
        if os.path.exists(p_inv):
            df_comp = pd.read_csv(p_inv, index_col=0)
            st.dataframe(df_comp.style.highlight_max(subset=['ROC AUC', 'Accuracy'], color='#D1FAE5'), use_container_width=True)

    st.markdown("---")
    st.success("✅ تم اختيار النماذج الأفضل وتدريبها وحفظها في مجلد `models/` لضمان أعلى أداء واستجابة فورية.")


# ==============================================================================
# PAGE 7: FASTAPI TESTER
# ==============================================================================
elif page == "🚀 اختبار الـ FastAPI (API Tester)":
    st.markdown('<p class="main-header">🚀 اختبار خادم FastAPI التفاعلي</p>', unsafe_allow_html=True)
    st.markdown(f"**عنوان الخادم الحالي:** `{api_url}`")

    if st.button("📡 فحص الاتصال بالخادم (Ping Health Check)"):
        try:
            r = requests.get(f"{api_url}/", timeout=3)
            if r.status_code == 200:
                st.success("الخادم متصل ويعمل بنجاح! 🟢")
                st.json(r.json())
            else:
                st.error(f"استجاب الخادم بكود غير متوقع: {r.status_code}")
        except Exception as e:
            st.error(f"تعذر الاتصال بالخادم عند `{api_url}`.\nتأكد من تشغيل الخادم عبر: `uvicorn app:app --port 8000`")

    st.markdown("---")
    st.subheader("تجربة استدعاء Endpoint المباشر:")
    endpoint_choice = st.selectbox(
        "اختر الـ Endpoint:",
        [
            "/predict/invoices (df3 Classification is_late)",
            "/predict/expenses (df2 Regression)",
            "/predict/cashflow (df1 Regression)",
            "/predict/payments (df Regression)"
        ]
    )

    if "/predict/invoices" in endpoint_choice:
        sample_payload = {
            "business_id": "SME_001",
            "customer_id": "CUST_044",
            "invoice_amount_egp": 15000.0,
            "terms_days": 14,
            "year": 2024,
            "month": 6,
            "day": 15,
            "day_of_week": 2
        }
        ep = "/predict/invoices"
    elif "/predict/expenses" in endpoint_choice:
        sample_payload = {
            "business_id": "SME_001",
            "sector": "Wholesale",
            "expense_type": "supplier_payments_egp",
            "date_year": 2024,
            "date_month": 5,
            "date_day": 15
        }
        ep = "/predict/expenses"
    elif "/predict/cashflow" in endpoint_choice:
        sample_payload = {
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
        }
        ep = "/predict/cashflow"
    else:
        sample_payload = {
            "business_id": "SME_003",
            "amount_egp": 8500.0,
            "year": 2024,
            "month": 6,
            "day": 20,
            "day_of_week": 3
        }
        ep = "/predict/payments"

    st.markdown("**الـ Payload المرسل (JSON):**")
    st.json(sample_payload)

    if st.button("🚀 إرسال طلب POST للـ API"):
        res, err = predict_via_api(ep, sample_payload)
        if err:
            st.error(err)
        else:
            st.success("تم استلام الاستجابة بنجاح:")
            st.json(res)


# ==============================================================================
# PAGE 8: DEPLOYMENT GUIDE
# ==============================================================================
elif page == "🌐 دليل النشر والتشغيل (Deployment Guide)":
    st.markdown('<p class="main-header">🌐 دليل تشغيل ونشر المشروع (Deployment Guide)</p>', unsafe_allow_html=True)

    st.markdown("""
    ### 💻 1. التشغيل المحلي (Local Execution):

    #### أ) تشغيل خادم FastAPI:
    افتح نافذة Terminal ونفذ الأمر التالي:
    ```bash
    uvicorn app:app --host 127.0.0.1 --port 8000 --reload
    ```
    يمكنك تصفح توثيق الـ API التفاعلي (Swagger UI) على:
    `http://127.0.0.1:8000/docs`

    #### ب) تشغيل واجهة Streamlit:
    افتح نافذة Terminal أخرى ونفذ:
    ```bash
    streamlit run streamlit_app.py
    ```
    ستفتح الواجهة التفاعلية تلقائياً على:
    `http://localhost:8501`

    ---

    ### ☁️ 2. النشر السحابي (Cloud Deployment):

    #### خيار 1: النشر المجاني عبر Streamlit Community Cloud
    1. ارفع مجلد المشروع على **GitHub**.
    2. اذهب إلى [share.streamlit.io](https://share.streamlit.io).
    3. اربط حساب GitHub واختر المستودع (Repository).
    4. حدد الملف الرئيسي: `streamlit_app.py`.
    5. اضغط **Deploy**! سيعمل تطبيقك على رابط عام متاح للجميع في دقائق.
    """)
