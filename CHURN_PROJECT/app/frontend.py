
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Churn Prediction", page_icon="📉")

st.title("📉 Churn Prediction")
st.write("This app uses a trained machine learning model to predict whether a customer is likely to churn.")

# ---------- Load trained pipeline ----------

models_path = Path('./models/churn_pipeline.pkl').resolve().parent.parent
churn_pipeline = joblib.load(models_path)

# ---------- Input form ----------
st.subheader("Customer details")

with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    submitted = st.form_submit_button("Predict churn")

# ---------- Prediction ----------
if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    df = pd.DataFrame([payload])
    prob = churn_pipeline.predict_proba(df)[0, 1]
    pred = int(prob >= 0.5)
    prob_percent = prob * 100

    st.subheader("🔍 Prediction Result")

    # Probability card
    st.markdown(
        f"""
        <div style="
            background-color:#1E1E1E;
            padding:20px;
            border-radius:10px;
            margin-top:20px;
            border: 1px solid #444;
        ">
            <h2 style="color:white; margin-bottom:10px;">📊 Churn Probability</h2>
            <h1 style="color:#4CAF50; font-size:48px; margin-top:-10px;">
                {prob_percent:.2f}%
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    if pred == 1:
        st.markdown(
            """
            <div style="
                background-color:#3B0000;
                padding:20px;
                border-radius:10px;
                margin-top:10px;
                border: 1px solid #660000;
            ">
                <h2 style="color:#FF6666;">⚠️ Customer is LIKELY to CHURN</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                background-color:#002B0A;
                padding:20px;
                border-radius:10px;
                margin-top:10px;
                border: 1px solid #004D1A;
            ">
                <h2 style="color:#66FF99;">✅ Customer is likely to STAY</h2>
            </div>
            """,
            unsafe_allow_html=True
        )


