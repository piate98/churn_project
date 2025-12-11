

import streamlit as st
import requests

API_URL_ROOT = "http://127.0.0.1:8000/"
API_URL_PREDICT = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Churn Prediction", page_icon="📉")

st.title("📉 Churn Prediction ")
st.write("This UI calls your **FastAPI backend (main.py)** and shows the results.")

# ---- Section 1: Check backend status ----
st.subheader("1. Check API status")

if st.button("Ping API"):
    try:
        resp = requests.get(API_URL_ROOT)
        st.write("Status code:", resp.status_code)
        st.json(resp.json())
    except Exception as e:
        st.error(f"Could not reach API: {e}")

st.markdown("---")

# ---- Section 2: Prediction form ----
st.subheader("2. Predict customer churn")

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

    try:
        resp = requests.post(API_URL_PREDICT, json=payload)

        if resp.status_code == 200:
            data = resp.json()
            prob = data.get("churn_probability", 0.0)
            pred = data.get("churn_predicted", 0)

            st.subheader("🔍 Prediction Result")

            # Format probability as a percentage
            prob_percent = prob * 100

            # Display in a nicer card layout
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

            st.markdown(" ")

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

        else:
            st.error(f"API returned error {resp.status_code}")
            st.text(resp.text)

    except Exception as e:
        st.error(f"Could not contact API: {e}")
