import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Banking Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# LOAD MODEL & SCALER
# =========================================================

try:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")

except Exception as e:
    st.error(f"❌ Error Loading Model Files: {e}")
    st.info("Make sure fraud_model.pkl and scaler.pkl are inside your project folder.")
    st.stop()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #F5F7FA;
}

.title {
    text-align: center;
    color: #1565C0;
    font-size: 45px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 20px;
}

.stButton > button {
    background-color: #1565C0;
    color: white;
    height: 55px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #0D47A1;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    "<div class='title'>🏦 AI Banking Fraud Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Detect Fraudulent Banking Transactions using Machine Learning</div>",
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# INPUT SECTION
# =========================================================

col1, col2, col3 = st.columns(3)

# =========================================================
# COLUMN 1
# =========================================================

with col1:

    transaction_amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=5000.0
    )

    login_attempts = st.slider(
        "🔐 Login Attempts",
        0,
        20,
        1
    )

    device_risk_score = st.slider(
        "📱 Device Risk Score",
        0.0,
        100.0,
        10.0
    )

    transfer_frequency = st.slider(
        "🔁 Transfer Frequency",
        0,
        50,
        2
    )

    anomaly_score = st.slider(
        "⚠️ Anomaly Score",
        0.0,
        100.0,
        10.0
    )

# =========================================================
# COLUMN 2
# =========================================================

with col2:

    account_age_days = st.number_input(
        "📅 Account Age Days",
        min_value=0,
        value=365
    )

    transaction_time_hour = st.slider(
        "⏰ Transaction Hour",
        0,
        23,
        12
    )

    failed_transactions_last_30d = st.slider(
        "❌ Failed Transactions (30 Days)",
        0,
        20,
        0
    )

    avg_monthly_balance = st.number_input(
        "🏦 Average Monthly Balance",
        min_value=0.0,
        value=50000.0
    )

    daily_transaction_count = st.slider(
        "📊 Daily Transaction Count",
        0,
        100,
        5
    )

# =========================================================
# COLUMN 3
# =========================================================

with col3:

    geo_distance_km = st.number_input(
        "🌍 Geo Distance (KM)",
        min_value=0.0,
        value=10.0
    )

    session_duration_minutes = st.number_input(
        "⌛ Session Duration",
        min_value=0.0,
        value=5.0
    )

    transaction_velocity_score = st.slider(
        "🚀 Transaction Velocity Score",
        0.0,
        100.0,
        10.0
    )

    payment_channel = st.selectbox(
        "💳 Payment Channel",
        ["ATM", "Mobile App", "Web Banking"]
    )

    authentication_type = st.selectbox(
        "🛡 Authentication Type",
        ["OTP", "Biometric", "Password"]
    )

# =========================================================
# FLAGS SECTION
# =========================================================

st.subheader("🚩 Transaction Flags")

flag1, flag2, flag3 = st.columns(3)

with flag1:

    card_present_flag = st.selectbox(
        "💳 Card Present",
        ["No", "Yes"]
    )

with flag2:

    international_transaction_flag = st.selectbox(
        "🌐 International Transaction",
        ["No", "Yes"]
    )

with flag3:

    suspicious_ip_flag = st.selectbox(
        "🧠 Suspicious IP",
        ["No", "Yes"]
    )

st.divider()

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("🔍 Predict Fraud"):

    try:

        # =================================================
        # MANUAL ENCODING
        # =================================================

        payment_channel_map = {
            "ATM": 0,
            "Mobile App": 1,
            "Web Banking": 2
        }

        authentication_type_map = {
            "OTP": 0,
            "Biometric": 1,
            "Password": 2
        }

        yes_no_map = {
            "No": 0,
            "Yes": 1
        }

        # =================================================
        # ENCODE VALUES
        # =================================================

        payment_channel_encoded = payment_channel_map[payment_channel]

        authentication_type_encoded = authentication_type_map[
            authentication_type
        ]

        card_present_encoded = yes_no_map[
            card_present_flag
        ]

        international_encoded = yes_no_map[
            international_transaction_flag
        ]

        suspicious_encoded = yes_no_map[
            suspicious_ip_flag
        ]

        # =================================================
        # CREATE INPUT DATAFRAME
        # =================================================

        columns = [
            "transaction_amount",
            "login_attempts",
            "device_risk_score",
            "transfer_frequency",
            "anomaly_score",
            "account_age_days",
            "transaction_time_hour",
            "failed_transactions_last_30d",
            "avg_monthly_balance",
            "daily_transaction_count",
            "geo_distance_km",
            "session_duration_minutes",
            "transaction_velocity_score",
            "payment_channel",
            "authentication_type",
            "card_present_flag",
            "international_transaction_flag",
            "suspicious_ip_flag"
        ]

        values = [[
            transaction_amount,
            login_attempts,
            device_risk_score,
            transfer_frequency,
            anomaly_score,
            account_age_days,
            transaction_time_hour,
            failed_transactions_last_30d,
            avg_monthly_balance,
            daily_transaction_count,
            geo_distance_km,
            session_duration_minutes,
            transaction_velocity_score,
            payment_channel_encoded,
            authentication_type_encoded,
            card_present_encoded,
            international_encoded,
            suspicious_encoded
        ]]

        input_data = pd.DataFrame(values, columns=columns)

        # =================================================
        # SCALE DATA
        # =================================================

        input_scaled = scaler.transform(input_data)

        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(input_scaled)[0]

        # =================================================
        # PROBABILITY
        # =================================================

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                input_scaled
            )[0][1]

        else:

            # fallback probability
            probability = 0.50

        # =================================================
        # CUSTOM RISK ANALYSIS
        # =================================================

        risk_score = (
            transaction_amount / 2000 +
            login_attempts * 2 +
            device_risk_score +
            anomaly_score +
            failed_transactions_last_30d * 3 +
            transaction_velocity_score
        ) / 6

        # =================================================
        # BOOST FRAUD PROBABILITY
        # =================================================

        if risk_score > 80:
            probability = max(probability, 0.92)

        elif risk_score > 70:
            probability = max(probability, 0.75)

        elif risk_score > 50:
            probability = max(probability, 0.45)

        # =================================================
        # FINAL PREDICTION
        # =================================================

        if probability >= 0.5:
            prediction = 1
        else:
            prediction = 0

        st.divider()

        # =================================================
        # RESULT SECTION
        # =================================================

        if prediction == 1:

            st.error("⚠️ FRAUDULENT TRANSACTION DETECTED")

        else:

            st.success("✅ LEGITIMATE TRANSACTION")

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        # =================================================
        # RISK ANALYSIS
        # =================================================

        st.subheader("📈 Risk Analysis")

        risk_percent = min(int(risk_score), 100)

        st.progress(risk_percent)

        st.write(
            f"Overall Risk Score: {risk_score:.2f}/100"
        )

    except Exception as e:

        st.error(f"❌ Prediction Error: {e}")