from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "scripts"))

from model_utils import load_or_train_model, score_customer


st.set_page_config(
    page_title="Customer Churn Risk App",
    layout="wide",
)

st.markdown(
    """
    <style>
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid #dbeafe;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        }
        .risk-low {
            color: #15803d;
            font-weight: 700;
        }
        .risk-medium {
            color: #b45309;
            font-weight: 700;
        }
        .risk-high {
            color: #b91c1c;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=True)
def get_model_bundle():
    return load_or_train_model()


def risk_class_name(risk_level: str) -> str:
    return {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
    }[risk_level]


model, metadata = get_model_bundle()
feature_importance_df = pd.DataFrame(metadata.get("feature_importance", [])).head(10)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.4rem;">Customer Churn Prediction App</h1>
        <p style="margin:0;font-size:1.05rem;">
            Enter customer details to estimate churn probability, assign a risk level, and get a recommended retention action.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "This app uses the churn model from the modeling phase and applies the same preprocessing before scoring each customer."
)

with st.sidebar:
    st.header("Model Snapshot")
    st.caption("Current risk bands")
    st.write("Low: 0.00 - 0.30")
    st.write("Medium: 0.30 - 0.70")
    st.write("High: 0.70 - 1.00")

    if not feature_importance_df.empty:
        st.caption("Top churn drivers")
        st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)


st.subheader("Customer Inputs")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", metadata["category_options"]["gender"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", metadata["category_options"]["Partner"])
        dependents = st.selectbox("Dependents", metadata["category_options"]["Dependents"])
        tenure = st.slider("Tenure (months)", min_value=1, max_value=72, value=int(metadata["numeric_defaults"]["tenure"]))

    with col2:
        phone_service = st.selectbox("Phone Service", metadata["category_options"]["PhoneService"])
        multiple_lines = st.selectbox("Multiple Lines", metadata["category_options"]["MultipleLines"])
        internet_service = st.selectbox("Internet Service", metadata["category_options"]["InternetService"])
        online_security = st.selectbox("Online Security", metadata["category_options"]["OnlineSecurity"])
        online_backup = st.selectbox("Online Backup", metadata["category_options"]["OnlineBackup"])

    with col3:
        device_protection = st.selectbox("Device Protection", metadata["category_options"]["DeviceProtection"])
        tech_support = st.selectbox("Tech Support", metadata["category_options"]["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", metadata["category_options"]["StreamingTV"])
        streaming_movies = st.selectbox("Streaming Movies", metadata["category_options"]["StreamingMovies"])
        contract = st.selectbox("Contract", metadata["category_options"]["Contract"])

    col4, col5, col6 = st.columns(3)
    with col4:
        paperless_billing = st.selectbox("Paperless Billing", metadata["category_options"]["PaperlessBilling"])
        payment_method = st.selectbox("Payment Method", metadata["category_options"]["PaymentMethod"])

    with col5:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=float(round(metadata["numeric_defaults"]["MonthlyCharges"], 2)),
            step=0.5,
        )

    with col6:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(round(metadata["numeric_defaults"]["TotalCharges"], 2)),
            step=10.0,
        )

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)


if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
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

    result = score_customer(model, metadata, customer_data)
    probability_pct = result["probability"] * 100
    risk_level = result["risk_level"]
    risk_css = risk_class_name(risk_level)

    st.subheader("Prediction Output")
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Churn Probability</h4>
                <h2>{probability_pct:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Risk Level</h4>
                <h2 class="{risk_css}">{risk_level}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Retention Action</h4>
                <p style="margin-bottom:0;">{result["action_recommendation"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Scored Customer Data")
    scored_row = result["input_df"].copy()
    scored_row["predicted_churn_probability"] = round(probability_pct, 2)
    scored_row["predicted_risk_level"] = risk_level
    st.dataframe(scored_row, use_container_width=True)

st.subheader("How to run")
st.code("streamlit run app.py", language="bash")
