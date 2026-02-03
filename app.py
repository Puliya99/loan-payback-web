import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Loan Payback Predictor", page_icon="💳", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "results" / "loan_payback_nn_model.pkl"
COLS_PATH  = BASE_DIR / "results" / "model_columns.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLS_PATH)
    return model, feature_columns

model, feature_columns = load_artifacts()

st.title("💳 Loan Payback Prediction System")
st.write("Enter applicant details below and click **Predict**.")

with st.form("loan_form"):
    annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
    debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, step=1)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=15000.0, step=500.0)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0, step=0.1)

    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Student", "Retired"])
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Debt consolidation", "Car", "Home", "Education", "Medical", "Vacation", "Business", "Other"]
    )
    grade_subgrade = st.selectbox(
        "Grade Subgrade",
        ["A1","A2","B1","B2","C1","C2","C3","C4","C5","D1","D2","D3","D4","D5","F1","F5"]
    )

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    input_data = {
        "annual_income": annual_income,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "gender": gender,
        "marital_status": marital_status,
        "education_level": education_level,
        "employment_status": employment_status,
        "loan_purpose": loan_purpose,
        "grade_subgrade": grade_subgrade,
    }

    X_input = pd.DataFrame([input_data], columns=feature_columns)

    prob = float(model.predict_proba(X_input)[0][1])

    st.subheader("📊 Prediction Result")
    st.metric("Probability of Loan Payback", f"{prob:.2%}")

    if prob >= 0.70:
        st.success("✅ Low Risk – Likely to Pay Back")
    elif prob >= 0.40:
        st.warning("⚠️ Medium Risk")
    else:
        st.error("❌ High Risk – Likely to Default")
