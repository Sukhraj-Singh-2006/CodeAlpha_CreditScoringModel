import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Prediction System")
st.write("Predict whether a loan applicant is a credit risk.")

model = joblib.load("model/credit_model.pkl")

# Inputs

person_age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

person_income = st.number_input(
    "Annual Income",
    min_value=0,
    value=50000
)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

person_emp_length = st.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    value=5.0
)

loan_intent = st.selectbox(
    "Loan Intent",
    [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000
)

loan_int_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    value=10.0
)

loan_percent_income = st.number_input(
    "Loan Percent Income",
    min_value=0.0,
    value=0.20
)

cb_person_default_on_file = st.selectbox(
    "Previous Default",
    ["N", "Y"]
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length",
    min_value=0,
    value=5
)

# Encoding
home_map = {
    "RENT": 3,
    "OWN": 2,
    "MORTGAGE": 0,
    "OTHER": 1
}

intent_map = {
    "DEBTCONSOLIDATION": 0,
    "EDUCATION": 1,
    "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3,
    "PERSONAL": 4,
    "VENTURE": 5
}

grade_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
}

default_map = {
    "N": 0,
    "Y": 1
}

if st.button("Predict Credit Risk"):

    input_df = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": home_map[person_home_ownership],
        "person_emp_length": person_emp_length,
        "loan_intent": intent_map[loan_intent],
        "loan_grade": grade_map[loan_grade],
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": default_map[cb_person_default_on_file],
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if probability < 0.30:
        st.success(f"🟢 Low Risk ({probability:.2%})")

    elif probability < 0.70:
        st.warning(f"🟡 Medium Risk ({probability:.2%})")

    else:
        st.error(f"🔴 High Risk ({probability:.2%})")

    st.progress(float(probability))

    st.write(f"Risk Probability: **{probability:.2%}**")