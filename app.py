# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Title
st.title("📊 Customer Churn Prediction System")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ churn_model.pkl not found. Please run train_model.py first.")
    st.stop()

# Input form
st.subheader("🔍 Enter Customer Details to Predict Churn")

with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Prepare data for prediction
    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical data
    for col in ['gender', 'Partner', 'Dependents']:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # Ensure all required features exist
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Show result
    st.subheader("📢 Prediction Result:")
    if prediction == 1:
        st.error(f"❌ The customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The customer is likely to stay (Probability: {probability:.2f})")

    # 🔍 Rule-based churn explanation
    st.subheader("🧠 Why this prediction? (Rule-based explanation)")

    reasons = []

    if tenure < 12:
        reasons.append("Low tenure (new customer)")
    if monthly_charges > 80:
        reasons.append("High monthly charges")
    if partner == "No":
        reasons.append("No partner")
    if dependents == "No":
        reasons.append("No dependents")
    if senior == 1:
        reasons.append("Senior citizen")

    if reasons:
        st.write("This prediction may be influenced by:")
        for r in reasons:
            st.markdown(f"- ✅ {r}")
    else:
        st.write("The customer shows stable features and low churn risk.")

# Divider
st.divider()

# 📊 ML-based Feature Importance
with st.expander("📊 Feature Importance (Model Insights)"):
    try:
        coef_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': np.abs(model.coef_[0])
        }).sort_values(by="Importance", ascending=False)

        st.write("The model assigns higher weights to these features:")
        st.dataframe(coef_df)

        # Optional: Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="Importance", y="Feature", data=coef_df.head(10), palette="Blues_r")
        ax.set_title("Top 10 Most Influential Features")
        st.pyplot(fig)

    except Exception as e:
        st.warning("Feature importance could not be displayed. This may not be a linear model.")
        st.error(e)

# 📊 Additional Visualizations
st.subheader("📈 Churn Data Insights & Visualizations")

try:
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # 1. Churn Distribution
    with st.expander("📊 Churn Distribution (Bar Chart)"):
        fig1 = plt.figure(figsize=(6,4))
        sns.countplot(x='Churn', data=df, palette='Set2')
        st.pyplot(fig1)

    # 2. Tenure vs Churn
    with st.expander("📈 Tenure vs Churn (Stacked Histogram)"):
        fig2 = plt.figure(figsize=(8,5))
        sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30, palette='husl')
        st.pyplot(fig2)

    # 3. Monthly Charges KDE Plot
    with st.expander("💰 Monthly Charges vs Churn (KDE Plot)"):
        fig3 = plt.figure(figsize=(8,5))
        sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True)
        st.pyplot(fig3)

    # 4. Contract Type Pie Chart
    with st.expander("🧾 Contract Type Distribution (Pie Chart)"):
        contract_counts = df['Contract'].value_counts()
        fig4, ax = plt.subplots()
        ax.pie(contract_counts, labels=contract_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        ax.axis('equal')
        st.pyplot(fig4)

    # 5. Correlation Heatmap
    with st.expander("📌 Correlation Heatmap"):
        encoded_df = df.copy()
        for col in encoded_df.select_dtypes(include='object').columns:
            encoded_df[col] = encoded_df[col].astype('category').cat.codes
        fig5 = plt.figure(figsize=(10,6))
        sns.heatmap(encoded_df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(fig5)

except Exception as e:
    st.error(f"Error loading visualizations: {e}")
