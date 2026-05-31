import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------- 
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Load Model
# ------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/fraud_hgb_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'models/fraud_hgb_model.pkl' exists.")
        return None

model = load_model()

# -----------------------------------------------------------------------------
# App Frontend
# -----------------------------------------------------------------------------
st.title("🛡️ Financial Fraud Detection System")
st.markdown("""
This application uses a **Hybrid Architecture**: 
1. **Rule-Based Engine:** Instantly blocks transactions with invalid accounting math.
2. **AI Engine:** Analyzes valid transactions using a Histogram-based Gradient Boosting Machine to catch sophisticated behavioral fraud.
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    step = st.number_input("Time Step (1 step = 1 hour)", min_value=1, value=1, step=1)
    trans_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=10000.0, step=100.0)

with col2:
    st.subheader("Account Balances")
    oldbalanceOrg = st.number_input("Sender Initial Balance ($)", min_value=0.0, value=20000.0, step=100.0)
    newbalanceOrig = st.number_input("Sender New Balance ($)", min_value=0.0, value=10000.0, step=100.0)
    oldbalanceDest = st.number_input("Recipient Initial Balance ($)", min_value=0.0, value=0.0, step=100.0)
    newbalanceDest = st.number_input("Recipient New Balance ($)", min_value=0.0, value=10000.0, step=100.0)

st.markdown("---")

# -----------------------------------------------------------------------------
# Hybrid Prediction Logic
# -----------------------------------------------------------------------------
if st.button("Verify Transaction", type="primary"):
    if model is not None:
        
        # ==========================================
        # LAYER 1: Hard-coded Rule Check (Math Engine)
        # ==========================================
        expected_sender_balance = oldbalanceOrg - amount
        expected_receiver_balance = oldbalanceDest + amount

        # If the math is wrong, reject immediately. No ML needed.
        if (newbalanceOrig != expected_sender_balance) or (newbalanceDest != expected_receiver_balance):
            st.error("### 🚨 SYSTEM ALERT: Data Discrepancy")
            st.write("Transaction blocked due to invalid accounting balances. The reported balances do not match the expected transfer amounts.")
            
        # ==========================================
        # LAYER 2: Machine Learning Check (AI Engine)
        # ==========================================
        else:
            # 1. Feature Engineering
            errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
            errorBalanceDest = oldbalanceDest + amount - newbalanceDest
            
            # 2. Handle Categorical Encoding (get_dummies with drop_first=True)
            type_CASH_OUT = 1 if trans_type == "CASH_OUT" else 0
            type_DEBIT = 1 if trans_type == "DEBIT" else 0
            type_PAYMENT = 1 if trans_type == "PAYMENT" else 0
            type_TRANSFER = 1 if trans_type == "TRANSFER" else 0
                
            # 3. Compile the feature array
            features = np.array([[
                step, 
                amount, 
                oldbalanceOrg, 
                newbalanceOrig, 
                oldbalanceDest, 
                newbalanceDest,
                errorBalanceOrig, 
                errorBalanceDest, 
                type_CASH_OUT, 
                type_DEBIT, 
                type_PAYMENT, 
                type_TRANSFER
            ]])
            
            # 4. Make Prediction (Removed scaler to match your HGB model)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1] 
            
            # 5. Display Results
            if prediction == 1:
                st.error(f"### 🚨 High Risk Detected by AI")
                st.write("The math is correct, but the model flagged this behavioral pattern as highly suspicious.")
                st.write(f"**Confidence Score:** {probability * 100:.2f}%")
            else:
                st.success(f"### ✅ Secure Transaction")
                st.write("Accounting verified and transaction behavior appears legitimate.")
                st.write(f"**Fraud Risk Score:** {probability * 100:.2f}%")
                
            # Optional: Display the engineered features
            with st.expander("View Engineered Math Errors (Backend Features)"):
                st.write(f"**Sender Balance Error:** ${errorBalanceOrig:.2f}")
                st.write(f"**Recipient Balance Error:** ${errorBalanceDest:.2f}")
                st.caption("Since this transaction passed Layer 1, these errors should strictly evaluate to $0.00.")