import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained model and scaler
# Note: Ensure these files exist in your 'models' directory as per your notebook
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# --- Page Configuration ---
st.set_page_config(
    page_title="SecurePay | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# --- Custom CSS for a Modern Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007BFF;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar / Navigation ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/shield.png")
    st.title("SecurePay AI")
    st.info("Our AI analyzes transaction patterns to keep your account safe.")

# --- Main Interface ---
st.title("💳 Transaction Security Scanner")
st.write("Complete the details below to verify the safety of your transfer.")

# Create two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sender Details")
    amount = st.number_input("Transfer Amount ($)", min_value=0.0, format="%.2f")
    oldbalanceOrg = st.number_input("Current Account Balance ($)", min_value=0.0, format="%.2f")
    # Automatically calculate the new balance for the user
    newbalanceOrig = oldbalanceOrg - amount
    st.caption(f"Estimated Remaining Balance: ${newbalanceOrig:,.2f}")

with col2:
    st.subheader("Transaction Metadata")
    transaction_type = st.selectbox(
        "Payment Method", 
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    )
    step = st.slider("Transaction Hour (1-744)", 1, 744, 1)

# --- Advanced / Hidden Receiver Details ---
# We keep these as expandable or set smart defaults so the user isn't annoyed
with st.expander("Receiver Security Details (Optional)"):
    st.write("Information about the recipient bank account.")
    oldbalanceDest = st.number_input("Recipient Initial Balance", min_value=0.0, value=0.0)
    newbalanceDest = oldbalanceDest + amount 

# --- Prediction Logic ---
if st.button("Verify Transaction"):
    # 1. Replicate Feature Engineering from your Notebook
    balance_diff_orig = oldbalanceOrg - newbalanceOrig - amount
    balance_diff_dest = newbalanceDest - oldbalanceDest - amount
    orig_balance_change = oldbalanceOrg - newbalanceOrig
    dest_balance_change = newbalanceDest - oldbalanceDest
    
    # 2. Replicate One-Hot Encoding (drop_first=True)
    type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
    type_DEBIT = 1 if transaction_type == "DEBIT" else 0
    type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
    type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

    # 3. Format exactly for the Scaler (11 Features)
    # Order: step, amount, isFlaggedFraud, diff_orig, diff_dest, change_orig, change_dest, dummies...
    features = [
        step, amount, 0, # isFlaggedFraud default to 0
        balance_diff_orig, balance_diff_dest, 
        orig_balance_change, dest_balance_change,
        type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
    ]

    # Process and Predict
    try:
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        
        st.divider()
        if prediction[0] == 1:
            st.error("### 🚨 High Risk Detected")
            st.warning("This transaction matches patterns associated with fraudulent activity. We recommend manual review.")
        else:
            st.success("### ✅ Secure Transaction")
            st.balloons()
            st.write("The AI has verified this transaction as low-risk.")
            
    except Exception as e:
        st.error(f"Error processing prediction: {e}")

# --- Footer ---
st.caption("SecurePay AI v1.0 | Powered by Logistic Regression & PaySim Dataset")