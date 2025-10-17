import streamlit as st
import pickle

# -------------------- Load Model & Scaler --------------------
model = pickle.load(open("fraud_model_6.pkl", "rb"))
scaler = pickle.load(open("scaler_6.pkl", "rb"))

# -------------------- Page Config --------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

# -------------------- Background Colour --------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #add8e6; /* light blue background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- Title --------------------
st.title("💳 Credit Card Fraud Detection")
st.write("Detect fraudulent transactions instantly")

# -------------------- Input Section --------------------
st.write("### Enter Transaction Details")

time = st.number_input("⏱ Time (seconds)", min_value=0, step=1)
amount = st.number_input("💰 Transaction Amount", min_value=0.0, step=0.01)
v1 = st.number_input("Feature V1", value=0.0, step=0.01)
v2 = st.number_input("Feature V2", value=0.0, step=0.01)
v3 = st.number_input("Feature V3", value=0.0, step=0.01)
v4 = st.number_input("Feature V4", value=0.0, step=0.01)

features = [time, amount, v1, v2, v3, v4]

# -------------------- Prediction Section --------------------
if st.button("🔍 Predict"):
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if pred == 1:
        st.error(f"❌ Fraud Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {prob:.2f})")
