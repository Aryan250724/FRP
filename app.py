import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Carbon Intelligence Dashboard", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
model = joblib.load("power_model.pkl")
df = pd.read_csv("anomaly_results.csv")

# -----------------------------
# Custom Styling (UI Upgrade)
# -----------------------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: #FFFFFF;}
.block-container {padding-top: 2rem;}

.card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("Proactive Carbon Intelligence Dashboard")
st.caption("Developed By GROUP-13")

st.info("Predict power output, estimate CO₂ emissions, and detect anomalies in real-time.")

# -----------------------------
# Layout Columns
# -----------------------------
col1, col2 = st.columns([1, 1])

# -----------------------------
# LEFT SIDE → INPUT
# -----------------------------
with col1:
    st.header("Input Parameters")

    at = st.slider("Temperature (AT)", float(df['AT'].min()), float(df['AT'].max()), 25.0)
    v  = st.slider("Vacuum (V)", float(df['V'].min()), float(df['V'].max()), 50.0)
    ap = st.slider("Pressure (AP)", float(df['AP'].min()), float(df['AP'].max()), 1010.0)
    rh = st.slider("Humidity (RH)", float(df['RH'].min()), float(df['RH'].max()), 60.0)

    input_data = pd.DataFrame([[at, v, ap, rh]], columns=['AT','V','AP','RH'])

    pred_pe = model.predict(input_data)[0]
    pred_co2 = pred_pe * 0.82

# -----------------------------
# RIGHT SIDE → RESULTS (CARDS)
# -----------------------------
with col2:
    st.header("Prediction Results")

    c1, c2 = st.columns(2)

    c1.metric("Power Output", f"{pred_pe:.2f}")
    c2.metric("CO₂ Emission", f"{pred_co2:.2f}")

    if pred_co2 > 380:
        st.error("High Emission ⚠️")
    elif pred_co2 > 360:
        st.warning("Moderate Emission ⚡")
    else:
        st.success("Low Emission ✅")

# -----------------------------
# ANOMALY GRAPH
# -----------------------------
st.header("Anomaly Detection")

fig, ax = plt.subplots(figsize=(10,4))

normal = df[df['Anomaly'] == 1]
anomaly = df[df['Anomaly'] == -1]

ax.scatter(normal.index, normal['CO2'], s=5, alpha=0.5, label="Normal")
ax.scatter(anomaly.index, anomaly['CO2'], s=25, color='red', label="Anomaly")

ax.set_title("CO₂ Emission Anomalies")
ax.set_xlabel("Index")
ax.set_ylabel("CO₂")
ax.legend()

st.pyplot(fig)

# Show anomaly count
st.info(f"Total anomalies detected: {len(anomaly)}")

# -----------------------------
# OPTIMIZATION
# -----------------------------
st.header("Optimal Conditions")

opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)

opt_col1.metric("AT", "30.92")
opt_col2.metric("V", "71.84")
opt_col3.metric("AP", "1021.40")
opt_col4.metric("RH", "92.07")

st.success("Minimum CO₂ ≈ 351")