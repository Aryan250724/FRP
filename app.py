import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_autorefresh import st_autorefresh

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Carbon Intelligence Dashboard",
    layout="wide"
)

# =========================================================
# AUTO REFRESH
# =========================================================

st_autorefresh(interval=3000, key="refresh")

# =========================================================
# LOAD MODEL & DATA
# =========================================================

model = joblib.load("power_model.pkl")

df = pd.read_csv(
    "final_carbon_monitoring_dataset.csv"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: white;
}

.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("Proactive Caerbon Intelligence Dashboard")

st.caption(
    "Developed by Group 13 "
)

st.info(
    "Machine learning system for power prediction, CO₂ estimation, anomaly detection, and operational optimization."
)

# =========================================================
# MODEL INFO
# =========================================================

st.success(
    "Prediction Engine: Gradient Boosting Regressor"
)

# =========================================================
# INPUT + RESULT SECTION
# =========================================================

col1, col2 = st.columns([1,1])

# -----------------------------
# INPUTS
# -----------------------------

with col1:

    st.header("Input Parameters")

    at = st.slider(
        "Ambient Temperature (AT)",
        float(df['AT'].min()),
        float(df['AT'].max()),
        25.0
    )

    v = st.slider(
        "Exhaust Vacuum (V)",
        float(df['V'].min()),
        float(df['V'].max()),
        50.0
    )

    ap = st.slider(
        "Ambient Pressure (AP)",
        float(df['AP'].min()),
        float(df['AP'].max()),
        1010.0
    )

    rh = st.slider(
        "Relative Humidity (RH)",
        float(df['RH'].min()),
        float(df['RH'].max()),
        60.0
    )

# =========================================================
# PREDICTION
# =========================================================

input_data = pd.DataFrame(
    [[at, v, ap, rh]],
    columns=['AT', 'V', 'AP', 'RH']
)

pred_pe = model.predict(input_data)[0]

pred_co2 = pred_pe * 0.82

# =========================================================
# RESULTS
# =========================================================

with col2:

    st.header("Prediction Results")

    k1, k2 = st.columns(2)

    k1.metric(
        "Predicted Power Output",
        f"{pred_pe:.2f}"
    )

    k2.metric(
        "Estimated CO₂",
        f"{pred_co2:.2f}"
    )

    if pred_co2 > 380:

        st.error("High Emission Level ⚠️")

    elif pred_co2 > 360:

        st.warning("Moderate Emission Level ⚡")

    else:

        st.success("Low Emission Level ✅")

# =========================================================
# LIVE SYSTEM MONITORING
# =========================================================

st.header("Live System Monitoring")

st.write(
    "Simulated real-time operational readings using historical plant data."
)

latest = df.sample(1).iloc[0]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Current CO₂",
    f"{latest['CO2']:.2f}"
)

c2.metric(
    "Current Power",
    f"{latest['PE']:.2f}"
)

c3.metric(
    "Current Temperature",
    f"{latest['AT']:.2f}"
)

if latest['Anomaly'] == -1:

    c4.error("Anomaly Detected")

else:

    c4.success("System Normal")

# =========================================================
# CO2 TREND + ANOMALIES
# =========================================================

st.header("CO₂ Emission Trend & Anomaly Detection")

sample_df = df.head(300)

fig1, ax1 = plt.subplots(figsize=(14,5))

# CO2 Trend
ax1.plot(
    sample_df.index,
    sample_df['CO2'],
    color='blue',
    linewidth=2,
    label='CO₂ Trend'
)

# Anomaly Points
anomaly_points = sample_df[
    sample_df['Anomaly'] == -1
]

ax1.scatter(
    anomaly_points.index,
    anomaly_points['CO2'],
    color='red',
    s=120,
    label='Detected Anomaly',
    zorder=5
)

ax1.set_title(
    "CO₂ Emission Trend with Detected Anomalies"
)

ax1.set_xlabel("Time Index")

ax1.set_ylabel("CO₂ Emission")

ax1.legend()

ax1.grid(True, alpha=0.3)

st.pyplot(fig1)

st.info(
    "Red points indicate abnormal emission behavior detected using Isolation Forest."
)

# =========================================================
# ACTUAL vs PREDICTED
# =========================================================

st.header("Actual vs Predicted Power Output")

sample_df2 = df.head(80)

fig2, ax2 = plt.subplots(figsize=(14,5))

ax2.plot(
    sample_df2.index,
    sample_df2['PE'],
    linewidth=3,
    label='Actual Output'
)

ax2.plot(
    sample_df2.index,
    sample_df2['Predicted_PE'],
    linestyle='--',
    linewidth=3,
    label='Predicted Output'
)

ax2.set_title(
    "Model Prediction Performance"
)

ax2.set_xlabel("Time Index")

ax2.set_ylabel("Power Output")

ax2.legend()

ax2.grid(True, alpha=0.3)

st.pyplot(fig2)

st.info(
    "The predicted curve closely follows the actual curve, indicating strong model performance."
)

# =========================================================
# HEATMAP
# =========================================================

st.header("Feature Correlation Heatmap")

fig3, ax3 = plt.subplots(figsize=(8,5))

corr = df[
    ['AT', 'V', 'AP', 'RH', 'PE', 'CO2']
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    ax=ax3
)

st.pyplot(fig3)

# =========================================================
# OPTIMIZATION
# =========================================================

st.header("Optimal Conditions for Minimum CO₂")

o1, o2, o3, o4 = st.columns(4)

o1.metric("AT", "30.92")

o2.metric("V", "71.84")

o3.metric("AP", "1021.40")

o4.metric("RH", "92.07")

st.success("Minimum CO₂ ≈ 351")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Simulated real-time intelligent carbon monitoring system using machine learning."
)