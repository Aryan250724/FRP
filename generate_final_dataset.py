import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("processed_powerplant.csv")

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model = joblib.load("power_model.pkl")

# =====================================================
# FEATURES
# =====================================================

X = df[['AT', 'V', 'AP', 'RH']]

# =====================================================
# PREDICT POWER OUTPUT
# =====================================================

df['Predicted_PE'] = model.predict(X)

# =====================================================
# ESTIMATE CO2
# =====================================================

df['CO2'] = df['Predicted_PE'] * 0.82

# =====================================================
# CALCULATE RESIDUALS
# =====================================================

df['Residual'] = df['PE'] - df['Predicted_PE']

# =====================================================
# ANOMALY DETECTION
# =====================================================

iso = IsolationForest(
    contamination=0.02,
    random_state=42
)

df['Anomaly'] = iso.fit_predict(
    df[['Residual']]
)

# =====================================================
# SAVE FINAL DATASET
# =====================================================

df.to_csv(
    "final_carbon_monitoring_dataset.csv",
    index=False
)

print("Final intelligent dataset created successfully!")