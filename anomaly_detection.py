import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("processed_powerplant.csv")

# -----------------------------
# 1️⃣ Train Gradient Boosting
# -----------------------------
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

gbr = GradientBoostingRegressor()
gbr.fit(X, y)

# Predict expected power
df['Predicted_PE'] = gbr.predict(X)

# Derive expected CO2
df['Predicted_CO2'] = df['Predicted_PE'] * 0.82

# -----------------------------
# 2️⃣ Compute residual
# -----------------------------
df['Residual'] = df['CO2'] - df['Predicted_CO2']

# -----------------------------
# 3️⃣ Scale residual
# -----------------------------
scaler = StandardScaler()
residual_scaled = scaler.fit_transform(df[['Residual']])

# -----------------------------
# 4️⃣ Isolation Forest
# -----------------------------
iso = IsolationForest(
    contamination=0.02,  # 2% anomalies
    random_state=42
)

df['Anomaly'] = iso.fit_predict(residual_scaled)

# Count anomalies
anomaly_count = len(df[df['Anomaly'] == -1])

print("Total anomalies detected:", anomaly_count)


#calcute percentage
total = len(df)
percentage = (anomaly_count / total) * 100

print("Total anomalies detected:", anomaly_count)
print("Anomaly percentage:", percentage)

# Save results
df.to_csv("anomaly_results.csv", index=False)
print("Anomaly results saved to anomaly_results.csv")