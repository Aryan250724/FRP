import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
df = pd.read_csv("processed_powerplant.csv")

# -----------------------------
# 2️⃣ Features & target
# -----------------------------
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# -----------------------------
# 3️⃣ Train model
# -----------------------------
model = GradientBoostingRegressor()
model.fit(X, y)

# -----------------------------
# 4️⃣ Define REALISTIC ranges (5%–95%)
# -----------------------------
AT_min, AT_max = df['AT'].quantile(0.05), df['AT'].quantile(0.95)
V_min,  V_max  = df['V'].quantile(0.05),  df['V'].quantile(0.95)
AP_min, AP_max = df['AP'].quantile(0.05), df['AP'].quantile(0.95)
RH_min, RH_max = df['RH'].quantile(0.05), df['RH'].quantile(0.95)

AT_range = np.linspace(AT_min, AT_max, 20)
V_range  = np.linspace(V_min,  V_max,  20)
AP_range = np.linspace(AP_min, AP_max, 20)
RH_range = np.linspace(RH_min, RH_max, 20)

# -----------------------------
# 5️⃣ Optimization (Random Search)
# -----------------------------
best_co2 = float('inf')
best_conditions = None

for _ in range(5000):  # more samples for better result
    at = np.random.choice(AT_range)
    v  = np.random.choice(V_range)
    ap = np.random.choice(AP_range)
    rh = np.random.choice(RH_range)

    input_data = pd.DataFrame([[at, v, ap, rh]], columns=['AT', 'V', 'AP', 'RH'])
    pred_pe = model.predict(input_data)[0]
    pred_co2 = pred_pe * 0.82

    if pred_co2 < best_co2:
        best_co2 = pred_co2
        best_conditions = (at, v, ap, rh)

# -----------------------------
# 6️⃣ Output results
# -----------------------------
print("Optimal Conditions for Minimum CO2:")
print(f"AT = {best_conditions[0]:.2f}")
print(f"V  = {best_conditions[1]:.2f}")
print(f"AP = {best_conditions[2]:.2f}")
print(f"RH = {best_conditions[3]:.2f}")
print(f"\nMinimum CO2 = {best_co2:.2f}")