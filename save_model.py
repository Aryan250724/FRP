import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load dataset
df = pd.read_csv("processed_powerplant.csv")

# Features and target
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Train model
model = GradientBoostingRegressor()
model.fit(X, y)

# Save model
joblib.dump(model, "power_model.pkl")

print("Model saved successfully!")