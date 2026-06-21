import joblib

model = joblib.load("power_model.pkl")
joblib.dump(model, "power_model.pkl")

print("Model re-saved successfully")