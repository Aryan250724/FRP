import pandas as pd
import matplotlib.pyplot as plt

# Load anomaly results
df = pd.read_csv("anomaly_results.csv")

# Select anomalies
anomalies = df[df['Anomaly'] == -1]

plt.figure(figsize=(12,6))

plt.scatter(df.index, df['CO2'], label="Normal", s=10)
plt.scatter(anomalies.index, anomalies['CO2'], color='red', label="Anomaly", s=25)

plt.title("CO2 Emission Anomaly Detection")
plt.xlabel("Observation Index")
plt.ylabel("CO2 Emission")

plt.legend()
plt.show()