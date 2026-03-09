import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv(
    "processed_powerplant.csv",
    parse_dates=['Timestamp'],
    index_col='Timestamp'
)

# We forecast CO2 only
data = df[['CO2']]

# Scale data (VERY IMPORTANT for LSTM)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences (past 24 hours -> next hour)
def create_sequences(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, 24)

# Train-test split (time-based)
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)