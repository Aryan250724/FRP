import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1️⃣ Load processed dataset
df = pd.read_csv(
    "processed_powerplant.csv",
    parse_dates=['Timestamp'],
    index_col='Timestamp'
)

# 2️⃣ Train-test split (time-based)
train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# 3️⃣ Define features and target
X_train = train[['AT', 'V', 'AP', 'RH']]
y_train = train['CO2']

X_test = test[['AT', 'V', 'AP', 'RH']]
y_test = test['CO2']

# 4️⃣ Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5️⃣ Predict
predictions = model.predict(X_test)

# 6️⃣ Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)