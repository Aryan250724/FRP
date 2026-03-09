import pandas as pd

# 1. Load original dataset
df = pd.read_csv("Folds5x2_pp.csv")

# 2. Create synthetic timestamp
df['Timestamp'] = pd.date_range(
    start='2020-01-01',
    periods=len(df),
    freq='h'
)

# 3. Sort and set index
df = df.sort_values('Timestamp')
df.set_index('Timestamp', inplace=True)

# 4. ADD CO2 COLUMN HERE  ← THIS IS WHERE IT GOES
df['CO2'] = df['PE'] * 0.82

# 5. Check result
print(df.head())
print(df.info())

# 6. Save processed dataset
df.to_csv("processed_powerplant.csv")