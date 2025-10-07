import pandas as pd
import numpy as np
from pathlib import Path

# -------- Paths --------
data_path = Path(r"C:\Users\Rares\Desktop\A.I\PersonalProject\MyDatasets")
file_path = data_path / "merged_weather_aqi_pruned.csv"

# -------- Load dataset --------
df = pd.read_csv(file_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.columns = df.columns.str.strip()

# --------------------------
# 0. Core lag features (strictly past)
# --------------------------
df['AQI_lag1'] = df['AQI_target'].shift(1)
df['AQI_lag2'] = df['AQI_target'].shift(2)
df['AQI_lag3'] = df['AQI_target'].shift(3)
df['AQI_lag7'] = df['AQI_target'].shift(7)
df['AQI_lag14'] = df['AQI_target'].shift(14)
df['AQI_lag30'] = df['AQI_target'].shift(30)

pollutants = {
    'Pm2.5': [1, 2, 3, 7, 14, 30],
    'Pm10': [1, 2, 3, 7, 14, 30],
    'No2': [1, 2, 7, 14],
    'So2': [1, 7],
    'Co': [1, 7]
}
for base, lags in pollutants.items():
    if base in df.columns:
        for lag in lags:
            df[f'{base}_lag{lag}'] = df[base].shift(lag)

# --------------------------
# 1. Rolling averages, std, min/max
# --------------------------
windows = [3, 7, 14, 30]
for base in ['Pm2.5', 'Pm10', 'No2', 'So2', 'Co']:
    if base in df.columns:
        lagged = df[base].shift(1)
        for w in windows:
            df[f'{base}_{w}d_avg'] = lagged.rolling(w, min_periods=1).mean()
            df[f'{base}_{w}d_std'] = lagged.rolling(w, min_periods=1).std()
            df[f'{base}_{w}d_max'] = lagged.rolling(w, min_periods=1).max()
            df[f'{base}_{w}d_min'] = lagged.rolling(w, min_periods=1).min()

# --------------------------
# 2. Date-based features
# --------------------------
df['day_of_year'] = df['Date'].dt.dayofyear
df['day_of_week'] = df['Date'].dt.dayofweek
df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# --------------------------
# 3. Ratios & trends
# --------------------------
df['Pm2.5_to_Pm10'] = df['Pm2.5_lag1'] / (df['Pm10_lag1'] + 1e-6)
df['NO2_to_SO2'] = df['No2_lag1'] / (df['So2_lag1'] + 1e-6)
df['CO_to_PM25'] = df['Co_lag1'] / (df['Pm2.5_lag1'] + 1e-6)

# Trend: 7d avg - 30d avg
if 'Pm2.5_7d_avg' in df.columns and 'Pm2.5_30d_avg' in df.columns:
    df['Pm2.5_trend'] = df['Pm2.5_7d_avg'] - df['Pm2.5_30d_avg']
if 'Pm10_7d_avg' in df.columns and 'Pm10_30d_avg' in df.columns:
    df['Pm10_trend'] = df['Pm10_7d_avg'] - df['Pm10_30d_avg']

# --------------------------
# 4. Logs
# --------------------------
for col in ['Pm2.5_lag1','Pm10_lag1','AQI_lag1']:
    if col in df.columns:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

# --------------------------
# 5. Outlier removal (basic IQR)
# --------------------------
for col in ['AQI_target','Pm2.5_lag1','Pm10_lag1','No2_lag1','So2_lag1','Co_lag1']:
    if col in df.columns:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR  # looser than before
        df = df[(df[col] >= lower) & (df[col] <= upper)]

df = df.dropna().reset_index(drop=True)

# --------------------------
# 6. Log-transform target
# --------------------------
df['log_AQI_target'] = np.log1p(df['AQI_target'].clip(lower=0))

# --------------------------
# 7. Save
# --------------------------
df.to_csv(file_path, index=False)
print(f"✅ Fully processed dataset saved to {file_path}")
