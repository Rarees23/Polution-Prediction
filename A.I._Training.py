import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import joblib

# -------- Paths --------
data_path = Path(r"C:\Users\Rares\Desktop\A.I\PersonalProject\MyDatasets")
dataset_file = data_path / "merged_weather_aqi_pruned.csv"  # your cleaned file
model_file = data_path / "xgb_aqi_t1_model.pkl"

# -------- Load dataset --------
df = pd.read_csv(dataset_file)
df.columns = df.columns.str.strip()

# -------- Define features and target --------
exclude_cols = ['Date', 'AQI_target']  # Date is not a predictor
X = df.drop(columns=exclude_cols)
y = df['AQI_target']  # t+1 AQI

# -------- Train-test split (time-based) --------
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# -------- Train XGBoost --------
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.85,
    gamma=0.5,
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror'
)

xgb_model.fit(X_train, y_train)

# -------- Evaluate model --------
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"t+1 AQI Prediction — RMSE: {rmse:.2f}, R²: {r2:.2f}")

# -------- Save trained model --------
joblib.dump(xgb_model, model_file)
print(f"Model saved to: {model_file}")

# -------- Optional: Feature importance --------
importance = xgb_model.get_booster().get_score(importance_type='gain')
importance_sorted = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
print("Top features by gain:")
for feature, score in list(importance_sorted.items())[:20]:
    print(f"{feature}: {score:.2f}")
