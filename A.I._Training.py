import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_PATH = Path(r"C:\Users\Rares\Desktop\A.I\PersonalProject\MyDatasets")
FILE_NAME = "merged_weather_aqi_pruned.csv"
RANDOM_SEED = 42
USE_LOG_TARGET = False   # True = train on log(AQI), False = raw AQI

# ---------------- Load Data ----------------
df = pd.read_csv(DATA_PATH / FILE_NAME)
df.columns = df.columns.str.strip()

# Features and target
exclude_cols = ['Date', 'AQI_target', 'log_AQI_target',
                'AQI_target_smooth', 'log_AQI_target_smooth']
X = df.drop(columns=[c for c in exclude_cols if c in df.columns])

if USE_LOG_TARGET:
    y = np.log1p(df['AQI_target'])   # log(1+AQI)
else:
    y = df['AQI_target']

# Train/test split (time-based)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ---------------- Train XGBRegressor ----------------
xgb_model = xgb.XGBRegressor(
    n_estimators=800,        
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.3,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    objective="reg:squarederror"
)

xgb_model.fit(X_train, y_train)

# ---------------- Predictions ----------------
y_pred = xgb_model.predict(X_test)

# Back-transform if log target was used
if USE_LOG_TARGET:
    y_pred = np.expm1(y_pred)
    y_test = np.expm1(y_test)

# ---------------- Metrics ----------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
smape = 100 * np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))
r2 = r2_score(y_test, y_pred)

print("\n=== XGBRegressor Model Performance ===")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"SMAPE: {smape:.2f}%")
print(f"R²   : {r2:.3f}")

# ---------------- Persistence Baseline ----------------
y_pred_persistence = y_test.shift(1).dropna()
y_true_persistence = y_test.iloc[1:]

rmse_persistence = np.sqrt(mean_squared_error(y_true_persistence, y_pred_persistence))
mae_persistence = mean_absolute_error(y_true_persistence, y_pred_persistence)
smape_persistence = 100 * np.mean(
    2 * np.abs(y_true_persistence - y_pred_persistence) /
    (np.abs(y_true_persistence) + np.abs(y_pred_persistence))
)
r2_persistence = r2_score(y_true_persistence, y_pred_persistence)

print("\n=== Persistence Baseline Performance ===")
print(f"RMSE : {rmse_persistence:.2f}")
print(f"MAE  : {mae_persistence:.2f}")
print(f"SMAPE: {smape_persistence:.2f}%")
print(f"R²   : {r2_persistence:.3f}")

# ---------------- True vs Predicted ----------------
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label="True AQI", alpha=0.8)
plt.plot(y_pred, label="Predicted AQI", alpha=0.8)
plt.title("AQI Prediction vs Actual")
plt.xlabel("Time (test set index)")
plt.ylabel("AQI")
plt.legend()
plt.show()

# ---------------- SHAP Explanations ----------------
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

print("\nGlobal SHAP summary plot (overall feature impact):")
shap.summary_plot(shap_values, X_test, plot_type="dot")
