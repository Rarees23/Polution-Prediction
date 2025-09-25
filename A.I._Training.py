import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import numpy as np
import shap
import matplotlib.pyplot as plt

# -------- Paths --------
data_path = Path(r"C:\Users\Rares\Desktop\A.I\PersonalProject\MyDatasets")
dataset_file = data_path / "merged_weather_aqi_pruned.csv"  

# -------- Load dataset --------
df = pd.read_csv(dataset_file)
df.columns = df.columns.str.strip()

# -------- Define features and target --------
exclude_cols = ['Date', 'AQI_target']
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

# -------- Evaluate XGBoost --------
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print("=== XGBoost Model Performance ===")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²  : {r2:.2f}\n")

# -------- Persistence Baseline --------
y_pred_persistence = y_test.shift(1).dropna()
y_true_persistence = y_test.iloc[1:]

rmse_persistence = np.sqrt(mean_squared_error(y_true_persistence, y_pred_persistence))
mae_persistence = mean_absolute_error(y_true_persistence, y_pred_persistence)
mape_persistence = np.mean(np.abs((y_true_persistence - y_pred_persistence) / y_true_persistence)) * 100
r2_persistence = r2_score(y_true_persistence, y_pred_persistence)

print("=== Persistence Baseline Performance ===")
print(f"RMSE: {rmse_persistence:.2f}")
print(f"MAE : {mae_persistence:.2f}")
print(f"MAPE: {mape_persistence:.2f}%")
print(f"R²  : {r2_persistence:.2f}\n")

# -------- Feature importance (XGBoost) --------
importance = xgb_model.get_booster().get_score(importance_type='gain')
importance_sorted = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
print("Top features by gain:")
for feature, score in list(importance_sorted.items())[:20]:
    print(f"{feature}: {score:.2f}")

# -------- SHAP explanations --------
print("\n=== SHAP Explanations ===")

# Create SHAP explainer for XGBoost
explainer = shap.Explainer(xgb_model, X_train)

# Calculate SHAP values for test set
shap_values = explainer(X_test)

# --- Global explanation ---
print("\nGlobal SHAP summary plot (overall feature impact):")
shap.summary_plot(shap_values, X_test, plot_type="dot")

