# =========================
# XGBoost AQI Prediction with Cross-Validation
# =========================

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load the dataset
# -------------------------
file_path = r"C:\Users\Rares\Desktop\A.I\PersonalProject\cleaned_dataset_with_season.csv"
df = pd.read_csv(file_path)

# -------------------------
# Separate features and target
# -------------------------
X = df.drop("AQI", axis=1)
y = df["AQI"]

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Initialize XGBoost Regressor
# -------------------------
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# -------------------------
# Train the model
# -------------------------
model.fit(X_train, y_train)

# -------------------------
# Make predictions
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Evaluate model on test set
# -------------------------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Set Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R^2 Score: {r2:.3f}")

# -------------------------
# Cross-Validation
# -------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("\nCross-Validation R² Scores:", cv_scores)
print("Mean CV R²:", np.mean(cv_scores))
print("Std CV R²:", np.std(cv_scores))

# -------------------------
# Feature Importance
# -------------------------
plt.figure(figsize=(10,6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance")
plt.show()
