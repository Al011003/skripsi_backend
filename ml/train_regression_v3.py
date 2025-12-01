# train_regression_v2.py - With improved features
from connect import load_financial_data
from preprocessing import preprocess_dataframe
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pickle

# ===========================
# 1. Load & preprocess data
# ===========================
print(">>> Loading and preprocessing data...")
df = load_financial_data()
processed_df, encoder, scaler = preprocess_dataframe(df)

# IMPROVED: Tambah IHSG, LQ45, dan Kuartal sebagai features
X_cols = ['tahun', 'kuartal', 'revneg', 'netprofneg', 'ihsg', 'lq45']

# ===========================
# 2. Train final NPM model
# ===========================
print("\n>>> Training improved NPM model with more features...")
print(f"Features: {X_cols}")

X_npm = processed_df[X_cols]
y_npm = processed_df['NPM_winsor']

# Hyperparameters - adjusted untuk prevent overfitting
params_npm = {
    "n_estimators": 300,
    "max_depth": 5,  # Reduced from 6 to prevent overfitting
    "learning_rate": 0.05,  # Reduced from 0.1 for better generalization
    "reg_alpha": 1.0,  # Increased regularization
    "reg_lambda": 5.0,  # Increased regularization
    "min_child_weight": 10,  # Increased from 5
    "subsample": 0.8,  # Reduced from 1.0
    "colsample_bytree": 0.8,  # Reduced from 1.0
    "random_state": 42,
    "verbosity": 0
}

model_npm = XGBRegressor(**params_npm)
model_npm.fit(X_npm, y_npm)

# ===========================
# 3. Evaluate
# ===========================
print("\n>>> Evaluating model...")
y_pred = model_npm.predict(X_npm)

r2 = r2_score(y_npm, y_pred)
mae = mean_absolute_error(y_npm, y_pred)
rmse = np.sqrt(mean_squared_error(y_npm, y_pred))

print("\n--- Training Metrics (SCALED) ---")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")

# Check prediction range
print("\n--- Prediction Range Check ---")
print(f"Training NPM range: [{y_npm.min():.4f}, {y_npm.max():.4f}]")
print(f"Predicted NPM range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

out_of_range = np.sum((y_pred < y_npm.min()) | (y_pred > y_npm.max()))
print(f"Predictions outside range: {out_of_range}/{len(y_pred)} ({out_of_range/len(y_pred)*100:.2f}%)")

# ===========================
# 4. Save model
# ===========================
print("\n>>> Saving model...")
with open("model_npm_v3.pkl", "wb") as f:
    pickle.dump(model_npm, f)

print("✅ Improved NPM model trained and saved as 'model_npm_v2.pkl'!")
print(f"\nModel improvements:")
print(f"  - Added features: kuartal, ihsg, lq45")
print(f"  - Regularization increased to prevent overfitting")
print(f"  - R² Score: {r2:.4f}")