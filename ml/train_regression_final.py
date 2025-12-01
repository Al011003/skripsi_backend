# train_regression_final.py - Stick to 4 features but optimized
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

# TETAP 4 features (proven best by your testing)
X_cols = ['tahun', 'revneg', 'netprofneg', 'kode_label']

# ===========================
# 2. Train final NPM model
# ===========================
print("\n>>> Training NPM model with 4 features...")
print(f"Features: {X_cols}")

X_npm = processed_df[X_cols]
y_npm = processed_df['NPM_winsor']

print(f"\nTarget (NPM_winsor) range: [{y_npm.min():.4f}, {y_npm.max():.4f}]")

# Hyperparameters - OPTIMIZED for better generalization
params_npm = {
    "n_estimators": 400,        # Increased for better learning
    "max_depth": 4,             # Reduced to prevent overfitting
    "learning_rate": 0.05,      # Slower learning = better generalization
    "reg_alpha": 1.0,           # L1 regularization
    "reg_lambda": 3.0,          # L2 regularization
    "min_child_weight": 10,     # Prevent overfitting
    "subsample": 0.8,           # Use 80% data per tree
    "colsample_bytree": 0.8,    # Use 80% features per tree
    "gamma": 0.1,               # Min loss reduction for split
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

# Inverse transform untuk lihat error dalam nilai asli
npm_mean = scaler.mean_[2]
npm_std = scaler.scale_[2]

y_npm_original = (y_npm * npm_std) + npm_mean
y_pred_original = (y_pred * npm_std) + npm_mean

mae_original = mean_absolute_error(y_npm_original, y_pred_original)
rmse_original = np.sqrt(mean_squared_error(y_npm_original, y_pred_original))

print("\n--- Training Metrics (ORIGINAL SCALE) ---")
print(f"MAE:  {mae_original:.4f}")
print(f"RMSE: {rmse_original:.4f}")

# Check prediction range
print("\n--- Prediction Range Check ---")
print(f"Training NPM range (scaled): [{y_npm.min():.4f}, {y_npm.max():.4f}]")
print(f"Predicted NPM range (scaled): [{y_pred.min():.4f}, {y_pred.max():.4f}]")

out_of_range = np.sum((y_pred < y_npm.min()) | (y_pred > y_npm.max()))
out_of_range_pct = out_of_range/len(y_pred)*100

print(f"\nPredictions outside training range: {out_of_range}/{len(y_pred)} ({out_of_range_pct:.2f}%)")

if out_of_range_pct > 5:
    print("⚠️  Warning: >5% predictions outside range - consider clipping in production")

# ===========================
# 4. Save model & metadata
# ===========================
print("\n>>> Saving model and metadata...")

# Save model
with open("model_npm.pkl", "wb") as f:
    pickle.dump(model_npm, f)

# Save metadata untuk production use
model_metadata = {
    'features': X_cols,
    'target': 'NPM_winsor',
    'npm_min_scaled': float(y_npm.min()),
    'npm_max_scaled': float(y_npm.max()),
    'npm_min_original': float(y_npm_original.min()),
    'npm_max_original': float(y_npm_original.max()),
    'scaler_mean': float(npm_mean),
    'scaler_std': float(npm_std),
    'metrics': {
        'r2': float(r2),
        'mae_original': float(mae_original),
        'rmse_original': float(rmse_original)
    }
}

with open("model_npm_metadata.pkl", "wb") as f:
    pickle.dump(model_metadata, f)

print("✅ NPM model trained and saved!")
print(f"\nFiles created:")
print(f"  - model_npm.pkl (model)")
print(f"  - model_npm_metadata.pkl (metadata)")
print(f"\nModel Performance:")
print(f"  - R² Score: {r2:.4f}")
print(f"  - MAE: {mae_original:.4f}")
print(f"  - Predictions outside range: {out_of_range_pct:.1f}%")