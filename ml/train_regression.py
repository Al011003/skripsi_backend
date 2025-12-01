# train_regression.py
from connect import load_financial_data
from preprocessing import preprocess_dataframe
from xgboost import XGBRegressor
import pickle

# ===========================
# 1. Load & preprocess data
# ===========================
print(">>> Loading and preprocessing data...")
df = load_financial_data()
processed_df, encoder, scaler = preprocess_dataframe(df)

# Feature columns (lowercase semua)
X_cols = ['tahun', 'revneg', 'netprofneg', 'kode_label']

# ===========================
# 2. Train final NPM model
# ===========================
print("\n>>> Training final model - NPM")
X_npm = processed_df[X_cols]
y_npm = processed_df['NPM_winsor']  # Huruf besar NPM!

params_npm = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "min_child_weight": 5,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "verbosity": 0
}

model_npm = XGBRegressor(**params_npm)
model_npm.fit(X_npm, y_npm)

with open("model_npm.pkl", "wb") as f:
    pickle.dump(model_npm, f)

print("✅ NPM model trained and saved!")