from connect import load_financial_data
from preprocessing import preprocess_dataframe
from xgboost import XGBClassifier
import pickle

# ===========================
# 1. Load & preprocess data
# ===========================
df = load_financial_data()
processed_df, encoder, scaler = preprocess_dataframe(df)

X_cols = ['tahun', 'kuartal', 'kode_label', 'lq45', 'ihsg']

# ===========================
# 2. Train final REVNEG model
# ===========================
print(">>> Training final model - REVNEG")
X_revneg = processed_df[X_cols]
y_revneg = processed_df['revneg']

params_revneg = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "subsample": 0.6,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 12.148148148148149,
    "random_state": 42,
    "verbosity": 0
}

model_revneg = XGBClassifier(**params_revneg)
model_revneg.fit(X_revneg, y_revneg)

with open("model_revneg.pkl", "wb") as f:
    pickle.dump(model_revneg, f)

print("✅ REVNEG model trained and saved!")

# ===========================
# 3. Train final NETPROFNEG model
# ===========================
print("\n>>> Training final model - NETPROFNEG")
X_netprofneg = processed_df[X_cols]
y_netprofneg = processed_df['netprofneg']

params_netprofneg = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.1580547112462005,
    "random_state": 42,
    "verbosity": 0
}

model_netprofneg = XGBClassifier(**params_netprofneg)
model_netprofneg.fit(X_netprofneg, y_netprofneg)

with open("model_netprofneg.pkl", "wb") as f:
    pickle.dump(model_netprofneg, f)

print("✅ NETPROFNEG model trained and saved!")
