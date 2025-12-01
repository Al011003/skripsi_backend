# validate_model.py - Test semua model
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from connect import load_financial_data
from preprocessing import preprocess_dataframe

print("="*60)
print("MODEL VALIDATION - Testing Accuracy & Predictions")
print("="*60)

# ===========================
# 1. Load Data & Models
# ===========================
print("\n[1/5] Loading data and models...")
df = load_financial_data()
processed_df, encoder, scaler = preprocess_dataframe(df)

with open("model_revneg.pkl", "rb") as f:
    model_revneg = pickle.load(f)

with open("model_netprofneg.pkl", "rb") as f:
    model_netprofneg = pickle.load(f)

with open("model_npm.pkl", "rb") as f:
    model_npm = pickle.load(f)

print("✅ Models loaded!")

# ===========================
# 2. Test REVNEG Model
# ===========================
print("\n[2/5] Testing REVNEG Classifier...")
X_revneg = processed_df[['tahun', 'kuartal', 'kode_label', 'lq45', 'ihsg']]
y_revneg = processed_df['revneg']

y_pred_revneg = model_revneg.predict(X_revneg)

print("\n--- REVNEG Classification Report ---")
print(classification_report(y_revneg, y_pred_revneg))
print("\nConfusion Matrix:")
print(confusion_matrix(y_revneg, y_pred_revneg))

# ===========================
# 3. Test NETPROFNEG Model
# ===========================
print("\n[3/5] Testing NETPROFNEG Classifier...")
X_netprofneg = processed_df[['tahun', 'kuartal', 'kode_label', 'lq45', 'ihsg']]
y_netprofneg = processed_df['netprofneg']

y_pred_netprofneg = model_netprofneg.predict(X_netprofneg)

print("\n--- NETPROFNEG Classification Report ---")
print(classification_report(y_netprofneg, y_pred_netprofneg))
print("\nConfusion Matrix:")
print(confusion_matrix(y_netprofneg, y_pred_netprofneg))

# ===========================
# 4. Test NPM Regression Model
# ===========================
print("\n[4/5] Testing NPM Regression...")
X_npm = processed_df[['tahun', 'revneg', 'netprofneg', 'kode_label']]
y_npm = processed_df['NPM_winsor']

y_pred_npm_scaled = model_npm.predict(X_npm)

# Metrics pada scaled values
r2 = r2_score(y_npm, y_pred_npm_scaled)
mae = mean_absolute_error(y_npm, y_pred_npm_scaled)
rmse = np.sqrt(mean_squared_error(y_npm, y_pred_npm_scaled))

print("\n--- NPM Regression Metrics (SCALED) ---")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")

# Inverse transform untuk melihat error dalam nilai asli
npm_mean = scaler.mean_[2]
npm_std = scaler.scale_[2]

y_npm_original = (y_npm * npm_std) + npm_mean
y_pred_npm_original = (y_pred_npm_scaled * npm_std) + npm_mean

mae_original = mean_absolute_error(y_npm_original, y_pred_npm_original)
rmse_original = np.sqrt(mean_squared_error(y_npm_original, y_pred_npm_original))

print("\n--- NPM Regression Metrics (ORIGINAL SCALE) ---")
print(f"MAE:  {mae_original:.4f}")
print(f"RMSE: {rmse_original:.4f}")

# ===========================
# 5. Check Prediction Range
# ===========================
print("\n[5/5] Checking Prediction Ranges...")

print("\n--- NPM_winsor (SCALED) ---")
print(f"Training range: [{y_npm.min():.4f}, {y_npm.max():.4f}]")
print(f"Prediction range: [{y_pred_npm_scaled.min():.4f}, {y_pred_npm_scaled.max():.4f}]")

print("\n--- NPM_winsor (ORIGINAL) ---")
print(f"Training range: [{y_npm_original.min():.4f}, {y_npm_original.max():.4f}]")
print(f"Prediction range: [{y_pred_npm_original.min():.4f}, {y_pred_npm_original.max():.4f}]")

# Cek berapa prediksi yang di luar range training
out_of_range = np.sum((y_pred_npm_scaled < y_npm.min()) | (y_pred_npm_scaled > y_npm.max()))
print(f"\n⚠️  Predictions outside training range: {out_of_range}/{len(y_pred_npm_scaled)} ({out_of_range/len(y_pred_npm_scaled)*100:.2f}%)")

# ===========================
# 6. Sample Predictions
# ===========================
print("\n" + "="*60)
print("SAMPLE PREDICTIONS (First 10 rows)")
print("="*60)

sample_df = pd.DataFrame({
    'Company': processed_df['kode_perusahaan'].head(10),
    'Tahun': processed_df['tahun'].head(10),
    'Kuartal': processed_df['kuartal'].head(10),
    'RevNeg_Actual': y_revneg.head(10).values,
    'RevNeg_Pred': y_pred_revneg[:10],
    'NetProfNeg_Actual': y_netprofneg.head(10).values,
    'NetProfNeg_Pred': y_pred_netprofneg[:10],
    'NPM_Actual': y_npm_original.head(10).values,
    'NPM_Pred': y_pred_npm_original[:10],
    'NPM_Error': (y_npm_original.head(10).values - y_pred_npm_original[:10])
})

print(sample_df.to_string(index=False))

# ===========================
# 7. Test dengan data baru (simulasi API call)
# ===========================
print("\n" + "="*60)
print("SIMULATION: API Call Prediction")
print("="*60)

# Ambil 1 sample data untuk simulasi
test_idx = 0
test_company = processed_df.iloc[test_idx]

print(f"\nTest Company: {test_company['kode_perusahaan']}")
print(f"Tahun: {test_company['tahun']}, Kuartal: {test_company['kuartal']}")
print(f"IHSG (scaled): {test_company['ihsg']:.4f}")
print(f"LQ45 (scaled): {test_company['lq45']:.4f}")

# Predict RevNeg
X_test_class = pd.DataFrame({
    'tahun': [test_company['tahun']],
    'kuartal': [test_company['kuartal']],
    'kode_label': [test_company['kode_label']],
    'lq45': [test_company['lq45']],
    'ihsg': [test_company['ihsg']]
})

revneg_pred = model_revneg.predict(X_test_class)[0]
netprofneg_pred = model_netprofneg.predict(X_test_class)[0]

print(f"\nRevNeg Prediction: {revneg_pred} (Actual: {test_company['revneg']})")
print(f"NetProfNeg Prediction: {netprofneg_pred} (Actual: {test_company['netprofneg']})")

# Predict NPM
X_test_reg = pd.DataFrame({
    'tahun': [test_company['tahun']],
    'revneg': [revneg_pred],
    'netprofneg': [netprofneg_pred],
    'kode_label': [test_company['kode_label']]
})

npm_pred_scaled = model_npm.predict(X_test_reg)[0]
npm_pred_original = (npm_pred_scaled * npm_std) + npm_mean

print(f"\nNPM Prediction (scaled): {npm_pred_scaled:.4f}")
print(f"NPM Prediction (original): {npm_pred_original:.4f}")
print(f"NPM Actual (original): {(test_company['NPM_winsor'] * npm_std) + npm_mean:.4f}")
print(f"Error: {abs(npm_pred_original - ((test_company['NPM_winsor'] * npm_std) + npm_mean)):.4f}")

print("\n" + "="*60)
print("VALIDATION COMPLETE!")
print("="*60)