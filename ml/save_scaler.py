# save_scaler.py - Jalanin sekali
import pickle
from connect import load_financial_data
from preprocessing import preprocess_dataframe

df = load_financial_data()
processed_df, encoder, scaler = preprocess_dataframe(df)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Scaler saved!")
print(f"Scaler features: {scaler.feature_names_in_}")
print(f"Scaler mean: {scaler.mean_}")
print(f"Scaler scale: {scaler.scale_}")