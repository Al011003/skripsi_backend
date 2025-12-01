# save_encoder.py
# Jalanin sekali aja buat save encoder
import pickle
from connect import load_financial_data
from preprocessing import preprocess_dataframe

print("Loading data and preprocessing...")
df = load_financial_data()
processed_df, encoder, scaler = preprocess_dataframe(df)

# Save encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Label encoder saved to ml/label_encoder.pkl")
print(f"Total companies: {len(encoder.classes_)}")
print(f"Companies: {encoder.classes_[:10]}...")  # Show first 10